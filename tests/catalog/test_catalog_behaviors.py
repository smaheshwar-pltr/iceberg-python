# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Consolidated behavior tests for InMemoryCatalog and SqlCatalog.
"""

import json
import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic_core import ValidationError
from pytest_lazy_fixtures import lf
from sqlalchemy.exc import IntegrityError

from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import (
    CommitFailedException,
    CommitStateUnknownException,
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchNamespaceError,
    NoSuchTableError,
    TableAlreadyExistsError,
    ValidationException,
)
from pyiceberg.exceptions import ValidationError as IcebergValidationError
from pyiceberg.io.pyarrow import _dataframe_to_data_files, schema_to_pyarrow
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table, TableProperties
from pyiceberg.table.snapshots import Operation
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, NullOrder, SortDirection, SortField, SortOrder
from pyiceberg.table.update import AddSchemaUpdate, SetCurrentSchemaUpdate
from pyiceberg.table.update.schema import UpdateSchema
from pyiceberg.table.update.snapshot import ManageSnapshots
from pyiceberg.transforms import BucketTransform, DayTransform, IdentityTransform, VoidTransform
from pyiceberg.typedef import Identifier, Record
from pyiceberg.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    StructType,
    TimestampNanoType,
)


# Name parsing tests
def test_namespace_from_tuple() -> None:
    identifier = ("com", "organization", "department", "my_table")
    namespace_from = Catalog.namespace_from(identifier)
    assert namespace_from == ("com", "organization", "department")


def test_namespace_from_str() -> None:
    identifier = "com.organization.department.my_table"
    namespace_from = Catalog.namespace_from(identifier)
    assert namespace_from == ("com", "organization", "department")


def test_name_from_tuple() -> None:
    identifier = ("com", "organization", "department", "my_table")
    name_from = Catalog.table_name_from(identifier)
    assert name_from == "my_table"


def test_name_from_str() -> None:
    identifier = "com.organization.department.my_table"
    name_from = Catalog.table_name_from(identifier)
    assert name_from == "my_table"


# Create table tests
def test_create_table(catalog: Catalog, test_table_identifier: Identifier, table_schema_simple: Schema) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_simple)
    loaded = catalog.load_table(test_table_identifier)
    assert loaded.name() == table.name()
    assert loaded.metadata_location == table.metadata_location


def test_create_table_keeps_write_default(catalog: Catalog, test_table_identifier: Identifier) -> None:
    catalog.create_namespace(Catalog.namespace_from(test_table_identifier))
    schema = Schema(NestedField(field_id=1, name="id", field_type=LongType(), required=False, write_default=2))

    table = catalog.create_table(test_table_identifier, schema)

    assert catalog.load_table(test_table_identifier).schema() == table.schema() == schema


_STRUCT_WITH_INITIAL_DEFAULT = StructType(
    NestedField(field_id=100, name="child", field_type=LongType(), required=False, initial_default=7, write_default=9)
)


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(lambda update: update.add_column("extra", LongType(), default_value=7), id="add-column"),
        pytest.param(lambda update: update.add_column("extra", _STRUCT_WITH_INITIAL_DEFAULT), id="add-struct-column"),
        pytest.param(
            lambda update: update.union_by_name(
                Schema(NestedField(field_id=2, name="extra", field_type=_STRUCT_WITH_INITIAL_DEFAULT, required=False))
            ),
            id="union-by-name",
        ),
    ],
)
def test_update_schema_rejects_initial_default_before_v3(
    catalog: Catalog, test_table_identifier: Identifier, update: Callable[[UpdateSchema], UpdateSchema]
) -> None:
    catalog.create_namespace(Catalog.namespace_from(test_table_identifier))
    table = catalog.create_table(test_table_identifier, Schema(NestedField(field_id=1, name="id", field_type=LongType())))

    with pytest.raises(ValueError, match="Cannot add column extra: initial defaults are only supported in format version 3"):
        update(table.update_schema()).commit()

    assert catalog.load_table(test_table_identifier).schema() == table.schema()


def test_update_schema_keeps_write_default_before_v3(catalog: Catalog, test_table_identifier: Identifier) -> None:
    catalog.create_namespace(Catalog.namespace_from(test_table_identifier))
    table = catalog.create_table(test_table_identifier, Schema(NestedField(field_id=1, name="id", field_type=LongType())))
    struct_type = StructType(NestedField(field_id=100, name="child", field_type=LongType(), required=False, write_default=9))

    table.update_schema().add_column("extra", struct_type).commit()

    child = catalog.load_table(test_table_identifier).schema().find_field("extra.child")
    assert (child.initial_default, child.write_default) == (None, 9)


@pytest.mark.parametrize("format_version", [1, 2])
def test_union_by_name_keeps_write_default_before_v3(catalog: Catalog, format_version: int) -> None:
    identifier = f"default.union_by_name_write_default_v{format_version}"
    catalog.create_namespace("default")
    existing_field = NestedField(field_id=2, name="a", field_type=LongType(), required=False)
    table = catalog.create_table(
        identifier,
        Schema(NestedField(field_id=1, name="s", field_type=StructType(existing_field), required=False)),
        properties={"format-version": str(format_version)},
    )
    new_schema = Schema(
        NestedField(
            field_id=1,
            name="s",
            field_type=StructType(
                existing_field, NestedField(field_id=3, name="b", field_type=LongType(), required=False, write_default=8)
            ),
            required=False,
        ),
        NestedField(field_id=4, name="c", field_type=LongType(), required=False, write_default=9),
    )

    table.update_schema().union_by_name(new_schema).commit()

    assert catalog.load_table(identifier).schema().as_struct() == new_schema.as_struct()


@pytest.mark.parametrize("format_version", [1, 2])
def test_union_by_name_rejects_initial_default_before_v3(catalog: Catalog, format_version: int) -> None:
    identifier = f"default.union_by_name_initial_default_v{format_version}"
    catalog.create_namespace("default")
    existing_field = NestedField(field_id=2, name="a", field_type=LongType(), required=False)
    table = catalog.create_table(
        identifier,
        Schema(NestedField(field_id=1, name="s", field_type=StructType(existing_field), required=False)),
        properties={"format-version": str(format_version)},
    )
    added_field = NestedField(field_id=3, name="b", field_type=LongType(), required=False, initial_default=7)

    for column, new_schema in [
        ("b", Schema(*table.schema().fields, added_field)),
        ("s.b", Schema(NestedField(field_id=1, name="s", field_type=StructType(existing_field, added_field), required=False))),
    ]:
        with pytest.raises(ValueError, match=f"Cannot add column {column}: initial defaults are only supported in format"):
            table.update_schema().union_by_name(new_schema).commit()

    assert catalog.load_table(identifier).schema() == table.schema()


def test_create_table_if_not_exists_duplicated_table(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table1 = catalog.create_table(test_table_identifier, table_schema_nested)
    table2 = catalog.create_table_if_not_exists(test_table_identifier, table_schema_nested)
    assert table1.name() == table2.name()


def test_create_table_raises_error_when_table_already_exists(
    catalog: Catalog, test_table_identifier: Identifier, table_schema_nested: Schema
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    catalog.create_table(test_table_identifier, table_schema_nested)
    with pytest.raises(TableAlreadyExistsError):
        catalog.create_table(test_table_identifier, table_schema_nested)


def test_table_exists(catalog: Catalog, test_table_identifier: Identifier, table_schema_nested: Schema) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    catalog.create_table(test_table_identifier, table_schema_nested, properties={"format-version": "2"})
    assert catalog.table_exists(test_table_identifier)


def test_table_exists_on_table_not_found(catalog: Catalog, test_table_identifier: Identifier) -> None:
    assert not catalog.table_exists(test_table_identifier)


def test_create_table_raises_error_when_namespace_does_not_exist(catalog: Catalog, table_schema_simple: Schema) -> None:
    with pytest.raises(NoSuchNamespaceError):
        catalog.create_table(("non_existent_ns", "table"), table_schema_simple)


def test_table_raises_error_on_table_not_found(catalog: Catalog, test_table_identifier: Identifier) -> None:
    identifier_str = ".".join(test_table_identifier)
    with pytest.raises(NoSuchTableError, match=f"Table does not exist: {identifier_str}"):
        catalog.load_table(test_table_identifier)


def test_create_table_without_namespace(catalog: Catalog, table_schema_nested: Schema, table_name: str) -> None:
    with pytest.raises(NoSuchNamespaceError):
        catalog.create_table(table_name, table_schema_nested)


@pytest.mark.parametrize("format_version", [1, 2])
def test_create_table_transaction(catalog: Catalog, format_version: int) -> None:
    identifier = f"default.arrow_create_table_transaction_{catalog.name}_{format_version}"
    try:
        catalog.create_namespace("default")
    except NamespaceAlreadyExistsError:
        pass

    try:
        catalog.drop_table(identifier=identifier)
    except NoSuchTableError:
        pass

    pa_table = pa.Table.from_pydict(
        {
            "foo": ["a", None, "z"],
        },
        schema=pa.schema([pa.field("foo", pa.large_string(), nullable=True)]),
    )

    pa_table_with_column = pa.Table.from_pydict(
        {
            "foo": ["a", None, "z"],
            "bar": [19, None, 25],
        },
        schema=pa.schema(
            [
                pa.field("foo", pa.large_string(), nullable=True),
                pa.field("bar", pa.int32(), nullable=True),
            ]
        ),
    )

    with catalog.create_table_transaction(
        identifier=identifier, schema=pa_table.schema, properties={"format-version": str(format_version)}
    ) as txn:
        with txn.update_snapshot().fast_append() as snapshot_update:
            for data_file in _dataframe_to_data_files(table_metadata=txn.table_metadata, df=pa_table, io=txn._table.io):
                snapshot_update.append_data_file(data_file)

        with txn.update_schema() as schema_txn:
            schema_txn.union_by_name(pa_table_with_column.schema)

        with txn.update_snapshot().fast_append() as snapshot_update:
            for data_file in _dataframe_to_data_files(
                table_metadata=txn.table_metadata, df=pa_table_with_column, io=txn._table.io
            ):
                snapshot_update.append_data_file(data_file)

    tbl = catalog.load_table(identifier=identifier)
    assert tbl.format_version == format_version
    assert len(tbl.scan().to_arrow()) == 6


REPLACE_TABLE_SCHEMA = Schema(
    NestedField(field_id=1, name="id", field_type=LongType(), required=False),
    NestedField(field_id=2, name="data", field_type=StringType(), required=False),
)
REPLACE_TABLE_SCHEMA_WITH_EXTRA = Schema(
    *REPLACE_TABLE_SCHEMA.fields, NestedField(field_id=3, name="extra", field_type=BooleanType(), required=False)
)
REPLACE_TABLE_SPEC = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="id"))


def _create_table_to_replace(catalog: Catalog, identifier: Identifier, **kwargs: Any) -> Table:
    catalog.create_namespace_if_not_exists(Catalog.namespace_from(identifier))
    return catalog.create_table(identifier, REPLACE_TABLE_SCHEMA, **kwargs)


def _rows(schema: Schema, *ids: int) -> pa.Table:
    return pa.Table.from_pylist([{"id": i, "data": str(i)} for i in ids], schema=schema.as_arrow())


def test_replace_table_transaction(catalog: Catalog, test_table_identifier: Identifier) -> None:
    table = _create_table_to_replace(catalog, test_table_identifier)
    table.append(_rows(table.schema(), 1, 2))
    original_snapshot = table.current_snapshot()
    assert original_snapshot is not None

    # Reorder the columns and add one, from PyArrow so that the schema has no field IDs
    schema = pa.schema([pa.field("data", pa.large_string()), pa.field("extra", pa.bool_()), pa.field("id", pa.int64())])
    txn = catalog.replace_table_transaction(test_table_identifier, schema=schema)
    txn.append(pa.Table.from_pylist([{"data": "3", "extra": True, "id": 3}], schema=schema))
    assert catalog.load_table(test_table_identifier).metadata == table.metadata

    replaced = txn.commit_transaction()

    assert replaced.metadata == catalog.load_table(test_table_identifier).metadata
    assert replaced.metadata.table_uuid == table.metadata.table_uuid
    assert replaced.location() == table.location()
    assert replaced.schema().schema_id == 1
    assert replaced.schema() == Schema(
        NestedField(field_id=2, name="data", field_type=StringType(), required=False),
        NestedField(field_id=3, name="extra", field_type=BooleanType(), required=False),
        NestedField(field_id=1, name="id", field_type=LongType(), required=False),
    )
    snapshot = replaced.current_snapshot()
    assert snapshot is not None
    assert snapshot.parent_snapshot_id is None
    assert [entry.snapshot_id for entry in replaced.history()] == [original_snapshot.snapshot_id, snapshot.snapshot_id]
    assert len(replaced.metadata.metadata_log) == len(table.metadata.metadata_log) + 1
    assert replaced.scan().to_arrow().to_pylist() == [{"data": "3", "extra": True, "id": 3}]
    assert replaced.scan(snapshot_id=original_snapshot.snapshot_id).to_arrow().to_pylist() == [
        {"id": 1, "data": "1"},
        {"id": 2, "data": "2"},
    ]


def test_replace_table_transaction_with_all_arguments(
    catalog: Catalog, test_table_identifier: Identifier, tmp_path: Path
) -> None:
    table = _create_table_to_replace(catalog, test_table_identifier, properties={"keep": "a", "override": "b"})
    table.append(_rows(table.schema(), 1))
    location = f"file://{tmp_path}/replaced"

    txn = catalog.replace_table_transaction(
        test_table_identifier,
        schema=REPLACE_TABLE_SCHEMA,
        location=f"{location}/",
        partition_spec=PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="data")),
        sort_order=SortOrder(SortField(source_id=1, transform=IdentityTransform(), direction=SortDirection.DESC)),
        properties={"override": "c", "new": "d"},
    )
    txn.append(_rows(REPLACE_TABLE_SCHEMA, 2))
    assert catalog.load_table(test_table_identifier).metadata == table.metadata

    replaced = txn.commit_transaction()

    assert replaced.spec() == PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="data"), spec_id=1
    )
    assert replaced.sort_order() == SortOrder(
        SortField(source_id=1, transform=IdentityTransform(), direction=SortDirection.DESC, null_order=NullOrder.NULLS_LAST),
        order_id=1,
    )
    assert replaced.properties == {"keep": "a", "override": "c", "new": "d"}
    assert replaced.location() == location
    snapshot = replaced.current_snapshot()
    assert snapshot is not None
    assert snapshot.manifest_list.startswith(location)
    assert all(task.file.file_path.startswith(location) for task in replaced.scan().plan_files())
    assert replaced.metadata_location.startswith(location)
    assert replaced.scan().to_arrow().to_pylist() == [{"id": 2, "data": "2"}]


def test_replace_table_transaction_requires_table_exists(catalog: Catalog) -> None:
    catalog.create_namespace("default")

    with pytest.raises(NoSuchTableError, match="Table does not exist: default.missing"):
        catalog.replace_table_transaction(("default", "missing"), schema=REPLACE_TABLE_SCHEMA)


def test_replace_table_transaction_without_writes(catalog: Catalog, test_table_identifier: Identifier) -> None:
    table = _create_table_to_replace(catalog, test_table_identifier)
    table.append(_rows(table.schema(), 1))
    snapshot = table.current_snapshot()
    assert snapshot is not None
    table.manage_snapshots().create_tag(snapshot.snapshot_id, "tag").create_branch(snapshot.snapshot_id, "branch").commit()

    replaced = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA).commit_transaction()

    # Only the main branch is removed
    assert replaced.current_snapshot() is None
    assert set(replaced.refs()) == {"tag", "branch"}
    assert replaced.snapshots() == table.snapshots()
    assert replaced.scan().to_arrow().num_rows == 0


def test_replace_table_transaction_reuses_schema(catalog: Catalog, test_table_identifier: Identifier) -> None:
    _create_table_to_replace(catalog, test_table_identifier)
    catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA_WITH_EXTRA).commit_transaction()

    replaced = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA).commit_transaction()

    assert replaced.schema().schema_id == 0
    assert [schema.schema_id for schema in replaced.metadata.schemas] == [0, 1]
    assert replaced.metadata.last_column_id == 3

    # A new schema takes the ID after the highest, not after the current one
    replaced = catalog.replace_table_transaction(
        test_table_identifier, schema=Schema(REPLACE_TABLE_SCHEMA.fields[0])
    ).commit_transaction()
    assert replaced.schema().schema_id == 2


def test_replace_table_transaction_reuses_partition_specs_and_sort_orders(
    catalog: Catalog, test_table_identifier: Identifier
) -> None:
    spec = REPLACE_TABLE_SPEC
    sort_order = SortOrder(SortField(source_id=1, transform=IdentityTransform()))
    _create_table_to_replace(catalog, test_table_identifier, partition_spec=spec, sort_order=sort_order)

    # The table was created sorted, so it has no unsorted order yet
    replaced = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA).commit_transaction()
    assert replaced.spec() == PartitionSpec(spec_id=1)
    assert replaced.sort_order() == UNSORTED_SORT_ORDER
    assert [sort_order.order_id for sort_order in replaced.metadata.sort_orders] == [1, 0]

    replaced = catalog.replace_table_transaction(
        test_table_identifier,
        schema=REPLACE_TABLE_SCHEMA,
        partition_spec=PartitionSpec(
            PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="id"),
            PartitionField(source_id=2, field_id=1001, transform=BucketTransform(4), name="data_bucket"),
        ),
        sort_order=sort_order,
    ).commit_transaction()
    assert replaced.spec() == PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="id"),
        PartitionField(source_id=2, field_id=1001, transform=BucketTransform(4), name="data_bucket"),
        spec_id=2,
    )
    assert replaced.sort_order().order_id == 1

    replaced = catalog.replace_table_transaction(
        test_table_identifier, schema=REPLACE_TABLE_SCHEMA, partition_spec=spec
    ).commit_transaction()
    assert replaced.spec().spec_id == 0
    assert replaced.metadata.last_partition_id == 1001
    assert replaced.sort_order().order_id == 0

    # A new spec and sort order take the ID after the highest, not after the current one
    replaced = catalog.replace_table_transaction(
        test_table_identifier,
        schema=REPLACE_TABLE_SCHEMA,
        partition_spec=PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="data")),
        sort_order=SortOrder(SortField(source_id=2)),
    ).commit_transaction()
    assert replaced.spec().spec_id == 3
    assert replaced.sort_order().order_id == 2


def test_replace_table_transaction_adds_sort_order(catalog: Catalog, test_table_identifier: Identifier) -> None:
    _create_table_to_replace(catalog, test_table_identifier, sort_order=SortOrder(SortField(source_id=1)))

    replaced = catalog.replace_table_transaction(
        test_table_identifier, schema=REPLACE_TABLE_SCHEMA, sort_order=SortOrder(SortField(source_id=2))
    ).commit_transaction()

    assert replaced.sort_order() == SortOrder(SortField(source_id=2), order_id=2)
    assert [sort_order.order_id for sort_order in replaced.metadata.sort_orders] == [1, 2]


def test_replace_table_transaction_identifier_fields(catalog: Catalog, test_table_identifier: Identifier) -> None:
    _create_table_to_replace(catalog, test_table_identifier)
    fields = (NestedField(field_id=1, name="id", field_type=LongType(), required=True), *REPLACE_TABLE_SCHEMA.fields[1:])

    replaced = catalog.replace_table_transaction(
        test_table_identifier, schema=Schema(*fields, identifier_field_ids=[1])
    ).commit_transaction()
    assert replaced.schema().identifier_field_ids == [1]

    # A schema that only differs in its identifier fields isn't reused
    replaced = catalog.replace_table_transaction(test_table_identifier, schema=Schema(*fields)).commit_transaction()
    assert replaced.schema().schema_id == 2
    assert replaced.schema().identifier_field_ids == []


def test_replace_table_transaction_rejects_schema_of_later_format_version(
    catalog: Catalog, test_table_identifier: Identifier
) -> None:
    _create_table_to_replace(catalog, test_table_identifier)
    schema = Schema(NestedField(field_id=1, name="ts", field_type=TimestampNanoType(), required=False))

    with pytest.raises(ValueError, match="timestamp_ns is only supported in 3 or higher. Current format version is: 2"):
        catalog.replace_table_transaction(test_table_identifier, schema=schema)


@pytest.mark.parametrize(
    "properties",
    [pytest.param({}, id="v1"), pytest.param({TableProperties.FORMAT_VERSION: "2"}, id="upgrade-to-v2")],
)
def test_replace_v1_table_transaction(catalog: Catalog, test_table_identifier: Identifier, properties: dict[str, str]) -> None:
    _create_table_to_replace(
        catalog,
        test_table_identifier,
        partition_spec=PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="id")),
        properties={TableProperties.FORMAT_VERSION: "1"},
    )

    replaced = catalog.replace_table_transaction(
        test_table_identifier,
        schema=REPLACE_TABLE_SCHEMA,
        partition_spec=PartitionSpec(
            PartitionField(source_id=2, field_id=1000, transform=BucketTransform(4), name="data_bucket")
        ),
        properties=properties,
    ).commit_transaction()

    assert replaced.format_version == int(properties.get(TableProperties.FORMAT_VERSION, 1))
    assert TableProperties.FORMAT_VERSION not in replaced.properties
    # The spec follows the v1 rules of the replaced table, so the removed field is carried forward as void
    assert replaced.spec() == PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=VoidTransform(), name="id"),
        PartitionField(source_id=2, field_id=1001, transform=BucketTransform(4), name="data_bucket"),
        spec_id=1,
    )


@pytest.mark.parametrize(
    "format_version, message",
    [("1", "Cannot downgrade v2 table to v1"), ("3", "Unsupported table format version: 3")],
)
def test_replace_table_transaction_rejects_format_version(
    catalog: Catalog, test_table_identifier: Identifier, format_version: str, message: str
) -> None:
    _create_table_to_replace(catalog, test_table_identifier)

    with pytest.raises(ValueError, match=message):
        catalog.replace_table_transaction(
            test_table_identifier, schema=REPLACE_TABLE_SCHEMA, properties={TableProperties.FORMAT_VERSION: format_version}
        )


def test_replace_table_transaction_rejects_invalid_definition(catalog: Catalog, test_table_identifier: Identifier) -> None:
    _create_table_to_replace(catalog, test_table_identifier)

    cases: list[tuple[dict[str, Any], type[Exception], str]] = [
        (
            {"partition_spec": PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=DayTransform(), name="day"))},
            IcebergValidationError,
            "Invalid source field data with type string for transform: day",
        ),
        (
            {"sort_order": SortOrder(SortField(source_id=2, transform=DayTransform()))},
            IcebergValidationError,
            "Invalid source field data with type string for transform: day",
        ),
        (
            {"properties": {TableProperties.ENCRYPTION_KEY_ID: "key"}},
            ValueError,
            r"Invalid properties for v2: \['encryption.key-id'\]",
        ),
    ]

    # Rejected when the transaction is created, before anything is written
    for kwargs, error, message in cases:
        with pytest.raises(error, match=message):
            catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA, **kwargs)


def test_concurrent_replace_table_transactions(catalog: Catalog, test_table_identifier: Identifier) -> None:
    for label, first_replace, second_replace in [
        ("unchanged", {}, {}),
        ("schema-first", {"schema": REPLACE_TABLE_SCHEMA_WITH_EXTRA}, {}),
        ("schema-second", {}, {"schema": REPLACE_TABLE_SCHEMA_WITH_EXTRA}),
        ("spec-first", {"partition_spec": REPLACE_TABLE_SPEC}, {}),
        ("spec-second", {}, {"partition_spec": REPLACE_TABLE_SPEC}),
        ("sort-order-first", {"sort_order": SortOrder(SortField(source_id=1))}, {}),
    ]:
        identifier = (*test_table_identifier[:-1], f"{test_table_identifier[-1]}_{label}")
        table = _create_table_to_replace(catalog, identifier)
        table.append(_rows(table.schema(), 1))
        second = catalog.replace_table_transaction(identifier, **{"schema": REPLACE_TABLE_SCHEMA, **second_replace})
        second.append(_rows(second.table_metadata.schema(), 3))
        expected = second.table_metadata

        first = catalog.replace_table_transaction(identifier, **{"schema": REPLACE_TABLE_SCHEMA, **first_replace})
        first.append(_rows(first.table_metadata.schema(), 2))
        first.commit_transaction()
        second.commit_transaction()

        # The last replace wins, also reverting the definition of the first
        replaced = catalog.load_table(identifier)
        assert replaced.schema().schema_id == expected.current_schema_id, label
        assert replaced.schema() == expected.schema(), label
        assert replaced.spec() == expected.spec(), label
        assert replaced.sort_order() == expected.sort_order(), label
        assert replaced.scan().to_arrow().to_pylist() == _rows(expected.schema(), 3).to_pylist(), label


def test_concurrent_replace_table_transactions_conflict(catalog: Catalog, test_table_identifier: Identifier) -> None:
    for label, create, first_replace, second_replace, message in [
        (
            "schema-with-new-field",
            {},
            {"schema": REPLACE_TABLE_SCHEMA_WITH_EXTRA},
            {"schema": REPLACE_TABLE_SCHEMA_WITH_EXTRA},
            "last assigned field id has changed",
        ),
        (
            "schema",
            {},
            {"schema": Schema(REPLACE_TABLE_SCHEMA.fields[0])},
            {"schema": Schema(REPLACE_TABLE_SCHEMA.fields[1])},
            "current schema id has changed",
        ),
        (
            "spec-with-new-field",
            {},
            {"partition_spec": REPLACE_TABLE_SPEC},
            {"partition_spec": REPLACE_TABLE_SPEC},
            "last assigned partition id has changed",
        ),
        ("spec", {"partition_spec": REPLACE_TABLE_SPEC}, {}, {}, "default spec id has changed"),
        (
            "sort-order",
            {},
            {"sort_order": SortOrder(SortField(source_id=1))},
            {"sort_order": SortOrder(SortField(source_id=2))},
            "default sort order id has changed",
        ),
    ]:
        identifier = (*test_table_identifier[:-1], f"{test_table_identifier[-1]}_{label}")
        _create_table_to_replace(catalog, identifier, **create)
        second = catalog.replace_table_transaction(identifier, **{"schema": REPLACE_TABLE_SCHEMA, **second_replace})
        second.append(_rows(second.table_metadata.schema(), 2))
        first = catalog.replace_table_transaction(identifier, **{"schema": REPLACE_TABLE_SCHEMA, **first_replace})
        first.append(_rows(first.table_metadata.schema(), 1))
        replaced = first.commit_transaction()
        manifest_list = second.table_metadata.snapshots[-1].manifest_list

        # Both replaces add a schema, spec or sort order with the same ID
        with pytest.raises(CommitFailedException, match=message):
            second.commit_transaction()

        assert catalog.load_table(identifier).metadata == replaced.metadata, label
        assert not os.path.exists(manifest_list.removeprefix("file://")), label
        with pytest.raises(RuntimeError, match="This transaction failed to commit and cannot be reused"):
            second.commit_transaction()


def test_replace_table_transaction_removes_concurrently_created_main(catalog: Catalog, test_table_identifier: Identifier) -> None:
    _create_table_to_replace(catalog, test_table_identifier)
    txn = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA)
    catalog.load_table(test_table_identifier).append(_rows(REPLACE_TABLE_SCHEMA, 1))

    replaced = txn.commit_transaction()

    assert replaced.current_snapshot() is None


def test_replace_table_transaction_with_concurrent_append(catalog: Catalog, test_table_identifier: Identifier) -> None:
    table = _create_table_to_replace(catalog, test_table_identifier)
    table.append(_rows(table.schema(), 1))
    txn = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA)
    txn.append(_rows(REPLACE_TABLE_SCHEMA, 3))
    # The append takes the sequence number of the replacement's snapshot
    catalog.load_table(test_table_identifier).append(_rows(REPLACE_TABLE_SCHEMA, 2))

    txn.commit_transaction()

    # The replacement's snapshot has no parent, so it can have the same sequence number
    replaced = catalog.load_table(test_table_identifier)
    assert replaced.scan().to_arrow().to_pylist() == [{"id": 3, "data": "3"}]
    assert [snapshot.sequence_number for snapshot in replaced.snapshots()] == [1, 2, 2]
    assert replaced.metadata.last_sequence_number == 2


def test_replace_table_transaction_conflicts_with_concurrent_appends(catalog: Catalog, test_table_identifier: Identifier) -> None:
    table = _create_table_to_replace(catalog, test_table_identifier)
    table.append(_rows(table.schema(), 1))
    txn = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA)
    txn.append(_rows(REPLACE_TABLE_SCHEMA, 3))
    staged_snapshot = txn.table_metadata.current_snapshot()
    assert staged_snapshot is not None
    # The appends take the sequence number of the replacement's snapshot and the one after it
    catalog.load_table(test_table_identifier).append(_rows(REPLACE_TABLE_SCHEMA, 2))
    catalog.load_table(test_table_identifier).append(_rows(REPLACE_TABLE_SCHEMA, 2))
    appended = catalog.load_table(test_table_identifier)

    with pytest.raises(ValidationException, match="Cannot add snapshot with sequence number 2 lower than last sequence number 3"):
        txn.commit_transaction()

    assert catalog.load_table(test_table_identifier).metadata == appended.metadata
    # The commit didn't land, so its files are deleted
    assert not appended.io.new_input(staged_snapshot.manifest_list).exists()


def test_replace_table_transaction_keeps_files_when_commit_outcome_is_unknown(
    catalog: Catalog, test_table_identifier: Identifier
) -> None:
    _create_table_to_replace(catalog, test_table_identifier)
    txn = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA)
    txn.append(_rows(REPLACE_TABLE_SCHEMA, 1))
    real_commit = catalog.commit_table

    def commit_then_lose_response(*args: Any, **kwargs: Any) -> Any:
        real_commit(*args, **kwargs)
        raise CommitStateUnknownException("response lost after the commit landed")

    with patch.object(catalog, "commit_table", side_effect=commit_then_lose_response):
        with pytest.raises(CommitStateUnknownException):
            txn.commit_transaction()

    # The commit landed, so its files must still be there
    assert catalog.load_table(test_table_identifier).scan().to_arrow().to_pylist() == [{"id": 1, "data": "1"}]
    with pytest.raises(RuntimeError, match="This transaction failed to commit and cannot be reused"):
        txn.commit_transaction()


def test_replace_table_transaction_that_landed_but_was_reported_failed_keeps_its_files(
    catalog: Catalog, test_table_identifier: Identifier
) -> None:
    _create_table_to_replace(catalog, test_table_identifier)
    txn = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA)
    txn.append(_rows(REPLACE_TABLE_SCHEMA, 1))
    real_commit = catalog.commit_table

    def commit_then_report_failure(*args: Any, **kwargs: Any) -> Any:
        real_commit(*args, **kwargs)
        raise CommitFailedException("response lost after the commit landed")

    with patch.object(catalog, "commit_table", side_effect=commit_then_report_failure):
        replaced = txn.commit_transaction()

    # The staged snapshot is in the table, so the commit landed
    assert replaced.scan().to_arrow().to_pylist() == [{"id": 1, "data": "1"}]
    assert catalog.load_table(test_table_identifier).scan().to_arrow().to_pylist() == [{"id": 1, "data": "1"}]


def test_replace_table_transaction_commits_later_changes_like_any_transaction(
    catalog: Catalog, test_table_identifier: Identifier
) -> None:
    # A v1 table, so that sequence numbers don't detect the conflict
    _create_table_to_replace(catalog, test_table_identifier, properties={TableProperties.FORMAT_VERSION: "1"})
    txn = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA)
    txn.append(_rows(REPLACE_TABLE_SCHEMA, 1))
    txn.commit_transaction()

    txn.append(_rows(REPLACE_TABLE_SCHEMA, 2))
    catalog.load_table(test_table_identifier).append(_rows(REPLACE_TABLE_SCHEMA, 3))
    txn.commit_transaction()

    # The append is retried on top of the concurrent one rather than overwriting it, and the replace's isn't redone
    rows = catalog.load_table(test_table_identifier).scan().to_arrow().to_pylist()
    assert sorted(row["id"] for row in rows) == [1, 2, 3]


def test_replace_table_transaction_keeps_default_values(catalog: Catalog, test_table_identifier: Identifier) -> None:
    table = _create_table_to_replace(catalog, test_table_identifier)
    with table.update_schema() as update:
        update.add_column("extra", IntegerType())
    with table.update_schema() as update:
        update.set_default_value("extra", 7)

    replaced = catalog.replace_table_transaction(test_table_identifier, schema=table.schema()).commit_transaction()

    assert replaced.schema().schema_id == table.schema().schema_id
    assert replaced.schema() == table.schema()


def test_replace_table_transaction_updates_name_mapping(
    catalog: Catalog, test_table_identifier: Identifier, tmp_path: Path
) -> None:
    table = _create_table_to_replace(catalog, test_table_identifier)
    # Files written without field IDs are read with the name mapping that add_files sets
    pq.write_table(pa.table({"id": [1], "data": ["1"]}), tmp_path / "before.parquet")
    table.add_files([f"file://{tmp_path}/before.parquet"])
    original_snapshot = table.current_snapshot()
    assert original_snapshot is not None

    pq.write_table(pa.table({"id": [2], "data": ["2"], "extra": [True]}), tmp_path / "after.parquet")
    with catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA_WITH_EXTRA) as txn:
        txn.add_files([f"file://{tmp_path}/after.parquet"])

    replaced = catalog.load_table(test_table_identifier)
    assert replaced.scan().to_arrow().to_pylist() == [{"id": 2, "data": "2", "extra": True}]
    assert replaced.scan(snapshot_id=original_snapshot.snapshot_id).to_arrow().to_pylist() == [{"id": 1, "data": "1"}]


def test_replace_v1_table_transaction_keeps_void_field_of_dropped_column(
    catalog: Catalog, test_table_identifier: Identifier
) -> None:
    table = _create_table_to_replace(
        catalog,
        test_table_identifier,
        partition_spec=PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="data")),
        properties={TableProperties.FORMAT_VERSION: "1"},
    )
    with table.update_spec() as update:
        update.remove_field("data")
    with table.update_schema() as update:
        update.delete_column("data")
    void_spec = table.spec()

    replaced = catalog.replace_table_transaction(
        test_table_identifier, schema=table.schema(), partition_spec=void_spec
    ).commit_transaction()
    assert replaced.spec() == void_spec

    # "data" is added back with a new field ID, while the void field keeps its name
    replaced = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA).commit_transaction()
    assert replaced.schema().find_field("data").field_id == 3
    assert replaced.spec() == void_spec


@pytest.mark.parametrize("remove_branch", [False, True], ids=["write-branch", "remove-branch"])
def test_replace_table_transaction_conflicts_with_concurrent_branch_commit(
    catalog: Catalog, test_table_identifier: Identifier, remove_branch: bool
) -> None:
    # A v1 table, so that sequence numbers don't detect the conflict
    table = _create_table_to_replace(catalog, test_table_identifier, properties={TableProperties.FORMAT_VERSION: "1"})
    table.append(_rows(table.schema(), 1))
    snapshot = table.current_snapshot()
    assert snapshot is not None
    table.manage_snapshots().create_branch(snapshot.snapshot_id, "branch").commit()
    txn = catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA)
    if remove_branch:
        ManageSnapshots(transaction=txn).remove_branch("branch").commit()
    else:
        txn.append(_rows(REPLACE_TABLE_SCHEMA, 3), branch="branch")
    catalog.load_table(test_table_identifier).append(_rows(REPLACE_TABLE_SCHEMA, 2), branch="branch")

    with pytest.raises(CommitFailedException, match=f"branch branch has changed: expected id {snapshot.snapshot_id}"):
        txn.commit_transaction()


def test_replace_table_transaction_writes_branch(catalog: Catalog, test_table_identifier: Identifier) -> None:
    table = _create_table_to_replace(catalog, test_table_identifier)
    table.append(_rows(table.schema(), 1))
    snapshot = table.current_snapshot()
    assert snapshot is not None
    table.manage_snapshots().create_branch(snapshot.snapshot_id, "branch").commit()

    with catalog.replace_table_transaction(test_table_identifier, schema=REPLACE_TABLE_SCHEMA) as txn:
        txn.append(_rows(REPLACE_TABLE_SCHEMA, 2), branch="branch")

    replaced = catalog.load_table(test_table_identifier)
    assert replaced.current_snapshot() is None
    assert sorted(row["id"] for row in replaced.scan().use_ref("branch").to_arrow().to_pylist()) == [1, 2]


def test_replace_table_transaction_dynamic_partition_overwrite(
    catalog: Catalog, test_table_identifier: Identifier, tmp_path: Path
) -> None:
    table = _create_table_to_replace(
        catalog,
        test_table_identifier,
        partition_spec=PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="data")),
    )
    table.append(_rows(table.schema(), 1))

    schema = Schema(
        NestedField(field_id=1, name="id", field_type=LongType(), required=False),
        NestedField(field_id=2, name="region", field_type=StringType(), required=False),
        NestedField(field_id=3, name="data", field_type=StringType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region"))
    location = f"file://{tmp_path}/replaced"
    df = pa.table({"id": [2, 3], "region": ["eu", "us"], "data": ["x", "y"]})
    with catalog.replace_table_transaction(test_table_identifier, schema=schema, partition_spec=spec, location=location) as txn:
        # The replacement has no main branch, so there is nothing to delete
        with pytest.warns(UserWarning, match="Delete operation did not match any records"):
            txn.dynamic_partition_overwrite(df)

    # The data files are written with the replacement's schema, spec and location
    replaced = catalog.load_table(test_table_identifier)
    assert sorted(replaced.scan().to_arrow().to_pylist(), key=lambda row: row["id"]) == df.to_pylist()
    files = [task.file for task in replaced.scan().plan_files()]
    assert {file.partition for file in files} == {Record("eu"), Record("us")}
    assert all(file.file_path.startswith(location) for file in files)


def test_replace_table_transaction_matches_dotted_names_by_parent(
    catalog: Catalog, test_table_identifier: Identifier, tmp_path: Path
) -> None:
    catalog.create_namespace(Catalog.namespace_from(test_table_identifier))
    struct_type = StructType(NestedField(field_id=2, name="a", field_type=LongType(), required=False))
    table = catalog.create_table(
        test_table_identifier, Schema(NestedField(field_id=1, name="p", field_type=struct_type, required=False))
    )
    pq.write_table(pa.table({"p": [{"a": 1}]}), tmp_path / "before.parquet")
    table.add_files([f"file://{tmp_path}/before.parquet"])
    original_snapshot = table.current_snapshot()
    assert original_snapshot is not None

    # A top-level "p.a" has the same full name as a in struct p, but is a different column
    schema = Schema(NestedField(field_id=1, name="p.a", field_type=LongType(), required=False))
    pq.write_table(pa.table({"p.a": [2]}), tmp_path / "after.parquet")
    with catalog.replace_table_transaction(test_table_identifier, schema=schema) as txn:
        txn.add_files([f"file://{tmp_path}/after.parquet"])

    replaced = catalog.load_table(test_table_identifier)
    assert replaced.schema().find_field("p.a").field_id == 3
    assert replaced.scan().to_arrow().to_pylist() == [{"p.a": 2}]
    assert replaced.scan(snapshot_id=original_snapshot.snapshot_id).to_arrow().to_pylist() == [{"p": {"a": 1}}]


def test_replace_table_transaction_updates_name_mapping_of_nested_field(
    catalog: Catalog, test_table_identifier: Identifier, tmp_path: Path
) -> None:
    catalog.create_namespace(Catalog.namespace_from(test_table_identifier))
    struct_type = StructType(NestedField(field_id=2, name="a", field_type=LongType(), required=False))
    table = catalog.create_table(
        test_table_identifier, Schema(NestedField(field_id=1, name="p", field_type=struct_type, required=False))
    )
    pq.write_table(pa.table({"p": [{"a": 1}]}), tmp_path / "before.parquet")
    table.add_files([f"file://{tmp_path}/before.parquet"])

    # The replacement adds b to the existing struct p
    struct_type = StructType(*struct_type.fields, NestedField(field_id=3, name="b", field_type=LongType(), required=False))
    schema = Schema(NestedField(field_id=1, name="p", field_type=struct_type, required=False))
    pq.write_table(pa.table({"p": [{"a": 2, "b": 3}]}), tmp_path / "after.parquet")
    with catalog.replace_table_transaction(test_table_identifier, schema=schema) as txn:
        txn.add_files([f"file://{tmp_path}/after.parquet"])

    assert catalog.load_table(test_table_identifier).scan().to_arrow().to_pylist() == [{"p": {"a": 2, "b": 3}}]


def test_replace_v1_table_transaction_keeps_void_fields_of_same_source(
    catalog: Catalog, test_table_identifier: Identifier
) -> None:
    table = _create_table_to_replace(
        catalog,
        test_table_identifier,
        partition_spec=PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=BucketTransform(4), name="p")),
        properties={TableProperties.FORMAT_VERSION: "1"},
    )
    with table.update_spec() as update:
        update.remove_field("p").add_field("id", BucketTransform(8), "q")
    with table.update_spec() as update:
        update.remove_field("q").add_field("data", IdentityTransform(), "q_1001")
    spec = table.spec()

    replaced = catalog.replace_table_transaction(
        test_table_identifier, schema=table.schema(), partition_spec=spec
    ).commit_transaction()

    assert replaced.spec() == spec


def test_replace_table_transaction_uses_partition_field_ids_once(catalog: Catalog, test_table_identifier: Identifier) -> None:
    table = _create_table_to_replace(
        catalog,
        test_table_identifier,
        partition_spec=PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=BucketTransform(4), name="p")),
        properties={TableProperties.FORMAT_VERSION: "1"},
    )
    # Upgrading a v1 table with two void fields for "id" to v2
    with table.update_spec() as update:
        update.remove_field("p").add_field("id", BucketTransform(8), "q")
    with table.update_spec() as update:
        update.remove_field("q")
    table.transaction().upgrade_table_version(2).commit_transaction()
    spec = PartitionSpec(
        *table.spec().fields, PartitionField(source_id=2, field_id=1002, transform=IdentityTransform(), name="data")
    )

    replaced = catalog.replace_table_transaction(
        test_table_identifier, schema=table.schema(), partition_spec=spec
    ).commit_transaction()

    assert replaced.spec().fields == spec.fields


def test_create_table_default_sort_order(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    assert table.sort_order().order_id == 0, "Order ID must match"
    assert table.sort_order().is_unsorted is True, "Order must be unsorted"
    catalog.drop_table(test_table_identifier)


def test_create_v1_table(catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested, properties={"format-version": "1"})
    assert table.sort_order().order_id == 0, "Order ID must match"
    assert table.sort_order().is_unsorted is True, "Order must be unsorted"
    assert table.format_version == 1
    assert table.spec() == UNPARTITIONED_PARTITION_SPEC
    catalog.drop_table(test_table_identifier)


@pytest.mark.parametrize("format_version", [1, 2])
def test_create_table_rejects_encryption_properties_before_v3(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier, format_version: int
) -> None:
    catalog.create_namespace(Catalog.namespace_from(test_table_identifier))

    with pytest.raises(ValueError, match=rf"Invalid properties for v{format_version}: \['encryption.key-id'\]"):
        catalog.create_table(
            test_table_identifier,
            table_schema_nested,
            properties={"format-version": str(format_version), "encryption.key-id": "key"},
        )

    assert not catalog.table_exists(test_table_identifier)


def test_create_table_custom_sort_order(catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    order = SortOrder(SortField(source_id=2, transform=IdentityTransform(), null_order=NullOrder.NULLS_FIRST))
    table = catalog.create_table(test_table_identifier, table_schema_nested, sort_order=order)
    given_sort_order = table.sort_order()
    assert given_sort_order.order_id == 1, "Order ID must match"
    assert len(given_sort_order.fields) == 1, "Order must have 1 field"
    assert given_sort_order.fields[0].direction == SortDirection.ASC, "Direction must match"
    assert given_sort_order.fields[0].null_order == NullOrder.NULLS_FIRST, "Null order must match"
    assert isinstance(given_sort_order.fields[0].transform, IdentityTransform), "Transform must match"
    catalog.drop_table(test_table_identifier)


def test_update_sort_order_default_null_order(
    catalog: Catalog, table_schema_simple: Schema, test_table_identifier: Identifier
) -> None:
    catalog.create_namespace(Catalog.namespace_from(test_table_identifier))
    table = catalog.create_table(test_table_identifier, table_schema_simple)

    # Without a null order, ascending fields are nulls-first and descending fields are nulls-last
    table.update_sort_order().asc("foo", IdentityTransform()).desc("bar", IdentityTransform()).commit()
    assert table.sort_order() == SortOrder(
        SortField(source_id=1, transform=IdentityTransform(), direction=SortDirection.ASC, null_order=NullOrder.NULLS_FIRST),
        SortField(source_id=2, transform=IdentityTransform(), direction=SortDirection.DESC, null_order=NullOrder.NULLS_LAST),
        order_id=1,
    )

    # An explicit null order is kept
    table.update_sort_order().asc("foo", IdentityTransform(), NullOrder.NULLS_LAST).desc(
        "bar", IdentityTransform(), NullOrder.NULLS_FIRST
    ).commit()
    assert table.sort_order() == SortOrder(
        SortField(source_id=1, transform=IdentityTransform(), direction=SortDirection.ASC, null_order=NullOrder.NULLS_LAST),
        SortField(source_id=2, transform=IdentityTransform(), direction=SortDirection.DESC, null_order=NullOrder.NULLS_FIRST),
        order_id=2,
    )
    catalog.drop_table(test_table_identifier)


def test_create_table_with_default_warehouse_location(
    warehouse: Path, catalog_with_warehouse: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier
) -> None:
    identifier_tuple = Catalog.identifier_to_tuple(test_table_identifier)
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog_with_warehouse.create_namespace(namespace)
    catalog_with_warehouse.create_table(test_table_identifier, table_schema_nested)
    table = catalog_with_warehouse.load_table(test_table_identifier)
    assert table.name() == identifier_tuple
    assert table.metadata_location.startswith(f"file://{warehouse}")
    assert os.path.exists(table.metadata_location[len("file://") :])
    catalog_with_warehouse.drop_table(test_table_identifier)


def test_create_table_location_override(
    catalog: Catalog,
    tmp_path: Path,
    table_schema_nested: Schema,
    test_table_identifier: Identifier,
    test_table_properties: dict[str, str],
) -> None:
    test_partition_spec = PartitionSpec(PartitionField(name="x", transform=IdentityTransform(), source_id=1, field_id=1000))
    new_location = f"file://{tmp_path}/new_location"
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(
        identifier=test_table_identifier,
        schema=table_schema_nested,
        location=new_location,
        partition_spec=test_partition_spec,
        properties=test_table_properties,
    )
    assert catalog.load_table(test_table_identifier) == table
    assert table.location() == new_location


def test_create_table_removes_trailing_slash_from_location(
    warehouse: Path, catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier
) -> None:
    identifier_tuple = Catalog.identifier_to_tuple(test_table_identifier)
    namespace = Catalog.namespace_from(test_table_identifier)
    table_name = Catalog.table_name_from(identifier_tuple)
    location = f"file://{warehouse}/{catalog.name}/{table_name}-given"
    catalog.create_namespace(namespace)
    catalog.create_table(test_table_identifier, table_schema_nested, location=f"{location}/")
    table = catalog.load_table(test_table_identifier)
    assert table.name() == identifier_tuple
    assert table.metadata_location.startswith(f"file://{warehouse}")
    assert os.path.exists(table.metadata_location[len("file://") :])
    assert table.location() == location
    catalog.drop_table(test_table_identifier)


def test_create_tables_idempotency(catalog: Catalog) -> None:
    # Second initialization should not fail even if tables are already created
    catalog.create_tables()  # type: ignore[attr-defined]
    catalog.create_tables()  # type: ignore[attr-defined]


def test_create_table_pyarrow_schema(
    catalog: Catalog,
    pyarrow_schema_simple_without_ids: pa.Schema,
    test_table_identifier: Identifier,
    test_table_properties: dict[str, str],
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(
        identifier=test_table_identifier,
        schema=pyarrow_schema_simple_without_ids,
        properties=test_table_properties,
    )
    assert catalog.load_table(test_table_identifier) == table


def test_write_pyarrow_schema(catalog: Catalog, test_table_identifier: Identifier) -> None:
    pyarrow_table = pa.Table.from_arrays(
        [
            pa.array([None, "A", "B", "C"]),  # 'foo' column
            pa.array([1, 2, 3, 4]),  # 'bar' column
            pa.array([True, None, False, True]),  # 'baz' column
            pa.array([None, "A", "B", "C"]),  # 'large' column
        ],
        schema=pa.schema(
            [
                pa.field("foo", pa.large_string(), nullable=True),
                pa.field("bar", pa.int32(), nullable=False),
                pa.field("baz", pa.bool_(), nullable=True),
                pa.field("large", pa.large_string(), nullable=True),
            ]
        ),
    )
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, pyarrow_table.schema)
    table.append(pyarrow_table)


@pytest.mark.parametrize(
    "schema,expected",
    [
        (lf("pyarrow_schema_simple_without_ids"), lf("iceberg_schema_simple_no_ids")),
        (lf("table_schema_simple"), lf("table_schema_simple")),
        (lf("table_schema_nested"), lf("table_schema_nested")),
        (lf("pyarrow_schema_nested_without_ids"), lf("iceberg_schema_nested_no_ids")),
    ],
)
def test_convert_schema_if_needed(
    schema: Schema | pa.Schema,
    expected: Schema,
    catalog: Catalog,
) -> None:
    assert expected == catalog._convert_schema_if_needed(schema)


# Register table tests


def test_register_table(catalog: Catalog, test_table_identifier: Identifier, metadata_location: str) -> None:
    identifier_tuple = Catalog.identifier_to_tuple(test_table_identifier)
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.register_table(test_table_identifier, metadata_location)
    assert table.name() == identifier_tuple
    assert table.metadata_location == metadata_location
    assert os.path.exists(metadata_location)
    catalog.drop_table(test_table_identifier)


def test_register_existing_table(catalog: Catalog, test_table_identifier: Identifier, metadata_location: str) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    catalog.register_table(test_table_identifier, metadata_location)
    with pytest.raises(TableAlreadyExistsError):
        catalog.register_table(test_table_identifier, metadata_location)


# Load table tests


def test_load_table(catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    loaded_table = catalog.load_table(test_table_identifier)
    assert table.name() == loaded_table.name()
    assert table.metadata_location == loaded_table.metadata_location
    assert table.metadata == loaded_table.metadata


def test_load_table_from_self_identifier(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier
) -> None:
    identifier_tuple = Catalog.identifier_to_tuple(test_table_identifier)
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    intermediate = catalog.load_table(test_table_identifier)
    assert intermediate.name() == identifier_tuple
    loaded_table = catalog.load_table(intermediate.name())
    assert table.name() == loaded_table.name()
    assert table.metadata_location == loaded_table.metadata_location
    assert table.metadata == loaded_table.metadata


# Rename table tests


def test_rename_table(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier, another_table_identifier: Identifier
) -> None:
    from_namespace = Catalog.namespace_from(test_table_identifier)
    to_namespace = Catalog.namespace_from(another_table_identifier)
    catalog.create_namespace(from_namespace)
    catalog.create_namespace(to_namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    assert table.name() == test_table_identifier
    catalog.rename_table(test_table_identifier, another_table_identifier)
    new_table = catalog.load_table(another_table_identifier)
    assert new_table.name() == another_table_identifier
    assert new_table.metadata_location == table.metadata_location
    with pytest.raises(NoSuchTableError):
        catalog.load_table(test_table_identifier)


def test_rename_table_from_self_identifier(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier, another_table_identifier: Identifier
) -> None:
    from_namespace = Catalog.namespace_from(test_table_identifier)
    to_namespace = Catalog.namespace_from(another_table_identifier)
    catalog.create_namespace(from_namespace)
    catalog.create_namespace(to_namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    assert table.name() == test_table_identifier
    catalog.rename_table(table.name(), another_table_identifier)
    new_table = catalog.load_table(another_table_identifier)
    assert new_table.name() == another_table_identifier
    assert new_table.metadata_location == table.metadata_location
    with pytest.raises(NoSuchTableError):
        catalog.load_table(table.name())
    with pytest.raises(NoSuchTableError):
        catalog.load_table(test_table_identifier)


def test_rename_table_to_existing_one(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier, another_table_identifier: Identifier
) -> None:
    from_namespace = Catalog.namespace_from(test_table_identifier)
    to_namespace = Catalog.namespace_from(another_table_identifier)
    catalog.create_namespace(from_namespace)
    catalog.create_namespace(to_namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    assert table.name() == test_table_identifier
    new_table = catalog.create_table(another_table_identifier, table_schema_nested)
    assert new_table.name() == another_table_identifier
    with pytest.raises(TableAlreadyExistsError):
        catalog.rename_table(test_table_identifier, another_table_identifier)


def test_rename_missing_table(catalog: Catalog, test_table_identifier: Identifier, another_table_identifier: Identifier) -> None:
    from_namespace = Catalog.namespace_from(test_table_identifier)
    to_namespace = Catalog.namespace_from(another_table_identifier)
    catalog.create_namespace(from_namespace)
    catalog.create_namespace(to_namespace)
    with pytest.raises(NoSuchTableError):
        catalog.rename_table(test_table_identifier, another_table_identifier)


def test_rename_table_to_missing_namespace(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier, another_table_identifier: Identifier
) -> None:
    from_namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(from_namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    assert table.name() == test_table_identifier
    with pytest.raises(NoSuchNamespaceError):
        catalog.rename_table(test_table_identifier, another_table_identifier)


# Drop table tests


def test_drop_table(catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier) -> None:
    identifier_tuple = Catalog.identifier_to_tuple(test_table_identifier)
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    assert table.name() == identifier_tuple
    catalog.drop_table(test_table_identifier)
    with pytest.raises(NoSuchTableError):
        catalog.load_table(test_table_identifier)


def test_drop_table_from_self_identifier(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier
) -> None:
    identifier_tuple = Catalog.identifier_to_tuple(test_table_identifier)
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    assert table.name() == identifier_tuple
    catalog.drop_table(table.name())
    with pytest.raises(NoSuchTableError):
        catalog.load_table(table.name())
    with pytest.raises(NoSuchTableError):
        catalog.load_table(test_table_identifier)


def test_drop_table_that_does_not_exist_raise_error(catalog: Catalog, test_table_identifier: Identifier) -> None:
    with pytest.raises(NoSuchTableError):
        catalog.drop_table(test_table_identifier)


def test_purge_table(catalog: Catalog, table_schema_simple: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    catalog.create_table(test_table_identifier, table_schema_simple)
    catalog.purge_table(test_table_identifier)
    with pytest.raises(NoSuchTableError, match=f"Table does not exist: {'.'.join(test_table_identifier)}"):
        catalog.load_table(test_table_identifier)


def test_purge_encrypted_table_without_kms_client(
    catalog: Catalog, test_table_identifier: Identifier, example_table_metadata_v3_encrypted: dict[str, Any], tmp_path: Path
) -> None:
    metadata_location = tmp_path / "00000-encrypted.metadata.json"
    metadata_location.write_text(json.dumps(example_table_metadata_v3_encrypted))
    catalog.create_namespace(Catalog.namespace_from(test_table_identifier))
    catalog.register_table(test_table_identifier, str(metadata_location))

    with pytest.raises(ValueError, match="no key management client was provided"):
        catalog.purge_table(test_table_identifier)

    assert catalog.table_exists(test_table_identifier)


# List tables tests


def test_list_tables(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier, another_table_identifier: Identifier
) -> None:
    namespace_1 = Catalog.namespace_from(test_table_identifier)
    namespace_2 = Catalog.namespace_from(another_table_identifier)
    catalog.create_namespace(namespace_1)
    catalog.create_namespace(namespace_2)
    catalog.create_table(test_table_identifier, table_schema_nested)
    catalog.create_table(another_table_identifier, table_schema_nested)
    identifier_list = catalog.list_tables(namespace_1)
    assert len(identifier_list) == 1
    assert test_table_identifier in identifier_list

    identifier_list = catalog.list_tables(namespace_2)
    assert len(identifier_list) == 1
    assert another_table_identifier in identifier_list


def test_list_tables_under_a_namespace(catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    catalog.create_table(test_table_identifier, table_schema_nested)
    new_namespace = ("new", "namespace")
    catalog.create_namespace(new_namespace)
    all_tables = catalog.list_tables(namespace=namespace)
    new_namespace_tables = catalog.list_tables(new_namespace)
    assert all_tables
    assert test_table_identifier in all_tables
    assert new_namespace_tables == []


def test_list_tables_when_missing_namespace(catalog: Catalog, test_namespace: Identifier) -> None:
    with pytest.raises(NoSuchNamespaceError):
        catalog.list_tables(test_namespace)


# Commit table tests
def test_commit_table(catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)
    last_updated_ms = table.metadata.last_updated_ms
    original_table_metadata_location = table.metadata_location
    original_table_last_updated_ms = table.metadata.last_updated_ms

    assert catalog._parse_metadata_version(table.metadata_location) == 0  # type: ignore[attr-defined]
    assert table.metadata.current_schema_id == 0

    transaction = table.transaction()
    update = transaction.update_schema()
    update.add_column(path="b", field_type=IntegerType())
    update.commit()
    transaction.commit_transaction()

    updated_table_metadata = table.metadata

    assert catalog._parse_metadata_version(table.metadata_location) == 1  # type: ignore[attr-defined]
    assert updated_table_metadata.current_schema_id == 1
    assert len(updated_table_metadata.schemas) == 2
    new_schema = next(schema for schema in updated_table_metadata.schemas if schema.schema_id == 1)
    assert new_schema
    assert new_schema == update._apply()
    assert new_schema.find_field("b").field_type == IntegerType()
    assert updated_table_metadata.last_updated_ms > last_updated_ms
    assert len(updated_table_metadata.metadata_log) == 1
    assert updated_table_metadata.metadata_log[0].metadata_file == original_table_metadata_location
    assert updated_table_metadata.metadata_log[0].timestamp_ms == original_table_last_updated_ms


def test_catalog_commit_table_applies_schema_updates(
    catalog: Catalog,
    table_schema_nested: Schema,
    test_table_identifier: Identifier,
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_nested)

    new_schema = Schema(
        NestedField(1, "x", LongType()),
        NestedField(2, "y", LongType(), doc="comment"),
        NestedField(3, "z", LongType()),
        NestedField(4, "add", LongType()),
    )

    response = table.catalog.commit_table(
        table,
        updates=(
            AddSchemaUpdate(schema=new_schema),
            SetCurrentSchemaUpdate(),
        ),
        requirements=(),
    )
    assert response.metadata.table_uuid == table.metadata.table_uuid
    assert len(response.metadata.schemas) == 2
    assert response.metadata.schemas[1] == new_schema
    assert response.metadata.current_schema_id == new_schema.schema_id


def test_concurrent_commit_table(catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table_a = catalog.create_table(test_table_identifier, table_schema_nested)
    table_b = catalog.load_table(test_table_identifier)

    with table_a.update_schema() as update:
        update.add_column(path="b", field_type=IntegerType())

    with pytest.raises(CommitFailedException, match="Requirement failed: current schema id has changed: expected 0, found 1"):
        # This one should fail since it already has been updated
        with table_b.update_schema() as update:
            update.add_column(path="c", field_type=IntegerType())


def test_delete_metadata_multiple(catalog: Catalog, table_schema_nested: Schema, random_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(random_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(random_table_identifier, table_schema_nested)

    original_metadata_location = table.metadata_location

    for i in range(5):
        with table.transaction() as transaction:
            with transaction.update_schema() as update:
                update.add_column(path=f"new_column_{i}", field_type=IntegerType())

    assert len(table.metadata.metadata_log) == 5
    assert os.path.exists(original_metadata_location[len("file://") :])

    # Set the max versions property to 2, and delete after commit
    new_property = {
        TableProperties.METADATA_PREVIOUS_VERSIONS_MAX: "2",
        TableProperties.METADATA_DELETE_AFTER_COMMIT_ENABLED: "true",
    }

    with table.transaction() as transaction:
        transaction.set_properties(properties=new_property)

    # Verify that only the most recent metadata files are kept
    assert len(table.metadata.metadata_log) == 2
    updated_metadata_1, updated_metadata_2 = table.metadata.metadata_log

    # new metadata log was added, so earlier metadata logs are removed.
    with table.transaction() as transaction:
        with transaction.update_schema() as update:
            update.add_column(path="new_column_x", field_type=IntegerType())

    assert len(table.metadata.metadata_log) == 2
    assert not os.path.exists(original_metadata_location[len("file://") :])
    assert not os.path.exists(updated_metadata_1.metadata_file[len("file://") :])
    assert os.path.exists(updated_metadata_2.metadata_file[len("file://") :])


# Table properties tests


def test_table_properties_int_value(catalog: Catalog, table_schema_simple: Schema, test_table_identifier: Identifier) -> None:
    # table properties can be set to int, but still serialized to string
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    property_with_int = {"property_name": 42}
    table = catalog.create_table(test_table_identifier, table_schema_simple, properties=property_with_int)
    assert isinstance(table.properties["property_name"], str)


def test_table_properties_raise_for_none_value(
    catalog: Catalog, table_schema_simple: Schema, test_table_identifier: Identifier
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    property_with_none = {"property_name": None}
    with pytest.raises(ValidationError) as exc_info:
        _ = catalog.create_table(test_table_identifier, table_schema_simple, properties=property_with_none)
    assert "None type is not a supported value in properties: property_name" in str(exc_info.value)


# Append table


def test_append_table(catalog: Catalog, table_schema_simple: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(test_table_identifier, table_schema_simple)

    df = pa.Table.from_pydict(
        {
            "foo": ["a"],
            "bar": [1],
            "baz": [True],
        },
        schema=schema_to_pyarrow(table_schema_simple),
    )

    table.append(df)

    # new snapshot is written in APPEND mode
    assert len(table.metadata.snapshots) == 1
    assert table.metadata.snapshots[0].snapshot_id == table.metadata.current_snapshot_id
    assert table.metadata.snapshots[0].parent_snapshot_id is None
    assert table.metadata.snapshots[0].sequence_number == 1
    assert table.metadata.snapshots[0].summary is not None
    assert table.metadata.snapshots[0].summary.operation == Operation.APPEND
    assert table.metadata.snapshots[0].summary["added-data-files"] == "1"
    assert table.metadata.snapshots[0].summary["added-records"] == "1"
    assert table.metadata.snapshots[0].summary["total-data-files"] == "1"
    assert table.metadata.snapshots[0].summary["total-records"] == "1"
    assert len(table.metadata.metadata_log) == 1

    # read back the data
    assert df == table.scan().to_arrow()


# Test writes
def test_table_writes_metadata_to_custom_location(
    catalog: Catalog,
    test_table_identifier: Identifier,
    table_schema_simple: Schema,
    warehouse: Path,
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    metadata_path = f"file://{warehouse}/custom/path"
    catalog.create_namespace(namespace)
    table = catalog.create_table(
        identifier=test_table_identifier,
        schema=table_schema_simple,
        properties={TableProperties.WRITE_METADATA_PATH: metadata_path},
    )
    df = pa.Table.from_pydict(
        {"foo": ["a"], "bar": [1], "baz": [True]},
        schema=schema_to_pyarrow(table_schema_simple),
    )
    table.append(df)
    snapshot = table.current_snapshot()
    assert snapshot is not None
    manifests = snapshot.manifests(table.io)
    location_provider = table.location_provider()

    assert location_provider.new_metadata_location("").startswith(metadata_path)
    assert manifests[0].manifest_path.startswith(metadata_path)
    assert table.location() != metadata_path
    assert table.metadata_location.startswith(metadata_path)


def test_table_writes_metadata_to_default_path(
    catalog: Catalog,
    test_table_identifier: Identifier,
    table_schema_simple: Schema,
    test_table_properties: dict[str, str],
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(
        identifier=test_table_identifier,
        schema=table_schema_simple,
        properties=test_table_properties,
    )
    metadata_path = f"{table.location()}/metadata"
    df = pa.Table.from_pydict(
        {"foo": ["a"], "bar": [1], "baz": [True]},
        schema=schema_to_pyarrow(table_schema_simple),
    )
    table.append(df)
    snapshot = table.current_snapshot()
    assert snapshot is not None
    manifests = snapshot.manifests(table.io)
    location_provider = table.location_provider()

    assert location_provider.new_metadata_location("").startswith(metadata_path)
    assert manifests[0].manifest_path.startswith(metadata_path)
    assert table.metadata_location.startswith(metadata_path)


def test_table_metadata_writes_reflect_latest_path(
    catalog: Catalog,
    test_table_identifier: Identifier,
    table_schema_simple: Schema,
    warehouse: Path,
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(
        identifier=test_table_identifier,
        schema=table_schema_simple,
    )

    initial_metadata_path = f"{table.location()}/metadata"
    assert table.location_provider().new_metadata_location("metadata.json") == f"{initial_metadata_path}/metadata.json"

    # update table with new path for metadata
    new_metadata_path = f"file://{warehouse}/custom/path"
    table.transaction().set_properties({TableProperties.WRITE_METADATA_PATH: new_metadata_path}).commit_transaction()

    assert table.location_provider().new_metadata_location("metadata.json") == f"{new_metadata_path}/metadata.json"


def test_table_writes_gzip_metadata_when_compression_codec_is_set(
    catalog: Catalog,
    test_table_identifier: Identifier,
    table_schema_simple: Schema,
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(
        identifier=test_table_identifier,
        schema=table_schema_simple,
        properties={TableProperties.METADATA_COMPRESSION: "gzip"},
    )
    assert table.metadata_location.endswith(".gz.metadata.json")
    with table.io.new_input(table.metadata_location).open() as stream:
        assert stream.read(2) == b"\x1f\x8b"  # gzip magic number

    df = pa.Table.from_pydict(
        {"foo": ["a"], "bar": [1], "baz": [True]},
        schema=schema_to_pyarrow(table_schema_simple),
    )
    table.append(df)

    # a later commit keeps the codec, and the version is parsed from the gzip file name
    assert table.metadata_location.endswith(".gz.metadata.json")
    assert table.metadata_location.split("/")[-1].startswith("00001-")
    loaded_table = catalog.load_table(test_table_identifier)
    assert loaded_table.metadata_location == table.metadata_location
    assert loaded_table.metadata == table.metadata


def test_table_metadata_compression_codec_default_and_set_on_commit(
    catalog: Catalog,
    test_table_identifier: Identifier,
    table_schema_simple: Schema,
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(identifier=test_table_identifier, schema=table_schema_simple)
    assert table.metadata_location.endswith(".metadata.json")
    assert not table.metadata_location.endswith(".gz.metadata.json")

    # the codec set in a commit applies to the metadata file that the commit writes
    table.transaction().set_properties({TableProperties.METADATA_COMPRESSION: "gzip"}).commit_transaction()
    assert table.metadata_location.endswith(".gz.metadata.json")
    assert catalog.load_table(test_table_identifier).metadata_location == table.metadata_location


def test_table_metadata_invalid_compression_codec_raises(
    catalog: Catalog,
    test_table_identifier: Identifier,
    table_schema_simple: Schema,
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    with pytest.raises(ValueError, match="snappy"):
        catalog.create_table(
            identifier=test_table_identifier,
            schema=table_schema_simple,
            properties={TableProperties.METADATA_COMPRESSION: "snappy"},
        )
    assert not catalog.table_exists(test_table_identifier)

    table = catalog.create_table(identifier=test_table_identifier, schema=table_schema_simple)
    with pytest.raises(ValueError, match="snappy"):
        table.transaction().set_properties({TableProperties.METADATA_COMPRESSION: "snappy"}).commit_transaction()
    assert catalog.load_table(test_table_identifier).metadata_location == table.metadata_location


def test_table_metadata_legacy_invalid_compression_codec_on_commit(
    catalog: Catalog,
    test_table_identifier: Identifier,
    metadata_location: str,
    tmp_path: Path,
) -> None:
    # metadata registered with a codec this version rejects: commits without changes still work
    # the table location is local, so the commit that removes the codec can write its metadata file
    metadata = json.loads(Path(metadata_location).read_text())
    metadata["location"] = f"file://{tmp_path}/table"
    metadata.setdefault("properties", {})[TableProperties.METADATA_COMPRESSION] = "snappy"
    legacy_location = str(tmp_path / "table" / "metadata" / "00000-6c97e413-d51b-4538-ac70-12fe2a85cb83.metadata.json")
    Path(legacy_location).parent.mkdir(parents=True)
    Path(legacy_location).write_text(json.dumps(metadata))
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.register_table(test_table_identifier, legacy_location)

    table.transaction().set_properties({}).commit_transaction()
    assert table.metadata_location == legacy_location

    # a change writes new metadata, so the unsupported codec raises
    with pytest.raises(ValueError, match="snappy"):
        table.transaction().set_properties({"owner": "test"}).commit_transaction()
    assert catalog.load_table(test_table_identifier).metadata_location == legacy_location

    # removing the property writes new metadata with the default codec
    table.transaction().remove_properties(TableProperties.METADATA_COMPRESSION).commit_transaction()
    assert table.metadata_location != legacy_location
    assert table.metadata_location.endswith(".metadata.json")


def test_transaction_writes_manifests_to_staged_metadata_path(
    catalog: Catalog,
    test_table_identifier: Identifier,
    table_schema_simple: Schema,
    warehouse: Path,
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(identifier=test_table_identifier, schema=table_schema_simple)
    metadata_path = f"file://{warehouse}/custom/path"
    df = pa.Table.from_pydict(
        {"foo": ["a"], "bar": [1], "baz": [True]},
        schema=schema_to_pyarrow(table_schema_simple),
    )

    with table.transaction() as txn:
        txn.set_properties({TableProperties.WRITE_METADATA_PATH: metadata_path})
        txn.append(df)

    snapshot = table.current_snapshot()
    assert snapshot is not None
    assert snapshot.manifest_list.startswith(metadata_path)
    assert all(manifest.manifest_path.startswith(metadata_path) for manifest in snapshot.manifests(table.io))
    assert table.metadata_location.startswith(metadata_path)


@pytest.mark.parametrize("format_version", [1, 2])
def test_write_and_evolve(catalog: Catalog, format_version: int) -> None:
    identifier = f"default.arrow_write_data_and_evolve_schema_v{format_version}"

    try:
        catalog.create_namespace("default")
    except NamespaceAlreadyExistsError:
        pass

    try:
        catalog.drop_table(identifier=identifier)
    except NoSuchTableError:
        pass

    pa_table = pa.Table.from_pydict(
        {
            "foo": ["a", None, "z"],
        },
        schema=pa.schema([pa.field("foo", pa.large_string(), nullable=True)]),
    )

    tbl = catalog.create_table(identifier=identifier, schema=pa_table.schema, properties={"format-version": str(format_version)})

    pa_table_with_column = pa.Table.from_pydict(
        {
            "foo": ["a", None, "z"],
            "bar": [19, None, 25],
        },
        schema=pa.schema(
            [
                pa.field("foo", pa.large_string(), nullable=True),
                pa.field("bar", pa.int32(), nullable=True),
            ]
        ),
    )

    with tbl.transaction() as txn:
        with txn.update_schema() as schema_txn:
            schema_txn.union_by_name(pa_table_with_column.schema)

        txn.append(pa_table_with_column)
        txn.overwrite(pa_table_with_column)
        txn.delete("foo = 'a'")


def test_dynamic_partition_overwrite_after_schema_update_in_transaction(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    schema = Schema(NestedField(1, "id", LongType()), NestedField(2, "region", StringType()))
    partition_spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region"))
    tbl = catalog.create_table("default.dynamic_partition_overwrite_schema", schema=schema, partition_spec=partition_spec)
    tbl.append(pa.table({"id": [1, 2], "region": ["eu", "us"]}))

    with tbl.transaction() as txn:
        with txn.update_schema() as update:
            update.add_column("data", StringType())
        txn.dynamic_partition_overwrite(pa.table({"id": [3], "region": ["eu"], "data": ["x"]}))

    assert sorted(tbl.scan().to_arrow().to_pylist(), key=lambda row: row["id"]) == [
        {"id": 2, "region": "us", "data": None},
        {"id": 3, "region": "eu", "data": "x"},
    ]


def test_dynamic_partition_overwrite_after_spec_update_in_transaction(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "id", LongType()), NestedField(2, "region", StringType()), NestedField(3, "city", StringType())
    )
    partition_spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region"))
    tbl = catalog.create_table("default.dynamic_partition_overwrite_spec", schema=schema, partition_spec=partition_spec)
    tbl.append(pa.table({"id": [1, 2], "region": ["eu", "us"], "city": ["Amsterdam", "Boston"]}))

    with tbl.transaction() as txn:
        with txn.update_spec() as update:
            update.remove_field("region").add_identity("city")
        txn.dynamic_partition_overwrite(pa.table({"id": [3, 4], "region": ["eu", "eu"], "city": ["Amsterdam", "Paris"]}))

    assert sorted(tbl.scan().to_arrow()["id"].to_pylist()) == [2, 3, 4]
    assert {(task.file.spec_id, task.file.partition) for task in tbl.scan().plan_files()} == {
        (0, Record("us")),
        (1, Record("Amsterdam")),
        (1, Record("Paris")),
    }


# Merge manifests
@pytest.mark.parametrize("format_version", [1, 2])
def test_merge_manifests_local_file_system(catalog: Catalog, arrow_table_with_null: pa.Table, format_version: int) -> None:
    # To catch manifest file name collision bug during merge:
    # https://github.com/apache/iceberg-python/pull/363#discussion_r1660691918
    catalog.create_namespace_if_not_exists("default")
    try:
        catalog.drop_table("default.test_merge_manifest")
    except NoSuchTableError:
        pass
    tbl = catalog.create_table(
        "default.test_merge_manifest",
        arrow_table_with_null.schema,
        properties={
            "commit.manifest-merge.enabled": "true",
            "commit.manifest.min-count-to-merge": "2",
            "format-version": format_version,
        },
    )

    for _ in range(5):
        tbl.append(arrow_table_with_null)

    assert len(tbl.scan().to_arrow()) == 5 * len(arrow_table_with_null)
    current_snapshot = tbl.current_snapshot()
    assert current_snapshot
    manifests = current_snapshot.manifests(tbl.io)
    assert len(manifests) == 1


# Add column to table


def test_add_column(catalog: Catalog, table_schema_simple: Schema, random_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(random_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(random_table_identifier, table_schema_simple)
    table.update_schema().add_column(path="new_column1", field_type=IntegerType()).commit()
    assert table.schema() == Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        NestedField(field_id=4, name="new_column1", field_type=IntegerType(), required=False),
        schema_id=1,
        identifier_field_ids=[2],
    )
    assert table.schema().schema_id == 1

    transaction = table.transaction()
    transaction.update_schema().add_column(path="new_column2", field_type=IntegerType(), doc="doc").commit()
    transaction.commit_transaction()

    assert table.schema() == Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        NestedField(field_id=4, name="new_column1", field_type=IntegerType(), required=False),
        NestedField(field_id=5, name="new_column2", field_type=IntegerType(), required=False, doc="doc"),
        identifier_field_ids=[2],
    )
    assert table.schema().schema_id == 2


def test_add_column_with_statement(catalog: Catalog, table_schema_simple: Schema, random_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(random_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(random_table_identifier, table_schema_simple)

    with table.update_schema() as tx:
        tx.add_column(path="new_column1", field_type=IntegerType())

    assert table.schema() == Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        NestedField(field_id=4, name="new_column1", field_type=IntegerType(), required=False),
        identifier_field_ids=[2],
    )
    assert table.schema().schema_id == 1

    with table.transaction() as tx:
        tx.update_schema().add_column(path="new_column2", field_type=IntegerType(), doc="doc").commit()

    assert table.schema() == Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        NestedField(field_id=4, name="new_column1", field_type=IntegerType(), required=False),
        NestedField(field_id=5, name="new_column2", field_type=IntegerType(), required=False, doc="doc"),
        identifier_field_ids=[2],
    )
    assert table.schema().schema_id == 2


def test_update_schema_with_statement_does_not_commit_on_exception(
    catalog: Catalog, table_schema_simple: Schema, random_table_identifier: Identifier
) -> None:
    namespace = Catalog.namespace_from(random_table_identifier)
    catalog.create_namespace(namespace)
    table = catalog.create_table(random_table_identifier, table_schema_simple)

    with pytest.raises(ValueError):
        with table.update_schema() as tx:
            tx.add_column(path="should_not_commit", field_type=IntegerType())
            int("boom")

    assert table.schema() == table_schema_simple
    assert table.schema().schema_id == 0

    reloaded = catalog.load_table(random_table_identifier)
    assert reloaded.schema() == table_schema_simple
    assert reloaded.schema().schema_id == 0


# Namespace tests


def test_create_namespace(catalog: Catalog, test_namespace: Identifier, test_table_properties: dict[str, str]) -> None:
    catalog.create_namespace(test_namespace, test_table_properties)
    assert catalog.namespace_exists(test_namespace)
    assert (Catalog.identifier_to_tuple(test_namespace)[:1]) in catalog.list_namespaces()
    assert test_table_properties == catalog.load_namespace_properties(test_namespace)


def test_create_namespace_raises_error_on_existing_namespace(
    catalog: Catalog, test_namespace: Identifier, test_table_properties: dict[str, str]
) -> None:
    catalog.create_namespace(test_namespace, test_table_properties)
    with pytest.raises(NamespaceAlreadyExistsError):
        catalog.create_namespace(test_namespace, test_table_properties)


def test_create_namespace_if_not_exists(catalog: Catalog, database_name: str) -> None:
    catalog.create_namespace(database_name)
    assert (database_name,) in catalog.list_namespaces()
    catalog.create_namespace_if_not_exists(database_name)
    assert (database_name,) in catalog.list_namespaces()


def test_create_namespaces_sharing_same_prefix(catalog: Catalog, test_namespace: Identifier) -> None:
    child_namespace = test_namespace + ("child",)
    # Parent first
    catalog.create_namespace(test_namespace)
    # Then child
    catalog.create_namespace(child_namespace)


def test_create_namespace_with_comment_and_location(catalog: Catalog, test_namespace: Identifier) -> None:
    test_location = "/test/location"
    test_properties = {
        "comment": "this is a test description",
        "location": test_location,
    }
    catalog.create_namespace(namespace=test_namespace, properties=test_properties)
    loaded_database_list = catalog.list_namespaces()
    assert Catalog.identifier_to_tuple(test_namespace)[:1] in loaded_database_list
    properties = catalog.load_namespace_properties(test_namespace)
    assert properties["comment"] == "this is a test description"
    assert properties["location"] == test_location


@pytest.mark.filterwarnings("ignore")
def test_create_namespace_with_null_properties(catalog: Catalog, test_namespace: Identifier) -> None:
    with pytest.raises(IntegrityError):
        catalog.create_namespace(namespace=test_namespace, properties={None: "value"})  # type: ignore

    with pytest.raises(IntegrityError):
        catalog.create_namespace(namespace=test_namespace, properties={"key": None})


@pytest.mark.parametrize("empty_namespace", ["", (), (""), ("", ""), " ", (" ")])
def test_create_namespace_with_empty_identifier(catalog: Catalog, empty_namespace: Any) -> None:
    with pytest.raises(NoSuchNamespaceError):
        catalog.create_namespace(empty_namespace)


# Get namespace tests


def test_get_namespace_metadata_raises_error_when_namespace_does_not_exist(catalog: Catalog, test_namespace: Identifier) -> None:
    namespace = ".".join(test_namespace)
    with pytest.raises(NoSuchNamespaceError, match=f"Namespace {namespace} does not exists"):
        catalog.load_namespace_properties(test_namespace)


def test_namespace_exists(catalog: Catalog) -> None:
    for ns in [("db1",), ("db1", "ns1"), ("db2", "ns1"), ("db3", "ns1", "ns2")]:
        catalog.create_namespace(ns)
        assert catalog.namespace_exists(ns)

    # `db2` exists because `db2.ns1` exists
    assert catalog.namespace_exists("db2")
    # `db3.ns1` exists because `db3.ns1.ns2` exists
    assert catalog.namespace_exists("db3.ns1")
    # make sure '_' is escaped in the query
    assert not catalog.namespace_exists("db_")
    # make sure '%' is escaped in the query
    assert not catalog.namespace_exists("db%")


# Namespace properties


def test_load_namespace_properties(catalog: Catalog, test_namespace: Identifier) -> None:
    warehouse_location = "/test/location"
    test_properties = {
        "comment": "this is a test description",
        "location": f"{warehouse_location}/{test_namespace}",
        "test_property1": "1",
        "test_property2": "2",
        "test_property3": "3",
    }

    catalog.create_namespace(test_namespace, test_properties)
    listed_properties = catalog.load_namespace_properties(test_namespace)
    for k, v in listed_properties.items():
        assert k in test_properties
        assert v == test_properties[k]


def test_load_namespace_properties_non_existing_namespace(catalog: Catalog) -> None:
    with pytest.raises(NoSuchNamespaceError):
        catalog.load_namespace_properties("does_not_exist")


def test_load_empty_namespace_properties(catalog: Catalog, test_namespace: Identifier) -> None:
    catalog.create_namespace(test_namespace)
    listed_properties = catalog.load_namespace_properties(test_namespace)
    assert listed_properties == {"exists": "true"}


# List namespaces tests


def test_list_namespaces(catalog: Catalog) -> None:
    namespace_list = ["db", "db.ns1", "db.ns1.ns2", "db.ns2", "db2", "db2.ns1", "db%"]
    for namespace in namespace_list:
        if not catalog.namespace_exists(namespace):
            catalog.create_namespace(namespace)

    ns_list = catalog.list_namespaces()
    for ns in [("db",), ("db%",), ("db2",)]:
        assert ns in ns_list

    ns_list = catalog.list_namespaces("db")
    assert sorted(ns_list) == [("db", "ns1"), ("db", "ns2")]

    ns_list = catalog.list_namespaces("db.ns1")
    assert sorted(ns_list) == [("db", "ns1", "ns2")]

    ns_list = catalog.list_namespaces("db.ns1.ns2")
    assert len(ns_list) == 0


def test_list_namespaces_fuzzy_match(catalog: Catalog) -> None:
    namespace_list = ["db.ns1", "db.ns1.ns2", "db.ns2", "db.ns1X.ns3", "db_.ns1.ns2", "db2.ns1.ns2"]
    for namespace in namespace_list:
        if not catalog.namespace_exists(namespace):
            catalog.create_namespace(namespace)

    assert catalog.list_namespaces("db.ns1") == [("db", "ns1", "ns2")]

    assert catalog.list_namespaces("db_.ns1") == [("db_", "ns1", "ns2")]


def test_list_non_existing_namespaces(catalog: Catalog) -> None:
    with pytest.raises(NoSuchNamespaceError):
        catalog.list_namespaces("does_not_exist")


# Update namespace properties tests


def test_update_namespace_properties(catalog: Catalog, test_namespace: Identifier) -> None:
    warehouse_location = "/test/location"
    test_properties = {
        "comment": "this is a test description",
        "location": f"{warehouse_location}/{test_namespace}",
        "test_property1": "1",
        "test_property2": "2",
        "test_property3": "3",
    }
    removals = {"test_property1", "test_property2", "test_property3", "should_not_removed"}
    updates = {"test_property4": "4", "test_property5": "5", "comment": "updated test description"}
    catalog.create_namespace(test_namespace, test_properties)
    update_report = catalog.update_namespace_properties(test_namespace, removals, updates)
    for k in updates.keys():
        assert k in update_report.updated
    for k in removals:
        if k == "should_not_removed":
            assert k in update_report.missing
        else:
            assert k in update_report.removed
    assert catalog.load_namespace_properties(test_namespace) == {
        "comment": "updated test description",
        "test_property4": "4",
        "test_property5": "5",
        "location": f"{warehouse_location}/{test_namespace}",
    }


def test_update_namespace_metadata_raises_error_when_namespace_does_not_exist(
    catalog: Catalog, test_namespace: Identifier, test_table_properties: dict[str, str]
) -> None:
    namespace = ".".join(test_namespace)
    with pytest.raises(NoSuchNamespaceError, match=f"Namespace {namespace} does not exists"):
        catalog.update_namespace_properties(test_namespace, updates=test_table_properties)


def test_update_namespace_metadata(catalog: Catalog, test_namespace: Identifier, test_table_properties: dict[str, str]) -> None:
    catalog.create_namespace(test_namespace, test_table_properties)
    new_metadata = {"key3": "value3", "key4": "value4"}
    summary = catalog.update_namespace_properties(test_namespace, updates=new_metadata)
    assert catalog.namespace_exists(test_namespace)
    assert new_metadata.items() <= catalog.load_namespace_properties(test_namespace).items()
    assert summary.removed == []
    assert sorted(summary.updated) == ["key3", "key4"]
    assert summary.missing == []


def test_update_namespace_metadata_removals(
    catalog: Catalog, test_namespace: Identifier, test_table_properties: dict[str, str]
) -> None:
    catalog.create_namespace(test_namespace, test_table_properties)
    new_metadata = {"key3": "value3", "key4": "value4"}
    remove_metadata = {"key1"}
    summary = catalog.update_namespace_properties(test_namespace, remove_metadata, new_metadata)
    assert catalog.namespace_exists(test_namespace)
    assert new_metadata.items() <= catalog.load_namespace_properties(test_namespace).items()
    assert remove_metadata.isdisjoint(catalog.load_namespace_properties(test_namespace).keys())
    assert summary.removed == ["key1"]
    assert sorted(summary.updated) == ["key3", "key4"]
    assert summary.missing == []


# Drop namespace tests


def test_drop_namespace(catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    assert catalog.namespace_exists(namespace)
    catalog.create_table(test_table_identifier, table_schema_nested)
    with pytest.raises(NamespaceNotEmptyError):
        catalog.drop_namespace(namespace)
    catalog.drop_table(test_table_identifier)
    catalog.drop_namespace(namespace)
    assert not catalog.namespace_exists(namespace)


def test_drop_namespace_raises_error_when_namespace_does_not_exist(catalog: Catalog) -> None:
    with pytest.raises(NoSuchNamespaceError):
        catalog.drop_namespace("does_not_exist")


def test_drop_namespace_raises_error_when_namespace_not_empty(
    catalog: Catalog, table_schema_nested: Schema, test_table_identifier: Identifier
) -> None:
    namespace = Catalog.namespace_from(test_table_identifier)
    catalog.create_namespace(namespace)
    catalog.create_table(test_table_identifier, table_schema_nested)
    with pytest.raises(NamespaceNotEmptyError, match=f"Namespace {'.'.join(namespace)} is not empty"):
        catalog.drop_namespace(namespace)


# RecordBatchReader streaming append/overwrite tests
#
# Streaming writes accept a pa.RecordBatchReader and microbatch it into target-sized
# Parquet files instead of materialising the full Arrow Table in memory. Tracks
# https://github.com/apache/iceberg-python/issues/2152.


def _simple_arrow_table() -> pa.Table:
    return pa.Table.from_pydict(
        {"foo": ["a", None, "z"]},
        schema=pa.schema([pa.field("foo", pa.large_string(), nullable=True)]),
    )


def _simple_record_batch_reader(num_batches: int = 3) -> tuple[pa.RecordBatchReader, int]:
    """Build an N-batch reader of the simple schema. Returns (reader, total_rows)."""
    pa_table = _simple_arrow_table()
    batches = pa_table.to_batches() * num_batches
    reader = pa.RecordBatchReader.from_batches(pa_table.schema, iter(batches))
    return reader, sum(b.num_rows for b in batches)


def test_append_record_batch_reader(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    identifier = f"default.append_record_batch_reader_{catalog.name}"
    reader, total_rows = _simple_record_batch_reader(num_batches=3)
    tbl = catalog.create_table(identifier=identifier, schema=reader.schema)

    tbl.append(reader)

    assert len(tbl.scan().to_arrow()) == total_rows


def test_append_record_batch_reader_microbatched(catalog: Catalog) -> None:
    """A reader bigger than the per-file target produces multiple Parquet files
    in a single snapshot — verifying the byte-budget microbatching path."""
    catalog.create_namespace("default")
    identifier = f"default.append_record_batch_reader_microbatch_{catalog.name}"
    reader, total_rows = _simple_record_batch_reader(num_batches=8)
    # Force every batch to roll a new file by setting an absurdly small target size.
    tbl = catalog.create_table(
        identifier=identifier,
        schema=reader.schema,
        properties={TableProperties.WRITE_TARGET_FILE_SIZE_BYTES: "1"},
    )

    tbl.append(reader)

    snapshot = tbl.metadata.current_snapshot()
    assert snapshot is not None
    assert snapshot.summary is not None
    added_files = snapshot.summary["added-data-files"]
    assert added_files is not None and int(added_files) > 1, snapshot.summary
    assert len(tbl.scan().to_arrow()) == total_rows


def test_append_record_batch_reader_empty(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    identifier = f"default.append_record_batch_reader_empty_{catalog.name}"
    schema = _simple_arrow_table().schema
    reader = pa.RecordBatchReader.from_batches(schema, iter([]))
    tbl = catalog.create_table(identifier=identifier, schema=schema)

    tbl.append(reader)

    assert len(tbl.scan().to_arrow()) == 0


def test_overwrite_record_batch_reader(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    identifier = f"default.overwrite_record_batch_reader_{catalog.name}"
    pa_table = _simple_arrow_table()
    tbl = catalog.create_table(identifier=identifier, schema=pa_table.schema)
    tbl.append(pa_table)
    assert len(tbl.scan().to_arrow()) == pa_table.num_rows

    reader, total_rows = _simple_record_batch_reader(num_batches=2)
    tbl.overwrite(reader)

    assert len(tbl.scan().to_arrow()) == total_rows


def test_append_record_batch_reader_to_partitioned_table_raises(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    identifier = f"default.append_record_batch_reader_partitioned_{catalog.name}"
    iceberg_schema = Schema(
        NestedField(1, "id", IntegerType(), required=False),
        NestedField(2, "bucket", StringType(), required=False),
    )
    partition_spec = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="bucket"),
    )
    tbl = catalog.create_table(identifier=identifier, schema=iceberg_schema, partition_spec=partition_spec)

    arrow_schema = schema_to_pyarrow(iceberg_schema)
    reader = pa.RecordBatchReader.from_batches(arrow_schema, iter([]))
    with pytest.raises(NotImplementedError, match="partitioned table"):
        tbl.append(reader)


def test_append_invalid_input_type_raises(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    identifier = f"default.append_invalid_input_{catalog.name}"
    pa_table = _simple_arrow_table()
    tbl = catalog.create_table(identifier=identifier, schema=pa_table.schema)
    with pytest.raises(ValueError, match="Expected pa.Table or pa.RecordBatchReader"):
        tbl.append("not an arrow object")


def test_append_nan_to_identity_partitioned_table(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    identifier = f"default.append_nan_identity_partition_{catalog.name}"
    iceberg_schema = Schema(
        NestedField(1, "id", IntegerType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
    )
    partition_spec = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="value"),
    )
    tbl = catalog.create_table(identifier=identifier, schema=iceberg_schema, partition_spec=partition_spec)

    tbl.append(
        pa.Table.from_pydict(
            {"id": [1, 2, 3, 4], "value": [1.0, float("nan"), None, float("nan")]},
            schema=schema_to_pyarrow(iceberg_schema),
        )
    )

    assert sorted(tbl.scan().to_arrow()["id"].to_pylist()) == [1, 2, 3, 4]
    assert sorted(tbl.scan(row_filter="value is nan").to_arrow()["id"].to_pylist()) == [2, 4]


def _table_with_null_partition(catalog: Catalog, name: str, partition_spec: PartitionSpec) -> Table:
    schema = Schema(NestedField(1, "p", StringType(), required=False), NestedField(2, "f", DoubleType(), required=False))
    catalog.create_namespace_if_not_exists("default")
    tbl = catalog.create_table(identifier=f"default.{name}_{catalog.name}", schema=schema, partition_spec=partition_spec)
    tbl.append(
        pa.Table.from_pydict(
            {
                "p": ["a", "a", "b", "b", "c", "c", None, None, "b", "b"],
                "f": [1.0, 1.0, 2.0, 2.0, float("nan"), float("nan"), 1.0, 1.0, None, None],
            },
            schema=schema_to_pyarrow(schema),
        )
    )
    return tbl


def test_count_matches_to_arrow_with_null_partition_values(catalog: Catalog) -> None:
    identity = PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="p"),
        PartitionField(source_id=2, field_id=1001, transform=IdentityTransform(), name="f"),
    )
    bucket = PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=BucketTransform(4), name="p_bucket"),
        PartitionField(source_id=2, field_id=1001, transform=IdentityTransform(), name="f"),
    )
    evolved = _table_with_null_partition(
        catalog,
        "count_null_partition_evolved",
        PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="p")),
    )
    # Files written under the old spec keep a NULL p partition value that the current spec no longer has
    with evolved.update_spec() as update:
        update.remove_field("p")
    evolved.append(evolved.scan().to_arrow())
    tables = [
        _table_with_null_partition(catalog, "count_null_partition_identity", identity),
        _table_with_null_partition(catalog, "count_null_partition_bucket", bucket),
        evolved,
    ]
    row_filters = ["p != 'a'", "p not in ('a', 'c')", "not p = 'a'", "p not like 'a%'", "f is not nan", "p is null", "true"]
    for tbl in tables:
        for row_filter in row_filters:
            scan = tbl.scan(row_filter=row_filter)
            assert scan.count() == len(scan.to_arrow()), (tbl.name(), row_filter)


def test_count_reads_null_partition_only_when_filter_compares_it(catalog: Catalog) -> None:
    partition_spec = PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="p"),
        PartitionField(source_id=2, field_id=1001, transform=IdentityTransform(), name="f"),
    )
    # Each row filter maps to (filter selecting files to delete, expected count), and count() must not read those files
    cases = {
        "true": ("true", 10),
        "p is null": ("p is null", 2),
        "p is not null or f = 1.0": ("p is null", 10),
        "p != 'a'": ("p is not null", 6),
        "f = 1.0": ("f = 1.0", 4),
    }
    for i, (row_filter, (metadata_only_filter, expected)) in enumerate(cases.items()):
        tbl = _table_with_null_partition(catalog, f"count_metadata_only_{i}", partition_spec)
        # Deleting those files makes any read of them fail
        for task in tbl.scan(row_filter=metadata_only_filter).plan_files():
            tbl.io.delete(task.file.file_path)
        assert tbl.scan(row_filter=row_filter).count() == expected, row_filter


def _nan_partitioned_table(catalog: Catalog, name: str) -> Table:
    iceberg_schema = Schema(
        NestedField(1, "id", IntegerType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
    )
    partition_spec = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="value"),
    )
    tbl = catalog.create_table(f"default.{name}_{catalog.name}", schema=iceberg_schema, partition_spec=partition_spec)
    tbl.append(_id_value_rows(tbl, [1, 2, 3, 4], [1.0, 1.0, float("nan"), float("nan")]))
    return tbl


def _id_value_rows(tbl: Table, ids: list[int], values: list[float]) -> pa.Table:
    return pa.Table.from_pydict({"id": ids, "value": values}, schema=schema_to_pyarrow(tbl.schema()))


def test_replace_files_in_nan_identity_partition(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    cases: list[tuple[str, Callable[[Table], None], list[int], list[int]]] = [
        ("delete_nan_file", lambda t: t.delete("value is nan"), [1, 2], []),
        ("rewrite_nan_file", lambda t: t.delete("id = 3"), [1, 2, 4], [4]),
        ("overwrite", lambda t: t.overwrite(_id_value_rows(t, [9], [2.0])), [9], []),
        (
            "dynamic_overwrite",
            lambda t: t.dynamic_partition_overwrite(_id_value_rows(t, [9], [float("nan")])),
            [1, 2, 9],
            [9],
        ),
    ]
    for name, operation, expected_ids, expected_nan_ids in cases:
        tbl = _nan_partitioned_table(catalog, name)

        operation(tbl)

        assert sorted(tbl.scan().to_arrow()["id"].to_pylist()) == expected_ids, name
        assert sorted(tbl.scan(row_filter="value is nan").to_arrow()["id"].to_pylist()) == expected_nan_ids, name


def test_scan_integer_column_with_decimal_literal(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    identifier = f"default.scan_integer_decimal_literal_{catalog.name}"
    tbl = catalog.create_table(identifier=identifier, schema=pa.schema([pa.field("x", pa.int32())]))
    tbl.append(pa.table({"x": pa.array([1, 2, 3, 4], pa.int32())}))

    # A literal with no fractional part, such as 2.0, still filters exactly
    assert sorted(tbl.scan(row_filter="x > 2.0").to_arrow()["x"].to_pylist()) == [3, 4]
    assert tbl.scan(row_filter="x = 2.0").to_arrow()["x"].to_pylist() == [2]
    # Rounding 2.6 to 3 would turn x > 2.6 into x > 3 and silently drop x = 3
    with pytest.raises(ValueError, match="Could not convert 2.6 into a int"):
        tbl.scan(row_filter="x > 2.6").to_arrow()


def test_scan_null_filter_on_list_and_map_columns(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    data = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int32()),
            "tags": pa.array([["a"], [], None], pa.list_(pa.string())),
            "props": pa.array([[("k", 1)], [], None], pa.map_(pa.string(), pa.int32())),
            "points": pa.array([[{"x": 1}], [], None], pa.list_(pa.struct([("x", pa.int32())]))),
        }
    )
    tbl = catalog.create_table(f"default.scan_null_filter_list_map_{catalog.name}", schema=data.schema)
    tbl.append(data)

    for column in ("tags", "props", "points"):
        assert tbl.scan(row_filter=f"{column} is null").to_arrow()["id"].to_pylist() == [3], column
        assert sorted(tbl.scan(row_filter=f"{column} is not null").to_arrow()["id"].to_pylist()) == [1, 2], column
        assert tbl.scan(row_filter=f"{column} is null", selected_fields=("id",)).to_arrow().to_pydict() == {"id": [3]}, column


def test_scan_negated_comparison_matches_file_pruning_for_nan(catalog: Catalog) -> None:
    catalog.create_namespace("default")
    identifier = f"default.scan_negated_comparison_nan_{catalog.name}"
    iceberg_schema = Schema(
        NestedField(1, "id", IntegerType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
    )
    tbl = catalog.create_table(identifier=identifier, schema=iceberg_schema)
    arrow_schema = schema_to_pyarrow(iceberg_schema)
    # value >= 6.0 keeps the first file (upper bound 7.0) and skips the second (upper bound 5.0)
    tbl.append(pa.Table.from_pydict({"id": [1, 2], "value": [float("nan"), 7.0]}, schema=arrow_schema))
    tbl.append(pa.Table.from_pydict({"id": [3, 4], "value": [float("nan"), 5.0]}, schema=arrow_schema))

    # NOT is rewritten to the complementary predicate at the file and at the row level, so NaN does not match
    # value >= 6.0 in either file
    for row_filter, expected_ids in [
        ("not value < 6.0", [2]),
        ("value >= 6.0", [2]),
        ("not (value < 6.0 and id = 1)", [2, 3, 4]),
        ("value >= 6.0 or id != 1", [2, 3, 4]),
    ]:
        assert tbl.scan(row_filter=row_filter).to_arrow().sort_by("id")["id"].to_pylist() == expected_ids, row_filter


def test_record_batch_reader_consumed_exactly_once(catalog: Catalog) -> None:
    """The streaming path must consume the underlying generator exactly once.
    A regression that drained the reader twice (e.g. an extra .schema access
    that materialised the iterator, or a retry-loop without a fresh reader)
    would silently lose data — the second pass is empty.
    """
    catalog.create_namespace("default")
    identifier = f"default.record_batch_reader_consumed_once_{catalog.name}"
    pa_table = _simple_arrow_table()
    consumed_batches = 0

    def tracking_batches() -> Generator[pa.RecordBatch, None, None]:
        nonlocal consumed_batches
        for batch in pa_table.to_batches() * 3:
            consumed_batches += 1
            yield batch

    reader = pa.RecordBatchReader.from_batches(pa_table.schema, tracking_batches())
    tbl = catalog.create_table(identifier=identifier, schema=pa_table.schema)

    tbl.append(reader)

    # The generator should have been driven to exhaustion exactly once: 3 batches.
    assert consumed_batches == 3
    assert len(tbl.scan().to_arrow()) == pa_table.num_rows * 3


def test_record_batch_reader_schema_mismatch_writes_no_files(catalog: Catalog) -> None:
    """A schema mismatch must fail before any data files are written. Otherwise
    we'd leak orphan parquet files in storage (and a partial commit that picks
    them up later via add_files would be a correctness disaster).
    """
    catalog.create_namespace("default")
    identifier = f"default.record_batch_reader_schema_mismatch_{catalog.name}"
    iceberg_schema = Schema(NestedField(1, "foo", StringType(), required=False))
    tbl = catalog.create_table(identifier=identifier, schema=iceberg_schema)

    bad_schema = pa.schema([pa.field("foo", pa.int64(), nullable=True)])
    bad_reader = pa.RecordBatchReader.from_batches(
        bad_schema,
        iter([pa.RecordBatch.from_pylist([{"foo": 1}], schema=bad_schema)]),
    )

    with pytest.raises(ValueError):
        tbl.append(bad_reader)

    # No snapshot should have been produced: the schema check runs before
    # _append_snapshot_producer opens.
    assert tbl.metadata.current_snapshot() is None
    assert len(tbl.scan().to_arrow()) == 0
