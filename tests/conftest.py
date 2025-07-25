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
# pylint:disable=redefined-outer-name
"""This contains global pytest configurations.

Fixtures contained in this file will be automatically used if provided as an argument
to any pytest function.

In the case where the fixture must be used in a pytest.mark.parametrize decorator, the string representation can be used
and the built-in pytest fixture request should be used as an additional argument in the function. The fixture can then be
retrieved using `request.getfixturevalue(fixture_name)`.
"""

import os
import re
import socket
import string
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from random import choice, randint
from tempfile import TemporaryDirectory
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
)

import boto3
import pytest
from moto import mock_aws

from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.catalog.noop import NoopCatalog
from pyiceberg.expressions import BoundReference
from pyiceberg.io import (
    ADLS_ACCOUNT_KEY,
    ADLS_ACCOUNT_NAME,
    ADLS_BLOB_STORAGE_AUTHORITY,
    ADLS_BLOB_STORAGE_SCHEME,
    ADLS_DFS_STORAGE_AUTHORITY,
    ADLS_DFS_STORAGE_SCHEME,
    GCS_PROJECT_ID,
    GCS_SERVICE_HOST,
    GCS_TOKEN,
    GCS_TOKEN_EXPIRES_AT_MS,
    fsspec,
    load_file_io,
)
from pyiceberg.io.fsspec import FsspecFileIO
from pyiceberg.manifest import DataFile, FileFormat
from pyiceberg.schema import Accessor, Schema
from pyiceberg.serializers import ToOutputFile
from pyiceberg.table import FileScanTask, Table
from pyiceberg.table.metadata import TableMetadataV1, TableMetadataV2
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FixedType,
    FloatType,
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
    TimestampType,
    TimestamptzType,
    TimeType,
    UUIDType,
)
from pyiceberg.utils.datetime import datetime_to_millis

if TYPE_CHECKING:
    import pyarrow as pa
    from moto.server import ThreadedMotoServer  # type: ignore
    from pyspark.sql import SparkSession

    from pyiceberg.io.pyarrow import PyArrowFileIO


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("unmarked")


def pytest_addoption(parser: pytest.Parser) -> None:
    # S3 options
    parser.addoption(
        "--s3.endpoint", action="store", default="http://localhost:9000", help="The S3 endpoint URL for tests marked as s3"
    )
    parser.addoption("--s3.access-key-id", action="store", default="admin", help="The AWS access key ID for tests marked as s3")
    parser.addoption(
        "--s3.secret-access-key", action="store", default="password", help="The AWS secret access key ID for tests marked as s3"
    )
    # ADLS options
    # Azurite provides default account name and key.  Those can be customized using env variables.
    # For more information, see README file at https://github.com/azure/azurite#default-storage-account
    parser.addoption(
        "--adls.endpoint",
        action="store",
        default="http://127.0.0.1:10000",
        help="The ADLS endpoint URL for tests marked as adls",
    )
    parser.addoption(
        "--adls.account-name", action="store", default="devstoreaccount1", help="The ADLS account key for tests marked as adls"
    )
    parser.addoption(
        "--adls.account-key",
        action="store",
        default="Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
        help="The ADLS secret account key for tests marked as adls",
    )
    parser.addoption(
        "--gcs.endpoint", action="store", default="http://0.0.0.0:4443", help="The GCS endpoint URL for tests marked gcs"
    )
    parser.addoption(
        "--gcs.oauth2.token", action="store", default="anon", help="The GCS authentication method for tests marked gcs"
    )
    parser.addoption("--gcs.project-id", action="store", default="test", help="The GCP project for tests marked gcs")


@pytest.fixture(scope="session")
def table_schema_simple() -> Schema:
    return Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        schema_id=1,
        identifier_field_ids=[2],
    )


@pytest.fixture(scope="session")
def table_schema_with_full_nested_fields() -> Schema:
    return Schema(
        NestedField(
            field_id=1,
            name="foo",
            field_type=StringType(),
            required=False,
            doc="foo doc",
            initial_default="foo initial",
            write_default="foo write",
        ),
        NestedField(
            field_id=2, name="bar", field_type=IntegerType(), required=True, doc="bar doc", initial_default=42, write_default=43
        ),
        NestedField(
            field_id=3,
            name="baz",
            field_type=BooleanType(),
            required=False,
            doc="baz doc",
            initial_default=True,
            write_default=False,
        ),
        schema_id=1,
        identifier_field_ids=[2],
    )


@pytest.fixture(scope="session")
def table_schema_nested() -> Schema:
    return Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        NestedField(
            field_id=4,
            name="qux",
            field_type=ListType(element_id=5, element_type=StringType(), element_required=True),
            required=True,
        ),
        NestedField(
            field_id=6,
            name="quux",
            field_type=MapType(
                key_id=7,
                key_type=StringType(),
                value_id=8,
                value_type=MapType(key_id=9, key_type=StringType(), value_id=10, value_type=IntegerType(), value_required=True),
                value_required=True,
            ),
            required=True,
        ),
        NestedField(
            field_id=11,
            name="location",
            field_type=ListType(
                element_id=12,
                element_type=StructType(
                    NestedField(field_id=13, name="latitude", field_type=FloatType(), required=False),
                    NestedField(field_id=14, name="longitude", field_type=FloatType(), required=False),
                ),
                element_required=True,
            ),
            required=True,
        ),
        NestedField(
            field_id=15,
            name="person",
            field_type=StructType(
                NestedField(field_id=16, name="name", field_type=StringType(), required=False),
                NestedField(field_id=17, name="age", field_type=IntegerType(), required=True),
            ),
            required=False,
        ),
        schema_id=1,
        identifier_field_ids=[2],
    )


@pytest.fixture(scope="session")
def table_schema_nested_with_struct_key_map() -> Schema:
    return Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=True),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        NestedField(
            field_id=4,
            name="qux",
            field_type=ListType(element_id=5, element_type=StringType(), element_required=True),
            required=True,
        ),
        NestedField(
            field_id=6,
            name="quux",
            field_type=MapType(
                key_id=7,
                key_type=StringType(),
                value_id=8,
                value_type=MapType(key_id=9, key_type=StringType(), value_id=10, value_type=IntegerType(), value_required=True),
                value_required=True,
            ),
            required=True,
        ),
        NestedField(
            field_id=11,
            name="location",
            field_type=MapType(
                key_id=18,
                value_id=19,
                key_type=StructType(
                    NestedField(field_id=21, name="address", field_type=StringType(), required=True),
                    NestedField(field_id=22, name="city", field_type=StringType(), required=True),
                    NestedField(field_id=23, name="zip", field_type=IntegerType(), required=True),
                ),
                value_type=StructType(
                    NestedField(field_id=13, name="latitude", field_type=FloatType(), required=True),
                    NestedField(field_id=14, name="longitude", field_type=FloatType(), required=True),
                ),
                value_required=True,
            ),
            required=True,
        ),
        NestedField(
            field_id=15,
            name="person",
            field_type=StructType(
                NestedField(field_id=16, name="name", field_type=StringType(), required=False),
                NestedField(field_id=17, name="age", field_type=IntegerType(), required=True),
            ),
            required=False,
        ),
        NestedField(
            field_id=24,
            name="points",
            field_type=ListType(
                element_id=25,
                element_type=StructType(
                    NestedField(field_id=26, name="x", field_type=LongType(), required=True),
                    NestedField(field_id=27, name="y", field_type=LongType(), required=True),
                ),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(field_id=28, name="float", field_type=FloatType(), required=True),
        NestedField(field_id=29, name="double", field_type=DoubleType(), required=True),
        schema_id=1,
        identifier_field_ids=[1],
    )


@pytest.fixture(scope="session")
def table_schema_with_all_types() -> Schema:
    return Schema(
        NestedField(field_id=1, name="boolean", field_type=BooleanType(), required=True),
        NestedField(field_id=2, name="integer", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="long", field_type=LongType(), required=True),
        NestedField(field_id=4, name="float", field_type=FloatType(), required=True),
        NestedField(field_id=5, name="double", field_type=DoubleType(), required=True),
        NestedField(field_id=6, name="decimal", field_type=DecimalType(32, 3), required=True),
        NestedField(field_id=7, name="date", field_type=DateType(), required=True),
        NestedField(field_id=8, name="time", field_type=TimeType(), required=True),
        NestedField(field_id=9, name="timestamp", field_type=TimestampType(), required=True),
        NestedField(field_id=10, name="timestamptz", field_type=TimestamptzType(), required=True),
        NestedField(field_id=11, name="string", field_type=StringType(), required=True),
        NestedField(field_id=12, name="uuid", field_type=UUIDType(), required=True),
        NestedField(field_id=14, name="fixed", field_type=FixedType(12), required=True),
        NestedField(field_id=13, name="binary", field_type=BinaryType(), required=True),
        NestedField(
            field_id=15,
            name="list",
            field_type=ListType(element_id=16, element_type=StringType(), element_required=True),
            required=True,
        ),
        NestedField(
            field_id=17,
            name="map",
            field_type=MapType(
                key_id=18,
                key_type=StringType(),
                value_id=19,
                value_type=IntegerType(),
                value_required=True,
            ),
            required=True,
        ),
        NestedField(
            field_id=20,
            name="struct",
            field_type=StructType(
                NestedField(field_id=21, name="inner_string", field_type=StringType(), required=False),
                NestedField(field_id=22, name="inner_int", field_type=IntegerType(), required=True),
            ),
        ),
        schema_id=1,
        identifier_field_ids=[2],
    )


@pytest.fixture(params=["abfs", "abfss", "wasb", "wasbs"])
def adls_scheme(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session")
def pyarrow_schema_simple_without_ids() -> "pa.Schema":
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("foo", pa.string(), nullable=True),
            pa.field("bar", pa.int32(), nullable=False),
            pa.field("baz", pa.bool_(), nullable=True),
        ]
    )


@pytest.fixture(scope="session")
def pyarrow_schema_nested_without_ids() -> "pa.Schema":
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("foo", pa.string(), nullable=False),
            pa.field("bar", pa.int32(), nullable=False),
            pa.field("baz", pa.bool_(), nullable=True),
            pa.field("qux", pa.list_(pa.string()), nullable=False),
            pa.field(
                "quux",
                pa.map_(
                    pa.string(),
                    pa.map_(pa.string(), pa.int32()),
                ),
                nullable=False,
            ),
            pa.field(
                "location",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("latitude", pa.float32(), nullable=False),
                            pa.field("longitude", pa.float32(), nullable=False),
                        ]
                    ),
                ),
                nullable=False,
            ),
            pa.field(
                "person",
                pa.struct(
                    [
                        pa.field("name", pa.string(), nullable=True),
                        pa.field("age", pa.int32(), nullable=False),
                    ]
                ),
                nullable=True,
            ),
        ]
    )


@pytest.fixture(scope="session")
def iceberg_schema_simple() -> Schema:
    return Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
    )


@pytest.fixture(scope="session")
def iceberg_schema_simple_no_ids() -> Schema:
    return Schema(
        NestedField(field_id=-1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=-1, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=-1, name="baz", field_type=BooleanType(), required=False),
    )


@pytest.fixture(scope="session")
def iceberg_table_schema_simple() -> Schema:
    return Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=False),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        schema_id=0,
        identifier_field_ids=[],
    )


@pytest.fixture(scope="session")
def iceberg_schema_nested() -> Schema:
    return Schema(
        NestedField(field_id=1, name="foo", field_type=StringType(), required=True),
        NestedField(field_id=2, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=3, name="baz", field_type=BooleanType(), required=False),
        NestedField(
            field_id=4,
            name="qux",
            field_type=ListType(element_id=5, element_type=StringType(), element_required=False),
            required=True,
        ),
        NestedField(
            field_id=6,
            name="quux",
            field_type=MapType(
                key_id=7,
                key_type=StringType(),
                value_id=8,
                value_type=MapType(key_id=9, key_type=StringType(), value_id=10, value_type=IntegerType(), value_required=False),
                value_required=False,
            ),
            required=True,
        ),
        NestedField(
            field_id=11,
            name="location",
            field_type=ListType(
                element_id=12,
                element_type=StructType(
                    NestedField(field_id=13, name="latitude", field_type=FloatType(), required=True),
                    NestedField(field_id=14, name="longitude", field_type=FloatType(), required=True),
                ),
                element_required=False,
            ),
            required=True,
        ),
        NestedField(
            field_id=15,
            name="person",
            field_type=StructType(
                NestedField(field_id=16, name="name", field_type=StringType(), required=False),
                NestedField(field_id=17, name="age", field_type=IntegerType(), required=True),
            ),
            required=False,
        ),
    )


@pytest.fixture(scope="session")
def iceberg_schema_nested_no_ids() -> Schema:
    return Schema(
        NestedField(field_id=-1, name="foo", field_type=StringType(), required=True),
        NestedField(field_id=-1, name="bar", field_type=IntegerType(), required=True),
        NestedField(field_id=-1, name="baz", field_type=BooleanType(), required=False),
        NestedField(
            field_id=-1,
            name="qux",
            field_type=ListType(element_id=-1, element_type=StringType(), element_required=False),
            required=True,
        ),
        NestedField(
            field_id=-1,
            name="quux",
            field_type=MapType(
                key_id=-1,
                key_type=StringType(),
                value_id=-1,
                value_type=MapType(key_id=-1, key_type=StringType(), value_id=-1, value_type=IntegerType(), value_required=False),
                value_required=False,
            ),
            required=True,
        ),
        NestedField(
            field_id=-1,
            name="location",
            field_type=ListType(
                element_id=-1,
                element_type=StructType(
                    NestedField(field_id=-1, name="latitude", field_type=FloatType(), required=True),
                    NestedField(field_id=-1, name="longitude", field_type=FloatType(), required=True),
                ),
                element_required=False,
            ),
            required=True,
        ),
        NestedField(
            field_id=-1,
            name="person",
            field_type=StructType(
                NestedField(field_id=-1, name="name", field_type=StringType(), required=False),
                NestedField(field_id=-1, name="age", field_type=IntegerType(), required=True),
            ),
            required=False,
        ),
    )


@pytest.fixture(scope="session")
def all_avro_types() -> Dict[str, Any]:
    return {
        "type": "record",
        "name": "all_avro_types",
        "fields": [
            {"name": "primitive_string", "type": "string", "field-id": 100},
            {"name": "primitive_int", "type": "int", "field-id": 200},
            {"name": "primitive_long", "type": "long", "field-id": 300},
            {"name": "primitive_float", "type": "float", "field-id": 400},
            {"name": "primitive_double", "type": "double", "field-id": 500},
            {"name": "primitive_bytes", "type": "bytes", "field-id": 600},
            {
                "type": "record",
                "name": "Person",
                "fields": [
                    {"name": "name", "type": "string", "field-id": 701},
                    {"name": "age", "type": "long", "field-id": 702},
                    {"name": "gender", "type": ["string", "null"], "field-id": 703},
                ],
                "field-id": 700,
            },
            {
                "name": "array_with_string",
                "type": {
                    "type": "array",
                    "items": "string",
                    "default": [],
                    "element-id": 801,
                },
                "field-id": 800,
            },
            {
                "name": "array_with_optional_string",
                "type": [
                    "null",
                    {
                        "type": "array",
                        "items": ["string", "null"],
                        "default": [],
                        "element-id": 901,
                    },
                ],
                "field-id": 900,
            },
            {
                "name": "array_with_optional_record",
                "type": [
                    "null",
                    {
                        "type": "array",
                        "items": [
                            "null",
                            {
                                "type": "record",
                                "name": "person",
                                "fields": [
                                    {"name": "name", "type": "string", "field-id": 1002},
                                    {"name": "age", "type": "long", "field-id": 1003},
                                    {"name": "gender", "type": ["string", "null"], "field-id": 1004},
                                ],
                            },
                        ],
                        "element-id": 1001,
                    },
                ],
                "field-id": 1000,
            },
            {
                "name": "map_with_longs",
                "type": {
                    "type": "map",
                    "values": "long",
                    "default": {},
                    "key-id": 1101,
                    "value-id": 1102,
                },
                "field-id": 1000,
            },
        ],
    }


EXAMPLE_TABLE_METADATA_V1 = {
    "format-version": 1,
    "table-uuid": "d20125c8-7284-442c-9aea-15fee620737c",
    "location": "s3://bucket/test/location",
    "last-updated-ms": 1602638573874,
    "last-column-id": 3,
    "schema": {
        "type": "struct",
        "fields": [
            {"id": 1, "name": "x", "required": True, "type": "long"},
            {"id": 2, "name": "y", "required": True, "type": "long", "doc": "comment"},
            {"id": 3, "name": "z", "required": True, "type": "long"},
        ],
    },
    "partition-spec": [{"name": "x", "transform": "identity", "source-id": 1, "field-id": 1000}],
    "properties": {},
    "current-snapshot-id": -1,
    "snapshots": [{"snapshot-id": 1925, "timestamp-ms": 1602638573822, "manifest-list": "s3://bucket/test/manifest-list"}],
}


@pytest.fixture(scope="session")
def example_table_metadata_v1() -> Dict[str, Any]:
    return EXAMPLE_TABLE_METADATA_V1


EXAMPLE_TABLE_METADATA_WITH_SNAPSHOT_V1 = {
    "format-version": 1,
    "table-uuid": "b55d9dda-6561-423a-8bfc-787980ce421f",
    "location": "s3://warehouse/database/table",
    "last-updated-ms": 1646787054459,
    "last-column-id": 2,
    "schema": {
        "type": "struct",
        "schema-id": 0,
        "fields": [
            {"id": 1, "name": "id", "required": False, "type": "int"},
            {"id": 2, "name": "data", "required": False, "type": "string"},
        ],
    },
    "current-schema-id": 0,
    "schemas": [
        {
            "type": "struct",
            "schema-id": 0,
            "fields": [
                {"id": 1, "name": "id", "required": False, "type": "int"},
                {"id": 2, "name": "data", "required": False, "type": "string"},
            ],
        }
    ],
    "partition-spec": [],
    "default-spec-id": 0,
    "partition-specs": [{"spec-id": 0, "fields": []}],
    "last-partition-id": 999,
    "default-sort-order-id": 0,
    "sort-orders": [{"order-id": 0, "fields": []}],
    "properties": {
        "owner": "bryan",
        "write.metadata.compression-codec": "gzip",
    },
    "current-snapshot-id": 3497810964824022504,
    "refs": {"main": {"snapshot-id": 3497810964824022504, "type": "branch"}},
    "snapshots": [
        {
            "snapshot-id": 3497810964824022504,
            "timestamp-ms": 1646787054459,
            "summary": {
                "operation": "append",
                "spark.app.id": "local-1646787004168",
                "added-data-files": "1",
                "added-records": "1",
                "added-files-size": "697",
                "changed-partition-count": "1",
                "total-records": "1",
                "total-files-size": "697",
                "total-data-files": "1",
                "total-delete-files": "0",
                "total-position-deletes": "0",
                "total-equality-deletes": "0",
            },
            "manifest-list": "s3://warehouse/database/table/metadata/snap-3497810964824022504-1-c4f68204-666b-4e50-a9df-b10c34bf6b82.avro",
            "schema-id": 0,
        }
    ],
    "snapshot-log": [{"timestamp-ms": 1646787054459, "snapshot-id": 3497810964824022504}],
    "metadata-log": [
        {
            "timestamp-ms": 1646787031514,
            "metadata-file": "s3://warehouse/database/table/metadata/00000-88484a1c-00e5-4a07-a787-c0e7aeffa805.gz.metadata.json",
        }
    ],
}


@pytest.fixture
def example_table_metadata_with_snapshot_v1() -> Dict[str, Any]:
    return EXAMPLE_TABLE_METADATA_WITH_SNAPSHOT_V1


EXAMPLE_TABLE_METADATA_NO_SNAPSHOT_V1 = {
    "format-version": 1,
    "table-uuid": "bf289591-dcc0-4234-ad4f-5c3eed811a29",
    "location": "s3://warehouse/database/table",
    "last-updated-ms": 1657810967051,
    "last-column-id": 3,
    "schema": {
        "type": "struct",
        "schema-id": 0,
        "identifier-field-ids": [2],
        "fields": [
            {"id": 1, "name": "foo", "required": False, "type": "string"},
            {"id": 2, "name": "bar", "required": True, "type": "int"},
            {"id": 3, "name": "baz", "required": False, "type": "boolean"},
        ],
    },
    "current-schema-id": 0,
    "schemas": [
        {
            "type": "struct",
            "schema-id": 0,
            "identifier-field-ids": [2],
            "fields": [
                {"id": 1, "name": "foo", "required": False, "type": "string"},
                {"id": 2, "name": "bar", "required": True, "type": "int"},
                {"id": 3, "name": "baz", "required": False, "type": "boolean"},
            ],
        }
    ],
    "partition-spec": [],
    "default-spec-id": 0,
    "last-partition-id": 999,
    "default-sort-order-id": 0,
    "sort-orders": [{"order-id": 0, "fields": []}],
    "properties": {
        "write.delete.parquet.compression-codec": "zstd",
        "write.metadata.compression-codec": "gzip",
        "write.summary.partition-limit": "100",
        "write.parquet.compression-codec": "zstd",
    },
    "current-snapshot-id": -1,
    "refs": {},
    "snapshots": [],
    "snapshot-log": [],
    "metadata-log": [],
}


@pytest.fixture
def example_table_metadata_no_snapshot_v1() -> Dict[str, Any]:
    return EXAMPLE_TABLE_METADATA_NO_SNAPSHOT_V1


@pytest.fixture
def example_table_metadata_v2_with_extensive_snapshots() -> Dict[str, Any]:
    def generate_snapshot(
        snapshot_id: int,
        parent_snapshot_id: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
        sequence_number: int = 0,
    ) -> Dict[str, Any]:
        return {
            "snapshot-id": snapshot_id,
            "parent-snapshot-id": parent_snapshot_id,
            "timestamp-ms": timestamp_ms or int(time.time() * 1000),
            "sequence-number": sequence_number,
            "summary": {"operation": "append"},
            "manifest-list": f"s3://a/b/{snapshot_id}.avro",
        }

    snapshots = []
    snapshot_log = []
    initial_snapshot_id = 3051729675574597004

    for i in range(2000):
        snapshot_id = initial_snapshot_id + i
        parent_snapshot_id = snapshot_id - 1 if i > 0 else None
        timestamp_ms = int(time.time() * 1000) - randint(0, 1000000)
        snapshots.append(generate_snapshot(snapshot_id, parent_snapshot_id, timestamp_ms, i))
        snapshot_log.append({"snapshot-id": snapshot_id, "timestamp-ms": timestamp_ms})

    return {
        "format-version": 2,
        "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
        "location": "s3://bucket/test/location",
        "last-sequence-number": 34,
        "last-updated-ms": 1602638573590,
        "last-column-id": 3,
        "current-schema-id": 1,
        "schemas": [
            {"type": "struct", "schema-id": 0, "fields": [{"id": 1, "name": "x", "required": True, "type": "long"}]},
            {
                "type": "struct",
                "schema-id": 1,
                "identifier-field-ids": [1, 2],
                "fields": [
                    {"id": 1, "name": "x", "required": True, "type": "long"},
                    {"id": 2, "name": "y", "required": True, "type": "long", "doc": "comment"},
                    {"id": 3, "name": "z", "required": True, "type": "long"},
                ],
            },
        ],
        "default-spec-id": 0,
        "partition-specs": [{"spec-id": 0, "fields": [{"name": "x", "transform": "identity", "source-id": 1, "field-id": 1000}]}],
        "last-partition-id": 1000,
        "default-sort-order-id": 3,
        "sort-orders": [
            {
                "order-id": 3,
                "fields": [
                    {"transform": "identity", "source-id": 2, "direction": "asc", "null-order": "nulls-first"},
                    {"transform": "bucket[4]", "source-id": 3, "direction": "desc", "null-order": "nulls-last"},
                ],
            }
        ],
        "properties": {"read.split.target.size": "134217728"},
        "current-snapshot-id": initial_snapshot_id + 1999,
        "snapshots": snapshots,
        "snapshot-log": snapshot_log,
        "metadata-log": [{"metadata-file": "s3://bucket/.../v1.json", "timestamp-ms": 1515100}],
        "refs": {"test": {"snapshot-id": initial_snapshot_id, "type": "tag", "max-ref-age-ms": 10000000}},
    }


EXAMPLE_TABLE_METADATA_V2 = {
    "format-version": 2,
    "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
    "location": "s3://bucket/test/location",
    "last-sequence-number": 34,
    "last-updated-ms": 1602638573590,
    "last-column-id": 3,
    "current-schema-id": 1,
    "schemas": [
        {"type": "struct", "schema-id": 0, "fields": [{"id": 1, "name": "x", "required": True, "type": "long"}]},
        {
            "type": "struct",
            "schema-id": 1,
            "identifier-field-ids": [1, 2],
            "fields": [
                {"id": 1, "name": "x", "required": True, "type": "long"},
                {"id": 2, "name": "y", "required": True, "type": "long", "doc": "comment"},
                {"id": 3, "name": "z", "required": True, "type": "long"},
            ],
        },
    ],
    "default-spec-id": 0,
    "partition-specs": [{"spec-id": 0, "fields": [{"name": "x", "transform": "identity", "source-id": 1, "field-id": 1000}]}],
    "last-partition-id": 1000,
    "default-sort-order-id": 3,
    "sort-orders": [
        {
            "order-id": 3,
            "fields": [
                {"transform": "identity", "source-id": 2, "direction": "asc", "null-order": "nulls-first"},
                {"transform": "bucket[4]", "source-id": 3, "direction": "desc", "null-order": "nulls-last"},
            ],
        }
    ],
    "properties": {"read.split.target.size": "134217728"},
    "current-snapshot-id": 3055729675574597004,
    "snapshots": [
        {
            "snapshot-id": 3051729675574597004,
            "timestamp-ms": 1515100955770,
            "sequence-number": 0,
            "summary": {"operation": "append"},
            "manifest-list": "s3://a/b/1.avro",
        },
        {
            "snapshot-id": 3055729675574597004,
            "parent-snapshot-id": 3051729675574597004,
            "timestamp-ms": 1555100955770,
            "sequence-number": 1,
            "summary": {"operation": "append"},
            "manifest-list": "s3://a/b/2.avro",
            "schema-id": 1,
        },
    ],
    "snapshot-log": [
        {"snapshot-id": 3051729675574597004, "timestamp-ms": 1515100955770},
        {"snapshot-id": 3055729675574597004, "timestamp-ms": 1555100955770},
    ],
    "metadata-log": [{"metadata-file": "s3://bucket/.../v1.json", "timestamp-ms": 1515100}],
    "refs": {"test": {"snapshot-id": 3051729675574597004, "type": "tag", "max-ref-age-ms": 10000000}},
}

EXAMPLE_TABLE_METADATA_V3 = {
    "format-version": 3,
    "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
    "location": "s3://bucket/test/location",
    "last-sequence-number": 34,
    "last-updated-ms": 1602638573590,
    "last-column-id": 3,
    "current-schema-id": 1,
    "schemas": [
        {"type": "struct", "schema-id": 0, "fields": [{"id": 1, "name": "x", "required": True, "type": "long"}]},
        {
            "type": "struct",
            "schema-id": 1,
            "identifier-field-ids": [1, 2],
            "fields": [
                {"id": 1, "name": "x", "required": True, "type": "long"},
                {"id": 2, "name": "y", "required": True, "type": "long", "doc": "comment"},
                {"id": 3, "name": "z", "required": True, "type": "long"},
                {"id": 4, "name": "u", "required": True, "type": "unknown"},
                {"id": 5, "name": "ns", "required": True, "type": "timestamp_ns"},
                {"id": 6, "name": "nstz", "required": True, "type": "timestamptz_ns"},
            ],
        },
    ],
    "default-spec-id": 0,
    "partition-specs": [{"spec-id": 0, "fields": [{"name": "x", "transform": "identity", "source-ids": [1], "field-id": 1000}]}],
    "last-partition-id": 1000,
    "default-sort-order-id": 3,
    "sort-orders": [
        {
            "order-id": 3,
            "fields": [
                {"transform": "identity", "source-ids": [2], "direction": "asc", "null-order": "nulls-first"},
                {"transform": "bucket[4]", "source-ids": [3], "direction": "desc", "null-order": "nulls-last"},
            ],
        }
    ],
    "properties": {"read.split.target.size": "134217728"},
    "current-snapshot-id": 3055729675574597004,
    "snapshots": [
        {
            "snapshot-id": 3051729675574597004,
            "timestamp-ms": 1515100955770,
            "sequence-number": 0,
            "summary": {"operation": "append"},
            "manifest-list": "s3://a/b/1.avro",
        },
        {
            "snapshot-id": 3055729675574597004,
            "parent-snapshot-id": 3051729675574597004,
            "timestamp-ms": 1555100955770,
            "sequence-number": 1,
            "summary": {"operation": "append"},
            "manifest-list": "s3://a/b/2.avro",
            "schema-id": 1,
        },
    ],
    "snapshot-log": [
        {"snapshot-id": 3051729675574597004, "timestamp-ms": 1515100955770},
        {"snapshot-id": 3055729675574597004, "timestamp-ms": 1555100955770},
    ],
    "metadata-log": [{"metadata-file": "s3://bucket/.../v1.json", "timestamp-ms": 1515100}],
    "refs": {"test": {"snapshot-id": 3051729675574597004, "type": "tag", "max-ref-age-ms": 10000000}},
}

TABLE_METADATA_V2_WITH_FIXED_AND_DECIMAL_TYPES = {
    "format-version": 2,
    "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
    "location": "s3://bucket/test/location",
    "last-sequence-number": 34,
    "last-updated-ms": 1602638573590,
    "last-column-id": 7,
    "current-schema-id": 1,
    "schemas": [
        {
            "type": "struct",
            "schema-id": 1,
            "identifier-field-ids": [1],
            "fields": [
                {"id": 1, "name": "x", "required": True, "type": "long"},
                {"id": 4, "name": "a", "required": True, "type": "decimal(16, 2)"},
                {"id": 5, "name": "b", "required": True, "type": "decimal(16, 8)"},
                {"id": 6, "name": "c", "required": True, "type": "fixed[16]"},
                {"id": 7, "name": "d", "required": True, "type": "fixed[18]"},
            ],
        }
    ],
    "default-spec-id": 0,
    "partition-specs": [{"spec-id": 0, "fields": [{"name": "x", "transform": "identity", "source-id": 1, "field-id": 1000}]}],
    "last-partition-id": 1000,
    "properties": {"read.split.target.size": "134217728"},
    "current-snapshot-id": 3055729675574597004,
    "snapshots": [
        {
            "snapshot-id": 3051729675574597004,
            "timestamp-ms": 1515100955770,
            "sequence-number": 0,
            "summary": {"operation": "append"},
            "manifest-list": "s3://a/b/1.avro",
        },
        {
            "snapshot-id": 3055729675574597004,
            "parent-snapshot-id": 3051729675574597004,
            "timestamp-ms": 1555100955770,
            "sequence-number": 1,
            "summary": {"operation": "append"},
            "manifest-list": "s3://a/b/2.avro",
            "schema-id": 1,
        },
    ],
    "snapshot-log": [
        {"snapshot-id": 3051729675574597004, "timestamp-ms": 1515100955770},
        {"snapshot-id": 3055729675574597004, "timestamp-ms": 1555100955770},
    ],
    "metadata-log": [{"metadata-file": "s3://bucket/.../v1.json", "timestamp-ms": 1515100}],
    "refs": {"test": {"snapshot-id": 3051729675574597004, "type": "tag", "max-ref-age-ms": 10000000}},
}

TABLE_METADATA_V2_WITH_STATISTICS = {
    "format-version": 2,
    "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
    "location": "s3://bucket/test/location",
    "last-sequence-number": 34,
    "last-updated-ms": 1602638573590,
    "last-column-id": 3,
    "current-schema-id": 0,
    "schemas": [
        {
            "type": "struct",
            "schema-id": 0,
            "fields": [
                {
                    "id": 1,
                    "name": "x",
                    "required": True,
                    "type": "long",
                }
            ],
        }
    ],
    "default-spec-id": 0,
    "partition-specs": [{"spec-id": 0, "fields": []}],
    "last-partition-id": 1000,
    "default-sort-order-id": 0,
    "sort-orders": [{"order-id": 0, "fields": []}],
    "properties": {},
    "current-snapshot-id": 3055729675574597004,
    "snapshots": [
        {
            "snapshot-id": 3051729675574597004,
            "timestamp-ms": 1515100955770,
            "sequence-number": 0,
            "summary": {"operation": "append"},
            "manifest-list": "s3://a/b/1.avro",
        },
        {
            "snapshot-id": 3055729675574597004,
            "parent-snapshot-id": 3051729675574597004,
            "timestamp-ms": 1555100955770,
            "sequence-number": 1,
            "summary": {"operation": "append"},
            "manifest-list": "s3://a/b/2.avro",
            "schema-id": 1,
        },
    ],
    "statistics": [
        {
            "snapshot-id": 3051729675574597004,
            "statistics-path": "s3://a/b/stats.puffin",
            "file-size-in-bytes": 413,
            "file-footer-size-in-bytes": 42,
            "blob-metadata": [
                {
                    "type": "apache-datasketches-theta-v1",
                    "snapshot-id": 3051729675574597004,
                    "sequence-number": 1,
                    "fields": [1],
                }
            ],
        },
        {
            "snapshot-id": 3055729675574597004,
            "statistics-path": "s3://a/b/stats.puffin",
            "file-size-in-bytes": 413,
            "file-footer-size-in-bytes": 42,
            "blob-metadata": [
                {
                    "type": "deletion-vector-v1",
                    "snapshot-id": 3055729675574597004,
                    "sequence-number": 1,
                    "fields": [1],
                }
            ],
        },
    ],
    "snapshot-log": [],
    "metadata-log": [],
}


@pytest.fixture
def example_table_metadata_v2() -> Dict[str, Any]:
    return EXAMPLE_TABLE_METADATA_V2


@pytest.fixture
def table_metadata_v2_with_fixed_and_decimal_types() -> Dict[str, Any]:
    return TABLE_METADATA_V2_WITH_FIXED_AND_DECIMAL_TYPES


@pytest.fixture
def table_metadata_v2_with_statistics() -> Dict[str, Any]:
    return TABLE_METADATA_V2_WITH_STATISTICS


@pytest.fixture
def example_table_metadata_v3() -> Dict[str, Any]:
    return EXAMPLE_TABLE_METADATA_V3


@pytest.fixture(scope="session")
def table_location(tmp_path_factory: pytest.TempPathFactory) -> str:
    from pyiceberg.io.pyarrow import PyArrowFileIO

    metadata_filename = f"{uuid.uuid4()}.metadata.json"
    metadata_location = str(tmp_path_factory.getbasetemp() / "metadata" / metadata_filename)
    version_hint_location = str(tmp_path_factory.getbasetemp() / "metadata" / "version-hint.text")
    metadata = TableMetadataV2(**EXAMPLE_TABLE_METADATA_V2)
    ToOutputFile.table_metadata(metadata, PyArrowFileIO().new_output(location=metadata_location), overwrite=True)

    with PyArrowFileIO().new_output(location=version_hint_location).create(overwrite=True) as s:
        s.write(metadata_filename.encode("utf-8"))

    return str(tmp_path_factory.getbasetemp())


@pytest.fixture(scope="session")
def metadata_location(tmp_path_factory: pytest.TempPathFactory) -> str:
    from pyiceberg.io.pyarrow import PyArrowFileIO

    metadata_location = str(tmp_path_factory.mktemp("metadata") / f"{uuid.uuid4()}.metadata.json")
    metadata = TableMetadataV2(**EXAMPLE_TABLE_METADATA_V2)
    ToOutputFile.table_metadata(metadata, PyArrowFileIO().new_output(location=metadata_location), overwrite=True)
    return metadata_location


@pytest.fixture(scope="session")
def metadata_location_gz(tmp_path_factory: pytest.TempPathFactory) -> str:
    from pyiceberg.io.pyarrow import PyArrowFileIO

    metadata_location = str(tmp_path_factory.mktemp("metadata") / f"{uuid.uuid4()}.gz.metadata.json")
    metadata = TableMetadataV2(**EXAMPLE_TABLE_METADATA_V2)
    ToOutputFile.table_metadata(metadata, PyArrowFileIO().new_output(location=metadata_location), overwrite=True)
    return metadata_location


manifest_entry_records = [
    {
        "status": 1,
        "snapshot_id": 8744736658442914487,
        "data_file": {
            "file_path": "/home/iceberg/warehouse/nyc/taxis_partitioned/data/VendorID=null/00000-633-d8a4223e-dc97-45a1-86e1-adaba6e8abd7-00001.parquet",
            "file_format": "PARQUET",
            "partition": {"VendorID": 1, "tpep_pickup_datetime": 1925},
            "record_count": 19513,
            "file_size_in_bytes": 388872,
            "block_size_in_bytes": 67108864,
            "column_sizes": [
                {"key": 1, "value": 53},
                {"key": 2, "value": 98153},
                {"key": 3, "value": 98693},
                {"key": 4, "value": 53},
                {"key": 5, "value": 53},
                {"key": 6, "value": 53},
                {"key": 7, "value": 17425},
                {"key": 8, "value": 18528},
                {"key": 9, "value": 53},
                {"key": 10, "value": 44788},
                {"key": 11, "value": 35571},
                {"key": 12, "value": 53},
                {"key": 13, "value": 1243},
                {"key": 14, "value": 2355},
                {"key": 15, "value": 12750},
                {"key": 16, "value": 4029},
                {"key": 17, "value": 110},
                {"key": 18, "value": 47194},
                {"key": 19, "value": 2948},
            ],
            "value_counts": [
                {"key": 1, "value": 19513},
                {"key": 2, "value": 19513},
                {"key": 3, "value": 19513},
                {"key": 4, "value": 19513},
                {"key": 5, "value": 19513},
                {"key": 6, "value": 19513},
                {"key": 7, "value": 19513},
                {"key": 8, "value": 19513},
                {"key": 9, "value": 19513},
                {"key": 10, "value": 19513},
                {"key": 11, "value": 19513},
                {"key": 12, "value": 19513},
                {"key": 13, "value": 19513},
                {"key": 14, "value": 19513},
                {"key": 15, "value": 19513},
                {"key": 16, "value": 19513},
                {"key": 17, "value": 19513},
                {"key": 18, "value": 19513},
                {"key": 19, "value": 19513},
            ],
            "null_value_counts": [
                {"key": 1, "value": 19513},
                {"key": 2, "value": 0},
                {"key": 3, "value": 0},
                {"key": 4, "value": 19513},
                {"key": 5, "value": 19513},
                {"key": 6, "value": 19513},
                {"key": 7, "value": 0},
                {"key": 8, "value": 0},
                {"key": 9, "value": 19513},
                {"key": 10, "value": 0},
                {"key": 11, "value": 0},
                {"key": 12, "value": 19513},
                {"key": 13, "value": 0},
                {"key": 14, "value": 0},
                {"key": 15, "value": 0},
                {"key": 16, "value": 0},
                {"key": 17, "value": 0},
                {"key": 18, "value": 0},
                {"key": 19, "value": 0},
            ],
            "nan_value_counts": [
                {"key": 16, "value": 0},
                {"key": 17, "value": 0},
                {"key": 18, "value": 0},
                {"key": 19, "value": 0},
                {"key": 10, "value": 0},
                {"key": 11, "value": 0},
                {"key": 12, "value": 0},
                {"key": 13, "value": 0},
                {"key": 14, "value": 0},
                {"key": 15, "value": 0},
            ],
            "lower_bounds": [
                {"key": 2, "value": b"2020-04-01 00:00"},
                {"key": 3, "value": b"2020-04-01 00:12"},
                {"key": 7, "value": b"\x03\x00\x00\x00"},
                {"key": 8, "value": b"\x01\x00\x00\x00"},
                {"key": 10, "value": b"\xf6(\\\x8f\xc2\x05S\xc0"},
                {"key": 11, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 13, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 14, "value": b"\x00\x00\x00\x00\x00\x00\xe0\xbf"},
                {"key": 15, "value": b")\\\x8f\xc2\xf5(\x08\xc0"},
                {"key": 16, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 17, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 18, "value": b"\xf6(\\\x8f\xc2\xc5S\xc0"},
                {"key": 19, "value": b"\x00\x00\x00\x00\x00\x00\x04\xc0"},
            ],
            "upper_bounds": [
                {"key": 2, "value": b"2020-04-30 23:5:"},
                {"key": 3, "value": b"2020-05-01 00:41"},
                {"key": 7, "value": b"\t\x01\x00\x00"},
                {"key": 8, "value": b"\t\x01\x00\x00"},
                {"key": 10, "value": b"\xcd\xcc\xcc\xcc\xcc,_@"},
                {"key": 11, "value": b"\x1f\x85\xebQ\\\xe2\xfe@"},
                {"key": 13, "value": b"\x00\x00\x00\x00\x00\x00\x12@"},
                {"key": 14, "value": b"\x00\x00\x00\x00\x00\x00\xe0?"},
                {"key": 15, "value": b"q=\n\xd7\xa3\xf01@"},
                {"key": 16, "value": b"\x00\x00\x00\x00\x00`B@"},
                {"key": 17, "value": b"333333\xd3?"},
                {"key": 18, "value": b"\x00\x00\x00\x00\x00\x18b@"},
                {"key": 19, "value": b"\x00\x00\x00\x00\x00\x00\x04@"},
            ],
            "key_metadata": None,
            "split_offsets": [4],
            "sort_order_id": 0,
        },
    },
    {
        "status": 1,
        "snapshot_id": 8744736658442914487,
        "data_file": {
            "file_path": "/home/iceberg/warehouse/nyc/taxis_partitioned/data/VendorID=1/00000-633-d8a4223e-dc97-45a1-86e1-adaba6e8abd7-00002.parquet",
            "file_format": "PARQUET",
            "partition": {"VendorID": 1, "tpep_pickup_datetime": None},
            "record_count": 95050,
            "file_size_in_bytes": 1265950,
            "block_size_in_bytes": 67108864,
            "column_sizes": [
                {"key": 1, "value": 318},
                {"key": 2, "value": 329806},
                {"key": 3, "value": 331632},
                {"key": 4, "value": 15343},
                {"key": 5, "value": 2351},
                {"key": 6, "value": 3389},
                {"key": 7, "value": 71269},
                {"key": 8, "value": 76429},
                {"key": 9, "value": 16383},
                {"key": 10, "value": 86992},
                {"key": 11, "value": 89608},
                {"key": 12, "value": 265},
                {"key": 13, "value": 19377},
                {"key": 14, "value": 1692},
                {"key": 15, "value": 76162},
                {"key": 16, "value": 4354},
                {"key": 17, "value": 759},
                {"key": 18, "value": 120650},
                {"key": 19, "value": 11804},
            ],
            "value_counts": [
                {"key": 1, "value": 95050},
                {"key": 2, "value": 95050},
                {"key": 3, "value": 95050},
                {"key": 4, "value": 95050},
                {"key": 5, "value": 95050},
                {"key": 6, "value": 95050},
                {"key": 7, "value": 95050},
                {"key": 8, "value": 95050},
                {"key": 9, "value": 95050},
                {"key": 10, "value": 95050},
                {"key": 11, "value": 95050},
                {"key": 12, "value": 95050},
                {"key": 13, "value": 95050},
                {"key": 14, "value": 95050},
                {"key": 15, "value": 95050},
                {"key": 16, "value": 95050},
                {"key": 17, "value": 95050},
                {"key": 18, "value": 95050},
                {"key": 19, "value": 95050},
            ],
            "null_value_counts": [
                {"key": 1, "value": 0},
                {"key": 2, "value": 0},
                {"key": 3, "value": 0},
                {"key": 4, "value": 0},
                {"key": 5, "value": 0},
                {"key": 6, "value": 0},
                {"key": 7, "value": 0},
                {"key": 8, "value": 0},
                {"key": 9, "value": 0},
                {"key": 10, "value": 0},
                {"key": 11, "value": 0},
                {"key": 12, "value": 95050},
                {"key": 13, "value": 0},
                {"key": 14, "value": 0},
                {"key": 15, "value": 0},
                {"key": 16, "value": 0},
                {"key": 17, "value": 0},
                {"key": 18, "value": 0},
                {"key": 19, "value": 0},
            ],
            "nan_value_counts": [
                {"key": 16, "value": 0},
                {"key": 17, "value": 0},
                {"key": 18, "value": 0},
                {"key": 19, "value": 0},
                {"key": 10, "value": 0},
                {"key": 11, "value": 0},
                {"key": 12, "value": 0},
                {"key": 13, "value": 0},
                {"key": 14, "value": 0},
                {"key": 15, "value": 0},
            ],
            "lower_bounds": [
                {"key": 1, "value": b"\x01\x00\x00\x00"},
                {"key": 2, "value": b"2020-04-01 00:00"},
                {"key": 3, "value": b"2020-04-01 00:03"},
                {"key": 4, "value": b"\x00\x00\x00\x00"},
                {"key": 5, "value": b"\x01\x00\x00\x00"},
                {"key": 6, "value": b"N"},
                {"key": 7, "value": b"\x01\x00\x00\x00"},
                {"key": 8, "value": b"\x01\x00\x00\x00"},
                {"key": 9, "value": b"\x01\x00\x00\x00"},
                {"key": 10, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 11, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 13, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 14, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 15, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 16, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 17, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 18, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
                {"key": 19, "value": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
            ],
            "upper_bounds": [
                {"key": 1, "value": b"\x01\x00\x00\x00"},
                {"key": 2, "value": b"2020-04-30 23:5:"},
                {"key": 3, "value": b"2020-05-01 00:1:"},
                {"key": 4, "value": b"\x06\x00\x00\x00"},
                {"key": 5, "value": b"c\x00\x00\x00"},
                {"key": 6, "value": b"Y"},
                {"key": 7, "value": b"\t\x01\x00\x00"},
                {"key": 8, "value": b"\t\x01\x00\x00"},
                {"key": 9, "value": b"\x04\x00\x00\x00"},
                {"key": 10, "value": b"\\\x8f\xc2\xf5(8\x8c@"},
                {"key": 11, "value": b"\xcd\xcc\xcc\xcc\xcc,f@"},
                {"key": 13, "value": b"\x00\x00\x00\x00\x00\x00\x1c@"},
                {"key": 14, "value": b"\x9a\x99\x99\x99\x99\x99\xf1?"},
                {"key": 15, "value": b"\x00\x00\x00\x00\x00\x00Y@"},
                {"key": 16, "value": b"\x00\x00\x00\x00\x00\xb0X@"},
                {"key": 17, "value": b"333333\xd3?"},
                {"key": 18, "value": b"\xc3\xf5(\\\x8f:\x8c@"},
                {"key": 19, "value": b"\x00\x00\x00\x00\x00\x00\x04@"},
            ],
            "key_metadata": None,
            "split_offsets": [4],
            "sort_order_id": 0,
        },
    },
]

manifest_file_records_v1 = [
    {
        "manifest_path": "/home/iceberg/warehouse/nyc/taxis_partitioned/metadata/0125c686-8aa6-4502-bdcc-b6d17ca41a3b-m0.avro",
        "manifest_length": 7989,
        "partition_spec_id": 0,
        "added_snapshot_id": 9182715666859759686,
        "added_data_files_count": 3,
        "existing_data_files_count": 0,
        "deleted_data_files_count": 0,
        "partitions": [
            {"contains_null": True, "contains_nan": False, "lower_bound": b"\x01\x00\x00\x00", "upper_bound": b"\x02\x00\x00\x00"}
        ],
        "added_rows_count": 237993,
        "existing_rows_count": 0,
        "deleted_rows_count": 0,
    }
]

manifest_file_records_v2 = [
    {
        "manifest_path": "/home/iceberg/warehouse/nyc/taxis_partitioned/metadata/0125c686-8aa6-4502-bdcc-b6d17ca41a3b-m0.avro",
        "manifest_length": 7989,
        "partition_spec_id": 0,
        "content": 1,
        "sequence_number": 3,
        "min_sequence_number": 3,
        "added_snapshot_id": 9182715666859759686,
        "added_files_count": 3,
        "existing_files_count": 0,
        "deleted_files_count": 0,
        "added_rows_count": 237993,
        "existing_rows_count": 0,
        "deleted_rows_count": 0,
        "partitions": [
            {"contains_null": True, "contains_nan": False, "lower_bound": b"\x01\x00\x00\x00", "upper_bound": b"\x02\x00\x00\x00"}
        ],
        "key_metadata": b"\x19\x25",
    }
]


@pytest.fixture(scope="session")
def avro_schema_manifest_file_v1() -> Dict[str, Any]:
    return {
        "type": "record",
        "name": "manifest_file",
        "fields": [
            {"name": "manifest_path", "type": "string", "doc": "Location URI with FS scheme", "field-id": 500},
            {"name": "manifest_length", "type": "long", "doc": "Total file size in bytes", "field-id": 501},
            {"name": "partition_spec_id", "type": "int", "doc": "Spec ID used to write", "field-id": 502},
            {
                "name": "added_snapshot_id",
                "type": ["null", "long"],
                "doc": "Snapshot ID that added the manifest",
                "default": None,
                "field-id": 503,
            },
            {
                "name": "added_data_files_count",
                "type": ["null", "int"],
                "doc": "Added entry count",
                "default": None,
                "field-id": 504,
            },
            {
                "name": "existing_data_files_count",
                "type": ["null", "int"],
                "doc": "Existing entry count",
                "default": None,
                "field-id": 505,
            },
            {
                "name": "deleted_data_files_count",
                "type": ["null", "int"],
                "doc": "Deleted entry count",
                "default": None,
                "field-id": 506,
            },
            {
                "name": "partitions",
                "type": [
                    "null",
                    {
                        "type": "array",
                        "items": {
                            "type": "record",
                            "name": "r508",
                            "fields": [
                                {
                                    "name": "contains_null",
                                    "type": "boolean",
                                    "doc": "True if any file has a null partition value",
                                    "field-id": 509,
                                },
                                {
                                    "name": "contains_nan",
                                    "type": ["null", "boolean"],
                                    "doc": "True if any file has a nan partition value",
                                    "default": None,
                                    "field-id": 518,
                                },
                                {
                                    "name": "lower_bound",
                                    "type": ["null", "bytes"],
                                    "doc": "Partition lower bound for all files",
                                    "default": None,
                                    "field-id": 510,
                                },
                                {
                                    "name": "upper_bound",
                                    "type": ["null", "bytes"],
                                    "doc": "Partition upper bound for all files",
                                    "default": None,
                                    "field-id": 511,
                                },
                            ],
                        },
                        "element-id": 508,
                    },
                ],
                "doc": "Summary for each partition",
                "default": None,
                "field-id": 507,
            },
            {"name": "added_rows_count", "type": ["null", "long"], "doc": "Added rows count", "default": None, "field-id": 512},
            {
                "name": "existing_rows_count",
                "type": ["null", "long"],
                "doc": "Existing rows count",
                "default": None,
                "field-id": 513,
            },
            {
                "name": "deleted_rows_count",
                "type": ["null", "long"],
                "doc": "Deleted rows count",
                "default": None,
                "field-id": 514,
            },
        ],
    }


@pytest.fixture(scope="session")
def avro_schema_manifest_file_v2() -> Dict[str, Any]:
    return {
        "type": "record",
        "name": "manifest_file",
        "fields": [
            {"name": "manifest_path", "type": "string", "doc": "Location URI with FS scheme", "field-id": 500},
            {"name": "manifest_length", "type": "long", "doc": "Total file size in bytes", "field-id": 501},
            {"name": "partition_spec_id", "type": "int", "doc": "Spec ID used to write", "field-id": 502},
            {"name": "content", "type": "int", "doc": "Contents of the manifest: 0=data, 1=deletes", "field-id": 517},
            {
                "name": "sequence_number",
                "type": ["null", "long"],
                "doc": "Sequence number when the manifest was added",
                "field-id": 515,
            },
            {
                "name": "min_sequence_number",
                "type": ["null", "long"],
                "doc": "Lowest sequence number in the manifest",
                "field-id": 516,
            },
            {"name": "added_snapshot_id", "type": "long", "doc": "Snapshot ID that added the manifest", "field-id": 503},
            {"name": "added_files_count", "type": "int", "doc": "Added entry count", "field-id": 504},
            {"name": "existing_files_count", "type": "int", "doc": "Existing entry count", "field-id": 505},
            {"name": "deleted_files_count", "type": "int", "doc": "Deleted entry count", "field-id": 506},
            {"name": "added_rows_count", "type": "long", "doc": "Added rows count", "field-id": 512},
            {"name": "existing_rows_count", "type": "long", "doc": "Existing rows count", "field-id": 513},
            {"name": "deleted_rows_count", "type": "long", "doc": "Deleted rows count", "field-id": 514},
            {
                "name": "partitions",
                "type": [
                    "null",
                    {
                        "type": "array",
                        "items": {
                            "type": "record",
                            "name": "r508",
                            "fields": [
                                {
                                    "name": "contains_null",
                                    "type": "boolean",
                                    "doc": "True if any file has a null partition value",
                                    "field-id": 509,
                                },
                                {
                                    "name": "contains_nan",
                                    "type": ["null", "boolean"],
                                    "doc": "True if any file has a nan partition value",
                                    "default": None,
                                    "field-id": 518,
                                },
                                {
                                    "name": "lower_bound",
                                    "type": ["null", "bytes"],
                                    "doc": "Partition lower bound for all files",
                                    "default": None,
                                    "field-id": 510,
                                },
                                {
                                    "name": "upper_bound",
                                    "type": ["null", "bytes"],
                                    "doc": "Partition upper bound for all files",
                                    "default": None,
                                    "field-id": 511,
                                },
                            ],
                        },
                        "element-id": 508,
                    },
                ],
                "doc": "Summary for each partition",
                "default": None,
                "field-id": 507,
            },
        ],
    }


@pytest.fixture(scope="session")
def avro_schema_manifest_entry() -> Dict[str, Any]:
    return {
        "type": "record",
        "name": "manifest_entry",
        "fields": [
            {"name": "status", "type": "int", "field-id": 0},
            {"name": "snapshot_id", "type": ["null", "long"], "default": None, "field-id": 1},
            {
                "name": "data_file",
                "type": {
                    "type": "record",
                    "name": "r2",
                    "fields": [
                        {"name": "file_path", "type": "string", "doc": "Location URI with FS scheme", "field-id": 100},
                        {
                            "name": "file_format",
                            "type": "string",
                            "doc": "File format name: avro, orc, or parquet",
                            "field-id": 101,
                        },
                        {
                            "name": "partition",
                            "type": {
                                "type": "record",
                                "name": "r102",
                                "fields": [
                                    {"field-id": 1000, "default": None, "name": "VendorID", "type": ["null", "int"]},
                                    {
                                        "field-id": 1001,
                                        "default": None,
                                        "name": "tpep_pickup_datetime",
                                        "type": ["null", {"type": "int", "logicalType": "date"}],
                                    },
                                ],
                            },
                            "field-id": 102,
                        },
                        {"name": "record_count", "type": "long", "doc": "Number of records in the file", "field-id": 103},
                        {"name": "file_size_in_bytes", "type": "long", "doc": "Total file size in bytes", "field-id": 104},
                        {"name": "block_size_in_bytes", "type": "long", "field-id": 105},
                        {
                            "name": "column_sizes",
                            "type": [
                                "null",
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k117_v118",
                                        "fields": [
                                            {"name": "key", "type": "int", "field-id": 117},
                                            {"name": "value", "type": "long", "field-id": 118},
                                        ],
                                    },
                                    "logicalType": "map",
                                },
                            ],
                            "doc": "Map of column id to total size on disk",
                            "default": None,
                            "field-id": 108,
                        },
                        {
                            "name": "value_counts",
                            "type": [
                                "null",
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k119_v120",
                                        "fields": [
                                            {"name": "key", "type": "int", "field-id": 119},
                                            {"name": "value", "type": "long", "field-id": 120},
                                        ],
                                    },
                                    "logicalType": "map",
                                },
                            ],
                            "doc": "Map of column id to total count, including null and NaN",
                            "default": None,
                            "field-id": 109,
                        },
                        {
                            "name": "null_value_counts",
                            "type": [
                                "null",
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k121_v122",
                                        "fields": [
                                            {"name": "key", "type": "int", "field-id": 121},
                                            {"name": "value", "type": "long", "field-id": 122},
                                        ],
                                    },
                                    "logicalType": "map",
                                },
                            ],
                            "doc": "Map of column id to null value count",
                            "default": None,
                            "field-id": 110,
                        },
                        {
                            "name": "nan_value_counts",
                            "type": [
                                "null",
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k138_v139",
                                        "fields": [
                                            {"name": "key", "type": "int", "field-id": 138},
                                            {"name": "value", "type": "long", "field-id": 139},
                                        ],
                                    },
                                    "logicalType": "map",
                                },
                            ],
                            "doc": "Map of column id to number of NaN values in the column",
                            "default": None,
                            "field-id": 137,
                        },
                        {
                            "name": "lower_bounds",
                            "type": [
                                "null",
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k126_v127",
                                        "fields": [
                                            {"name": "key", "type": "int", "field-id": 126},
                                            {"name": "value", "type": "bytes", "field-id": 127},
                                        ],
                                    },
                                    "logicalType": "map",
                                },
                            ],
                            "doc": "Map of column id to lower bound",
                            "default": None,
                            "field-id": 125,
                        },
                        {
                            "name": "upper_bounds",
                            "type": [
                                "null",
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k129_v130",
                                        "fields": [
                                            {"name": "key", "type": "int", "field-id": 129},
                                            {"name": "value", "type": "bytes", "field-id": 130},
                                        ],
                                    },
                                    "logicalType": "map",
                                },
                            ],
                            "doc": "Map of column id to upper bound",
                            "default": None,
                            "field-id": 128,
                        },
                        {
                            "name": "key_metadata",
                            "type": ["null", "bytes"],
                            "doc": "Encryption key metadata blob",
                            "default": None,
                            "field-id": 131,
                        },
                        {
                            "name": "split_offsets",
                            "type": ["null", {"type": "array", "items": "long", "element-id": 133}],
                            "doc": "Splittable offsets",
                            "default": None,
                            "field-id": 132,
                        },
                        {
                            "name": "sort_order_id",
                            "type": ["null", "int"],
                            "doc": "Sort order ID",
                            "default": None,
                            "field-id": 140,
                        },
                    ],
                },
                "field-id": 2,
            },
        ],
    }


@pytest.fixture(scope="session")
def simple_struct() -> StructType:
    return StructType(
        NestedField(id=1, name="required_field", field_type=StringType(), required=True, doc="this is a doc"),
        NestedField(id=2, name="optional_field", field_type=IntegerType()),
    )


@pytest.fixture(scope="session")
def simple_list() -> ListType:
    return ListType(element_id=22, element=StringType(), element_required=True)


@pytest.fixture(scope="session")
def simple_map() -> MapType:
    return MapType(key_id=19, key_type=StringType(), value_id=25, value_type=DoubleType(), value_required=False)


@pytest.fixture(scope="session")
def generated_manifest_entry_file(avro_schema_manifest_entry: Dict[str, Any]) -> Generator[str, None, None]:
    from fastavro import parse_schema, writer

    parsed_schema = parse_schema(avro_schema_manifest_entry)

    with TemporaryDirectory() as tmpdir:
        tmp_avro_file = tmpdir + "/manifest.avro"
        with open(tmp_avro_file, "wb") as out:
            writer(out, parsed_schema, manifest_entry_records)
        yield tmp_avro_file


@pytest.fixture(scope="session")
def generated_manifest_file_file_v1(
    avro_schema_manifest_file_v1: Dict[str, Any], generated_manifest_entry_file: str
) -> Generator[str, None, None]:
    from fastavro import parse_schema, writer

    parsed_schema = parse_schema(avro_schema_manifest_file_v1)

    # Make sure that a valid manifest_path is set
    manifest_file_records_v1[0]["manifest_path"] = generated_manifest_entry_file

    with TemporaryDirectory() as tmpdir:
        tmp_avro_file = tmpdir + "/manifest.avro"
        with open(tmp_avro_file, "wb") as out:
            writer(out, parsed_schema, manifest_file_records_v1)
        yield tmp_avro_file


@pytest.fixture(scope="session")
def generated_manifest_file_file_v2(
    avro_schema_manifest_file_v2: Dict[str, Any], generated_manifest_entry_file: str
) -> Generator[str, None, None]:
    from fastavro import parse_schema, writer

    parsed_schema = parse_schema(avro_schema_manifest_file_v2)

    # Make sure that a valid manifest_path is set
    manifest_file_records_v2[0]["manifest_path"] = generated_manifest_entry_file

    with TemporaryDirectory() as tmpdir:
        tmp_avro_file = tmpdir + "/manifest.avro"
        with open(tmp_avro_file, "wb") as out:
            writer(out, parsed_schema, manifest_file_records_v2)
        yield tmp_avro_file


@pytest.fixture(scope="session")
def iceberg_manifest_entry_schema() -> Schema:
    return Schema(
        NestedField(field_id=0, name="status", field_type=IntegerType(), required=True),
        NestedField(field_id=1, name="snapshot_id", field_type=LongType(), required=False),
        NestedField(
            field_id=2,
            name="data_file",
            field_type=StructType(
                NestedField(
                    field_id=100,
                    name="file_path",
                    field_type=StringType(),
                    doc="Location URI with FS scheme",
                    required=True,
                ),
                NestedField(
                    field_id=101,
                    name="file_format",
                    field_type=StringType(),
                    doc="File format name: avro, orc, or parquet",
                    required=True,
                ),
                NestedField(
                    field_id=102,
                    name="partition",
                    field_type=StructType(
                        NestedField(
                            field_id=1000,
                            name="VendorID",
                            field_type=IntegerType(),
                            required=False,
                        ),
                        NestedField(
                            field_id=1001,
                            name="tpep_pickup_datetime",
                            field_type=DateType(),
                            required=False,
                        ),
                    ),
                    required=True,
                ),
                NestedField(
                    field_id=103,
                    name="record_count",
                    field_type=LongType(),
                    doc="Number of records in the file",
                    required=True,
                ),
                NestedField(
                    field_id=104,
                    name="file_size_in_bytes",
                    field_type=LongType(),
                    doc="Total file size in bytes",
                    required=True,
                ),
                NestedField(
                    field_id=105,
                    name="block_size_in_bytes",
                    field_type=LongType(),
                    required=True,
                ),
                NestedField(
                    field_id=108,
                    name="column_sizes",
                    field_type=MapType(
                        key_id=117,
                        key_type=IntegerType(),
                        value_id=118,
                        value_type=LongType(),
                        value_required=True,
                    ),
                    doc="Map of column id to total size on disk",
                    required=False,
                ),
                NestedField(
                    field_id=109,
                    name="value_counts",
                    field_type=MapType(
                        key_id=119,
                        key_type=IntegerType(),
                        value_id=120,
                        value_type=LongType(),
                        value_required=True,
                    ),
                    doc="Map of column id to total count, including null and NaN",
                    required=False,
                ),
                NestedField(
                    field_id=110,
                    name="null_value_counts",
                    field_type=MapType(
                        key_id=121,
                        key_type=IntegerType(),
                        value_id=122,
                        value_type=LongType(),
                        value_required=True,
                    ),
                    doc="Map of column id to null value count",
                    required=False,
                ),
                NestedField(
                    field_id=137,
                    name="nan_value_counts",
                    field_type=MapType(
                        key_id=138,
                        key_type=IntegerType(),
                        value_id=139,
                        value_type=LongType(),
                        value_required=True,
                    ),
                    doc="Map of column id to number of NaN values in the column",
                    required=False,
                ),
                NestedField(
                    field_id=125,
                    name="lower_bounds",
                    field_type=MapType(
                        key_id=126,
                        key_type=IntegerType(),
                        value_id=127,
                        value_type=BinaryType(),
                        value_required=True,
                    ),
                    doc="Map of column id to lower bound",
                    required=False,
                ),
                NestedField(
                    field_id=128,
                    name="upper_bounds",
                    field_type=MapType(
                        key_id=129,
                        key_type=IntegerType(),
                        value_id=130,
                        value_type=BinaryType(),
                        value_required=True,
                    ),
                    doc="Map of column id to upper bound",
                    required=False,
                ),
                NestedField(
                    field_id=131,
                    name="key_metadata",
                    field_type=BinaryType(),
                    doc="Encryption key metadata blob",
                    required=False,
                ),
                NestedField(
                    field_id=132,
                    name="split_offsets",
                    field_type=ListType(
                        element_id=133,
                        element_type=LongType(),
                        element_required=True,
                    ),
                    doc="Splittable offsets",
                    required=False,
                ),
                NestedField(
                    field_id=140,
                    name="sort_order_id",
                    field_type=IntegerType(),
                    doc="Sort order ID",
                    required=False,
                ),
            ),
            required=True,
        ),
        schema_id=1,
        identifier_field_ids=[],
    )


@pytest.fixture
def fsspec_fileio(request: pytest.FixtureRequest) -> FsspecFileIO:
    properties = {
        "s3.endpoint": request.config.getoption("--s3.endpoint"),
        "s3.access-key-id": request.config.getoption("--s3.access-key-id"),
        "s3.secret-access-key": request.config.getoption("--s3.secret-access-key"),
    }
    return fsspec.FsspecFileIO(properties=properties)


@pytest.fixture
def fsspec_fileio_gcs(request: pytest.FixtureRequest) -> FsspecFileIO:
    properties = {
        GCS_SERVICE_HOST: request.config.getoption("--gcs.endpoint"),
        GCS_TOKEN: request.config.getoption("--gcs.oauth2.token"),
        GCS_PROJECT_ID: request.config.getoption("--gcs.project-id"),
    }
    return fsspec.FsspecFileIO(properties=properties)


@pytest.fixture
def adls_fsspec_fileio(request: pytest.FixtureRequest) -> Generator[FsspecFileIO, None, None]:
    from azure.storage.blob import BlobServiceClient

    azurite_url = request.config.getoption("--adls.endpoint")
    azurite_account_name = request.config.getoption("--adls.account-name")
    azurite_account_key = request.config.getoption("--adls.account-key")
    azurite_connection_string = f"DefaultEndpointsProtocol=http;AccountName={azurite_account_name};AccountKey={azurite_account_key};BlobEndpoint={azurite_url}/{azurite_account_name};"
    properties = {
        "adls.connection-string": azurite_connection_string,
        "adls.account-name": azurite_account_name,
    }

    bbs = BlobServiceClient.from_connection_string(conn_str=azurite_connection_string)
    bbs.create_container("tests")
    yield fsspec.FsspecFileIO(properties=properties)
    bbs.delete_container("tests")
    bbs.close()


@pytest.fixture
def pyarrow_fileio_gcs(request: pytest.FixtureRequest) -> "PyArrowFileIO":
    from pyiceberg.io.pyarrow import PyArrowFileIO

    properties = {
        GCS_SERVICE_HOST: request.config.getoption("--gcs.endpoint"),
        GCS_TOKEN: request.config.getoption("--gcs.oauth2.token"),
        GCS_PROJECT_ID: request.config.getoption("--gcs.project-id"),
        GCS_TOKEN_EXPIRES_AT_MS: datetime_to_millis(datetime.now()) + 60 * 1000,
    }
    return PyArrowFileIO(properties=properties)


@pytest.fixture
def pyarrow_fileio_adls(request: pytest.FixtureRequest) -> Generator[Any, None, None]:
    from azure.storage.blob import BlobServiceClient

    from pyiceberg.io.pyarrow import PyArrowFileIO

    azurite_url = request.config.getoption("--adls.endpoint")
    azurite_scheme, azurite_authority = azurite_url.split("://", 1)

    azurite_account_name = request.config.getoption("--adls.account-name")
    azurite_account_key = request.config.getoption("--adls.account-key")
    azurite_connection_string = f"DefaultEndpointsProtocol=http;AccountName={azurite_account_name};AccountKey={azurite_account_key};BlobEndpoint={azurite_url}/{azurite_account_name};"
    properties = {
        ADLS_ACCOUNT_NAME: azurite_account_name,
        ADLS_ACCOUNT_KEY: azurite_account_key,
        ADLS_BLOB_STORAGE_AUTHORITY: azurite_authority,
        ADLS_DFS_STORAGE_AUTHORITY: azurite_authority,
        ADLS_BLOB_STORAGE_SCHEME: azurite_scheme,
        ADLS_DFS_STORAGE_SCHEME: azurite_scheme,
    }

    bbs = BlobServiceClient.from_connection_string(conn_str=azurite_connection_string)
    bbs.create_container("warehouse")
    yield PyArrowFileIO(properties=properties)
    bbs.delete_container("warehouse")
    bbs.close()


def aws_credentials() -> None:
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(name="_aws_credentials")
def fixture_aws_credentials() -> Generator[None, None, None]:
    """Yield a mocked AWS Credentials for moto."""
    yield aws_credentials()  # type: ignore
    os.environ.pop("AWS_ACCESS_KEY_ID")
    os.environ.pop("AWS_SECRET_ACCESS_KEY")
    os.environ.pop("AWS_SECURITY_TOKEN")
    os.environ.pop("AWS_SESSION_TOKEN")
    os.environ.pop("AWS_DEFAULT_REGION")


@pytest.fixture(scope="session")
def moto_server() -> "ThreadedMotoServer":
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="localhost", port=5001)

    # this will throw an exception if the port is already in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((server._ip_address, server._port))

    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="session")
def moto_endpoint_url(moto_server: "ThreadedMotoServer") -> str:
    _url = f"http://{moto_server._ip_address}:{moto_server._port}"
    return _url


@pytest.fixture(name="_s3", scope="function")
def fixture_s3(_aws_credentials: None, moto_endpoint_url: str) -> Generator[boto3.client, None, None]:
    """Yield a mocked S3 client."""
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1", endpoint_url=moto_endpoint_url)


@pytest.fixture(name="_glue")
def fixture_glue(_aws_credentials: None) -> Generator[boto3.client, None, None]:
    """Yield a mocked glue client."""
    with mock_aws():
        yield boto3.client("glue", region_name="us-east-1")


@pytest.fixture(name="_dynamodb")
def fixture_dynamodb(_aws_credentials: None) -> Generator[boto3.client, None, None]:
    """Yield a mocked DynamoDB client."""
    with mock_aws():
        yield boto3.client("dynamodb", region_name="us-east-1")


@pytest.fixture(scope="session")
def empty_home_dir_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    home_path = str(tmp_path_factory.mktemp("home"))
    return home_path


RANDOM_LENGTH = 20
NUM_TABLES = 2


@pytest.fixture()
def table_name() -> str:
    prefix = "my_iceberg_table-"
    random_tag = "".join(choice(string.ascii_letters) for _ in range(RANDOM_LENGTH))
    return (prefix + random_tag).lower()


@pytest.fixture()
def table_list(table_name: str) -> List[str]:
    return [f"{table_name}_{idx}" for idx in range(NUM_TABLES)]


@pytest.fixture()
def database_name() -> str:
    prefix = "my_iceberg_database-"
    random_tag = "".join(choice(string.ascii_letters) for _ in range(RANDOM_LENGTH))
    return (prefix + random_tag).lower()


@pytest.fixture()
def database_list(database_name: str) -> List[str]:
    return [f"{database_name}_{idx}" for idx in range(NUM_TABLES)]


@pytest.fixture()
def hierarchical_namespace_name() -> str:
    prefix = "my_iceberg_ns-"
    random_tag1 = "".join(choice(string.ascii_letters) for _ in range(RANDOM_LENGTH))
    random_tag2 = "".join(choice(string.ascii_letters) for _ in range(RANDOM_LENGTH))
    return ".".join([prefix + random_tag1, prefix + random_tag2]).lower()


@pytest.fixture()
def hierarchical_namespace_list(hierarchical_namespace_name: str) -> List[str]:
    return [f"{hierarchical_namespace_name}_{idx}" for idx in range(NUM_TABLES)]


BUCKET_NAME = "test_bucket"
TABLE_METADATA_LOCATION_REGEX = re.compile(
    r"""s3://test_bucket/my_iceberg_database-[a-z]{20}.db/
    my_iceberg_table-[a-z]{20}/metadata/
    [0-9]{5}-[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}.metadata.json""",
    re.X,
)

UNIFIED_AWS_SESSION_PROPERTIES = {
    "client.access-key-id": "client.access-key-id",
    "client.secret-access-key": "client.secret-access-key",
    "client.region": "client.region",
    "client.session-token": "client.session-token",
}


@pytest.fixture(name="_bucket_initialize")
def fixture_s3_bucket(_s3) -> None:  # type: ignore
    _s3.create_bucket(Bucket=BUCKET_NAME)


def get_bucket_name() -> str:
    """Set the environment variable AWS_TEST_BUCKET for a default bucket to test."""
    bucket_name = os.getenv("AWS_TEST_BUCKET")
    if bucket_name is None:
        raise ValueError("Please specify a bucket to run the test by setting environment variable AWS_TEST_BUCKET")
    return bucket_name


def get_glue_endpoint() -> Optional[str]:
    """Set the optional environment variable AWS_TEST_GLUE_ENDPOINT for a glue endpoint to test."""
    return os.getenv("AWS_TEST_GLUE_ENDPOINT")


def get_s3_path(bucket_name: str, database_name: Optional[str] = None, table_name: Optional[str] = None) -> str:
    result_path = f"s3://{bucket_name}"
    if database_name is not None:
        result_path += f"/{database_name}.db"

    if table_name is not None:
        result_path += f"/{table_name}"
    return result_path


@pytest.fixture(name="s3", scope="module")
def fixture_s3_client() -> boto3.client:
    """Real S3 client for AWS Integration Tests."""
    yield boto3.client("s3")


def clean_up(test_catalog: Catalog) -> None:
    """Clean all databases and tables created during the integration test."""
    for database_tuple in test_catalog.list_namespaces():
        database_name = database_tuple[0]
        if "my_iceberg_database-" in database_name:
            for identifier in test_catalog.list_tables(database_name):
                test_catalog.drop_table(identifier)
            test_catalog.drop_namespace(database_name)


@pytest.fixture
def data_file(table_schema_simple: Schema, tmp_path: str) -> str:
    import pyarrow as pa
    from pyarrow import parquet as pq

    from pyiceberg.io.pyarrow import schema_to_pyarrow

    table = pa.table(
        {"foo": ["a", "b", "c"], "bar": [1, 2, 3], "baz": [True, False, None]},
        schema=schema_to_pyarrow(table_schema_simple),
    )

    file_path = f"{tmp_path}/0000-data.parquet"
    pq.write_table(table=table, where=file_path)
    return file_path


@pytest.fixture
def example_task(data_file: str) -> FileScanTask:
    return FileScanTask(
        data_file=DataFile.from_args(file_path=data_file, file_format=FileFormat.PARQUET, file_size_in_bytes=1925),
    )


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_sql")


@pytest.fixture
def table_v1(example_table_metadata_v1: Dict[str, Any]) -> Table:
    table_metadata = TableMetadataV1(**example_table_metadata_v1)
    return Table(
        identifier=("database", "table"),
        metadata=table_metadata,
        metadata_location=f"{table_metadata.location}/uuid.metadata.json",
        io=load_file_io(),
        catalog=NoopCatalog("NoopCatalog"),
    )


@pytest.fixture
def table_v2(example_table_metadata_v2: Dict[str, Any]) -> Table:
    table_metadata = TableMetadataV2(**example_table_metadata_v2)
    return Table(
        identifier=("database", "table"),
        metadata=table_metadata,
        metadata_location=f"{table_metadata.location}/uuid.metadata.json",
        io=load_file_io(),
        catalog=NoopCatalog("NoopCatalog"),
    )


@pytest.fixture
def table_v2_with_fixed_and_decimal_types(
    table_metadata_v2_with_fixed_and_decimal_types: Dict[str, Any],
) -> Table:
    table_metadata = TableMetadataV2(
        **table_metadata_v2_with_fixed_and_decimal_types,
    )
    return Table(
        identifier=("database", "table"),
        metadata=table_metadata,
        metadata_location=f"{table_metadata.location}/uuid.metadata.json",
        io=load_file_io(),
        catalog=NoopCatalog("NoopCatalog"),
    )


@pytest.fixture
def table_v2_with_extensive_snapshots(example_table_metadata_v2_with_extensive_snapshots: Dict[str, Any]) -> Table:
    table_metadata = TableMetadataV2(**example_table_metadata_v2_with_extensive_snapshots)
    return Table(
        identifier=("database", "table"),
        metadata=table_metadata,
        metadata_location=f"{table_metadata.location}/uuid.metadata.json",
        io=load_file_io(),
        catalog=NoopCatalog("NoopCatalog"),
    )


@pytest.fixture
def table_v2_with_statistics(table_metadata_v2_with_statistics: Dict[str, Any]) -> Table:
    table_metadata = TableMetadataV2(**table_metadata_v2_with_statistics)
    return Table(
        identifier=("database", "table"),
        metadata=table_metadata,
        metadata_location=f"{table_metadata.location}/uuid.metadata.json",
        io=load_file_io(),
        catalog=NoopCatalog("NoopCatalog"),
    )


@pytest.fixture
def bound_reference_str() -> BoundReference[str]:
    return BoundReference(field=NestedField(1, "field", StringType(), required=False), accessor=Accessor(position=0, inner=None))


@pytest.fixture
def bound_reference_binary() -> BoundReference[str]:
    return BoundReference(field=NestedField(1, "field", BinaryType(), required=False), accessor=Accessor(position=0, inner=None))


@pytest.fixture
def bound_reference_uuid() -> BoundReference[str]:
    return BoundReference(field=NestedField(1, "field", UUIDType(), required=False), accessor=Accessor(position=0, inner=None))


@pytest.fixture(scope="session")
def session_catalog() -> Catalog:
    return load_catalog(
        "local",
        **{
            "type": "rest",
            "uri": "http://localhost:8181",
            "s3.endpoint": "http://localhost:9000",
            "s3.access-key-id": "admin",
            "s3.secret-access-key": "password",
        },
    )


@pytest.fixture(scope="session")
def session_catalog_hive() -> Catalog:
    return load_catalog(
        "local",
        **{
            "type": "hive",
            "uri": "http://localhost:9083",
            "s3.endpoint": "http://localhost:9000",
            "s3.access-key-id": "admin",
            "s3.secret-access-key": "password",
        },
    )


@pytest.fixture(scope="session")
def spark() -> "SparkSession":
    import importlib.metadata

    from pyspark.sql import SparkSession

    # Remember to also update `dev/Dockerfile`
    spark_version = ".".join(importlib.metadata.version("pyspark").split(".")[:2])
    scala_version = "2.12"
    iceberg_version = "1.9.0"

    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        f"--packages org.apache.iceberg:iceberg-spark-runtime-{spark_version}_{scala_version}:{iceberg_version},"
        f"org.apache.iceberg:iceberg-aws-bundle:{iceberg_version} pyspark-shell"
    )
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    spark = (
        SparkSession.builder.appName("PyIceberg integration test")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.integration", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.integration.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config("spark.sql.catalog.integration.cache-enabled", "false")
        .config("spark.sql.catalog.integration.uri", "http://localhost:8181")
        .config("spark.sql.catalog.integration.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config("spark.sql.catalog.integration.warehouse", "s3://warehouse/wh/")
        .config("spark.sql.catalog.integration.s3.endpoint", "http://localhost:9000")
        .config("spark.sql.catalog.integration.s3.path-style-access", "true")
        .config("spark.sql.defaultCatalog", "integration")
        .config("spark.sql.catalog.hive", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.hive.type", "hive")
        .config("spark.sql.catalog.hive.uri", "http://localhost:9083")
        .config("spark.sql.catalog.hive.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config("spark.sql.catalog.hive.warehouse", "s3://warehouse/hive/")
        .config("spark.sql.catalog.hive.s3.endpoint", "http://localhost:9000")
        .config("spark.sql.catalog.hive.s3.path-style-access", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    return spark


TEST_DATA_WITH_NULL = {
    "bool": [False, None, True],
    "string": ["a", None, "z"],
    # Go over the 16 bytes to kick in truncation
    "string_long": ["a" * 22, None, "z" * 22],
    "int": [1, None, 9],
    "long": [1, None, 9],
    "float": [0.0, None, 0.9],
    "double": [0.0, None, 0.9],
    # 'time': [1_000_000, None, 3_000_000],  # Example times: 1s, none, and 3s past midnight #Spark does not support time fields
    "timestamp": [datetime(2023, 1, 1, 19, 25, 00), None, datetime(2023, 3, 1, 19, 25, 00)],
    "timestamptz": [
        datetime(2023, 1, 1, 19, 25, 00, tzinfo=timezone.utc),
        None,
        datetime(2023, 3, 1, 19, 25, 00, tzinfo=timezone.utc),
    ],
    "date": [date(2023, 1, 1), None, date(2023, 3, 1)],
    # Not supported by Spark
    # 'time': [time(1, 22, 0), None, time(19, 25, 0)],
    # Not natively supported by Arrow
    # 'uuid': [uuid.UUID('00000000-0000-0000-0000-000000000000').bytes, None, uuid.UUID('11111111-1111-1111-1111-111111111111').bytes],
    "binary": [b"\01", None, b"\22"],
    "fixed": [
        uuid.UUID("00000000-0000-0000-0000-000000000000").bytes,
        None,
        uuid.UUID("11111111-1111-1111-1111-111111111111").bytes,
    ],
}


@pytest.fixture(scope="session")
def pa_schema() -> "pa.Schema":
    import pyarrow as pa

    return pa.schema(
        [
            ("bool", pa.bool_()),
            ("string", pa.large_string()),
            ("string_long", pa.large_string()),
            ("int", pa.int32()),
            ("long", pa.int64()),
            ("float", pa.float32()),
            ("double", pa.float64()),
            # Not supported by Spark
            # ("time", pa.time64('us')),
            ("timestamp", pa.timestamp(unit="us")),
            ("timestamptz", pa.timestamp(unit="us", tz="UTC")),
            ("date", pa.date32()),
            # Not supported by Spark
            # ("time", pa.time64("us")),
            # Not natively supported by Arrow
            # ("uuid", pa.fixed(16)),
            ("binary", pa.large_binary()),
            ("fixed", pa.binary(16)),
        ]
    )


@pytest.fixture(scope="session")
def arrow_table_with_null(pa_schema: "pa.Schema") -> "pa.Table":
    """Pyarrow table with all kinds of columns."""
    import pyarrow as pa

    return pa.Table.from_pydict(
        {
            "bool": [False, None, True],
            "string": ["a", None, "z"],
            # Go over the 16 bytes to kick in truncation
            "string_long": ["a" * 22, None, "z" * 22],
            "int": [1, None, 9],
            "long": [1, None, 9],
            "float": [0.0, None, 0.9],
            "double": [0.0, None, 0.9],
            # 'time': [1_000_000, None, 3_000_000],  # Example times: 1s, none, and 3s past midnight #Spark does not support time fields
            "timestamp": [datetime(2023, 1, 1, 19, 25, 00), None, datetime(2023, 3, 1, 19, 25, 00)],
            "timestamptz": [
                datetime(2023, 1, 1, 19, 25, 00, tzinfo=timezone.utc),
                None,
                datetime(2023, 3, 1, 19, 25, 00, tzinfo=timezone.utc),
            ],
            "date": [date(2023, 1, 1), None, date(2023, 3, 1)],
            # Not supported by Spark
            # 'time': [time(1, 22, 0), None, time(19, 25, 0)],
            # Not natively supported by Arrow
            # 'uuid': [uuid.UUID('00000000-0000-0000-0000-000000000000').bytes, None, uuid.UUID('11111111-1111-1111-1111-111111111111').bytes],
            "binary": [b"\01", None, b"\22"],
            "fixed": [
                uuid.UUID("00000000-0000-0000-0000-000000000000").bytes,
                None,
                uuid.UUID("11111111-1111-1111-1111-111111111111").bytes,
            ],
        },
        schema=pa_schema,
    )


@pytest.fixture(scope="session")
def arrow_table_without_data(pa_schema: "pa.Schema") -> "pa.Table":
    """Pyarrow table without data."""
    import pyarrow as pa

    return pa.Table.from_pylist([], schema=pa_schema)


@pytest.fixture(scope="session")
def arrow_table_with_only_nulls(pa_schema: "pa.Schema") -> "pa.Table":
    """Pyarrow table with only null values."""
    import pyarrow as pa

    return pa.Table.from_pylist([{}, {}], schema=pa_schema)


@pytest.fixture(scope="session")
def arrow_table_date_timestamps() -> "pa.Table":
    """Pyarrow table with only date, timestamp and timestamptz values."""
    import pyarrow as pa

    return pa.Table.from_pydict(
        {
            "date": [date(2023, 12, 31), date(2024, 1, 1), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 1), None],
            "timestamp": [
                datetime(2023, 12, 31, 0, 0, 0),
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 31, 0, 0, 0),
                datetime(2024, 2, 1, 0, 0, 0),
                datetime(2024, 2, 1, 6, 0, 0),
                None,
            ],
            "timestamptz": [
                datetime(2023, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 31, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 2, 1, 6, 0, 0, tzinfo=timezone.utc),
                None,
            ],
        },
        schema=pa.schema(
            [
                ("date", pa.date32()),
                ("timestamp", pa.timestamp(unit="us")),
                ("timestamptz", pa.timestamp(unit="us", tz="UTC")),
            ]
        ),
    )


@pytest.fixture(scope="session")
def table_date_timestamps_schema() -> Schema:
    """Iceberg table Schema with only date, timestamp and timestamptz values."""
    return Schema(
        NestedField(field_id=1, name="date", field_type=DateType(), required=False),
        NestedField(field_id=2, name="timestamp", field_type=TimestampType(), required=False),
        NestedField(field_id=3, name="timestamptz", field_type=TimestamptzType(), required=False),
    )


@pytest.fixture(scope="session")
def arrow_table_schema_with_all_timestamp_precisions() -> "pa.Schema":
    """Pyarrow Schema with all supported timestamp types."""
    import pyarrow as pa

    return pa.schema(
        [
            ("timestamp_s", pa.timestamp(unit="s")),
            ("timestamptz_s", pa.timestamp(unit="s", tz="UTC")),
            ("timestamp_ms", pa.timestamp(unit="ms")),
            ("timestamptz_ms", pa.timestamp(unit="ms", tz="UTC")),
            ("timestamp_us", pa.timestamp(unit="us")),
            ("timestamptz_us", pa.timestamp(unit="us", tz="UTC")),
            ("timestamp_ns", pa.timestamp(unit="ns")),
            ("timestamptz_ns", pa.timestamp(unit="ns", tz="UTC")),
            ("timestamptz_us_etc_utc", pa.timestamp(unit="us", tz="Etc/UTC")),
            ("timestamptz_ns_z", pa.timestamp(unit="ns", tz="Z")),
            ("timestamptz_s_0000", pa.timestamp(unit="s", tz="+00:00")),
        ]
    )


@pytest.fixture(scope="session")
def arrow_table_with_all_timestamp_precisions(arrow_table_schema_with_all_timestamp_precisions: "pa.Schema") -> "pa.Table":
    """Pyarrow table with all supported timestamp types."""
    import pandas as pd
    import pyarrow as pa

    test_data = pd.DataFrame(
        {
            "timestamp_s": [datetime(2023, 1, 1, 19, 25, 00), None, datetime(2023, 3, 1, 19, 25, 00)],
            "timestamptz_s": [
                datetime(2023, 1, 1, 19, 25, 00, tzinfo=timezone.utc),
                None,
                datetime(2023, 3, 1, 19, 25, 00, tzinfo=timezone.utc),
            ],
            "timestamp_ms": [datetime(2023, 1, 1, 19, 25, 00), None, datetime(2023, 3, 1, 19, 25, 00)],
            "timestamptz_ms": [
                datetime(2023, 1, 1, 19, 25, 00, tzinfo=timezone.utc),
                None,
                datetime(2023, 3, 1, 19, 25, 00, tzinfo=timezone.utc),
            ],
            "timestamp_us": [datetime(2023, 1, 1, 19, 25, 00), None, datetime(2023, 3, 1, 19, 25, 00)],
            "timestamptz_us": [
                datetime(2023, 1, 1, 19, 25, 00, tzinfo=timezone.utc),
                None,
                datetime(2023, 3, 1, 19, 25, 00, tzinfo=timezone.utc),
            ],
            "timestamp_ns": [
                pd.Timestamp(year=2024, month=7, day=11, hour=3, minute=30, second=0, microsecond=12, nanosecond=6),
                None,
                pd.Timestamp(year=2024, month=7, day=11, hour=3, minute=30, second=0, microsecond=12, nanosecond=7),
            ],
            "timestamptz_ns": [
                datetime(2023, 1, 1, 19, 25, 00, tzinfo=timezone.utc),
                None,
                datetime(2023, 3, 1, 19, 25, 00, tzinfo=timezone.utc),
            ],
            "timestamptz_us_etc_utc": [
                datetime(2023, 1, 1, 19, 25, 00, tzinfo=timezone.utc),
                None,
                datetime(2023, 3, 1, 19, 25, 00, tzinfo=timezone.utc),
            ],
            "timestamptz_ns_z": [
                pd.Timestamp(year=2024, month=7, day=11, hour=3, minute=30, second=0, microsecond=12, nanosecond=6, tz="UTC"),
                None,
                pd.Timestamp(year=2024, month=7, day=11, hour=3, minute=30, second=0, microsecond=12, nanosecond=7, tz="UTC"),
            ],
            "timestamptz_s_0000": [
                datetime(2023, 1, 1, 19, 25, 1, tzinfo=timezone.utc),
                None,
                datetime(2023, 3, 1, 19, 25, 1, tzinfo=timezone.utc),
            ],
        }
    )
    return pa.Table.from_pandas(test_data, schema=arrow_table_schema_with_all_timestamp_precisions)


@pytest.fixture(scope="session")
def arrow_table_schema_with_all_microseconds_timestamp_precisions() -> "pa.Schema":
    """Pyarrow Schema with all microseconds timestamp."""
    import pyarrow as pa

    return pa.schema(
        [
            ("timestamp_s", pa.timestamp(unit="us")),
            ("timestamptz_s", pa.timestamp(unit="us", tz="UTC")),
            ("timestamp_ms", pa.timestamp(unit="us")),
            ("timestamptz_ms", pa.timestamp(unit="us", tz="UTC")),
            ("timestamp_us", pa.timestamp(unit="us")),
            ("timestamptz_us", pa.timestamp(unit="us", tz="UTC")),
            ("timestamp_ns", pa.timestamp(unit="us")),
            ("timestamptz_ns", pa.timestamp(unit="us", tz="UTC")),
            ("timestamptz_us_etc_utc", pa.timestamp(unit="us", tz="UTC")),
            ("timestamptz_ns_z", pa.timestamp(unit="us", tz="UTC")),
            ("timestamptz_s_0000", pa.timestamp(unit="us", tz="UTC")),
        ]
    )


@pytest.fixture(scope="session")
def table_schema_with_all_microseconds_timestamp_precision() -> Schema:
    """Iceberg table Schema with only date, timestamp and timestamptz values."""
    return Schema(
        NestedField(field_id=1, name="timestamp_s", field_type=TimestampType(), required=False),
        NestedField(field_id=2, name="timestamptz_s", field_type=TimestamptzType(), required=False),
        NestedField(field_id=3, name="timestamp_ms", field_type=TimestampType(), required=False),
        NestedField(field_id=4, name="timestamptz_ms", field_type=TimestamptzType(), required=False),
        NestedField(field_id=5, name="timestamp_us", field_type=TimestampType(), required=False),
        NestedField(field_id=6, name="timestamptz_us", field_type=TimestamptzType(), required=False),
        NestedField(field_id=7, name="timestamp_ns", field_type=TimestampType(), required=False),
        NestedField(field_id=8, name="timestamptz_ns", field_type=TimestamptzType(), required=False),
        NestedField(field_id=9, name="timestamptz_us_etc_utc", field_type=TimestamptzType(), required=False),
        NestedField(field_id=10, name="timestamptz_ns_z", field_type=TimestamptzType(), required=False),
        NestedField(field_id=11, name="timestamptz_s_0000", field_type=TimestamptzType(), required=False),
    )


@pytest.fixture(scope="session")
def table_schema_with_promoted_types() -> Schema:
    """Iceberg table Schema with longs, doubles and uuid in simple and nested types."""
    return Schema(
        NestedField(field_id=1, name="long", field_type=LongType(), required=False),
        NestedField(
            field_id=2,
            name="list",
            field_type=ListType(element_id=4, element_type=LongType(), element_required=False),
            required=True,
        ),
        NestedField(
            field_id=3,
            name="map",
            field_type=MapType(
                key_id=5,
                key_type=StringType(),
                value_id=6,
                value_type=LongType(),
                value_required=False,
            ),
            required=True,
        ),
        NestedField(field_id=7, name="double", field_type=DoubleType(), required=False),
        NestedField(field_id=8, name="uuid", field_type=UUIDType(), required=False),
    )


@pytest.fixture(scope="session")
def pyarrow_schema_with_promoted_types() -> "pa.Schema":
    """Pyarrow Schema with longs, doubles and uuid in simple and nested types."""
    import pyarrow as pa

    return pa.schema(
        (
            pa.field("long", pa.int32(), nullable=True),  # can support upcasting integer to long
            pa.field("list", pa.list_(pa.int32()), nullable=False),  # can support upcasting integer to long
            pa.field("map", pa.map_(pa.string(), pa.int32()), nullable=False),  # can support upcasting integer to long
            pa.field("double", pa.float32(), nullable=True),  # can support upcasting float to double
            pa.field("uuid", pa.binary(length=16), nullable=True),  # can support upcasting fixed to uuid
        )
    )


@pytest.fixture(scope="session")
def pyarrow_table_with_promoted_types(pyarrow_schema_with_promoted_types: "pa.Schema") -> "pa.Table":
    """Pyarrow table with longs, doubles and uuid in simple and nested types."""
    import pyarrow as pa

    return pa.Table.from_pydict(
        {
            "long": [1, 9],
            "list": [[1, 1], [2, 2]],
            "map": [{"a": 1}, {"b": 2}],
            "double": [1.1, 9.2],
            "uuid": [
                uuid.UUID("00000000-0000-0000-0000-000000000000").bytes,
                uuid.UUID("11111111-1111-1111-1111-111111111111").bytes,
            ],
        },
        schema=pyarrow_schema_with_promoted_types,
    )
