#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
import inspect
from _decimal import Decimal
from datetime import datetime
from enum import Enum
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID

import pytest
from fastavro import reader, writer

import pyiceberg.avro.file as avro
from pyiceberg.avro.codecs.deflate import DeflateCodec
from pyiceberg.avro.file import AvroFileHeader
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (
    DEFAULT_BLOCK_SIZE,
    MANIFEST_ENTRY_SCHEMAS,
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestEntry,
    ManifestEntryStatus,
)
from pyiceberg.schema import Schema
from pyiceberg.typedef import Record, TableVersion
from pyiceberg.types import (
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FixedType,
    FloatType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    TimestampType,
    TimestamptzType,
    TimeType,
    UUIDType,
)
from pyiceberg.utils.schema_conversion import AvroSchemaConversion


def get_deflate_compressor() -> None:
    header = AvroFileHeader(bytes(0), {"avro.codec": "deflate"}, bytes(16))
    assert header.compression_codec() == DeflateCodec


def get_null_compressor() -> None:
    header = AvroFileHeader(bytes(0), {"avro.codec": "null"}, bytes(16))
    assert header.compression_codec() is None


def test_unknown_codec() -> None:
    header = AvroFileHeader(bytes(0), {"avro.codec": "unknown"}, bytes(16))

    with pytest.raises(ValueError) as exc_info:
        header.compression_codec()

    assert "Unsupported codec: unknown" in str(exc_info.value)


def test_missing_schema() -> None:
    header = AvroFileHeader(bytes(0), {}, bytes(16))

    with pytest.raises(ValueError) as exc_info:
        header.get_schema()

    assert "No schema found in Avro file headers" in str(exc_info.value)


# helper function to serialize our objects to dicts to enable
# direct comparison with the dicts returned by fastavro
def todict(obj: Any) -> Any:
    if isinstance(obj, dict):
        data = []
        for k, v in obj.items():
            data.append({"key": k, "value": v})
        return data
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, "__iter__") and not isinstance(obj, str) and not isinstance(obj, bytes):
        return [todict(v) for v in obj]
    elif isinstance(obj, Record):
        return {key: todict(value) for key, value in inspect.getmembers(obj) if not callable(value) and not key.startswith("_")}
    else:
        return obj


def test_write_manifest_entry_with_iceberg_read_with_fastavro_v1() -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://some-path/some-file.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(),
        record_count=131327,
        file_size_in_bytes=220669226,
        column_sizes={1: 220661854},
        value_counts={1: 131327},
        null_value_counts={1: 0},
        nan_value_counts={},
        lower_bounds={1: b"aaaaaaaaaaaaaaaa"},
        upper_bounds={1: b"zzzzzzzzzzzzzzzz"},
        key_metadata=b"\xde\xad\xbe\xef",
        split_offsets=[4, 133697593],
        equality_ids=[],
        sort_order_id=4,
    )
    entry = ManifestEntry.from_args(
        status=ManifestEntryStatus.ADDED,
        snapshot_id=8638475580105682862,
        sequence_number=0,
        file_sequence_number=0,
        data_file=data_file,
    )

    additional_metadata = {"foo": "bar"}

    with TemporaryDirectory() as tmpdir:
        tmp_avro_file = tmpdir + "/manifest_entry.avro"

        with avro.AvroOutputFile[ManifestEntry](
            output_file=PyArrowFileIO().new_output(tmp_avro_file),
            file_schema=MANIFEST_ENTRY_SCHEMAS[1],
            schema_name="manifest_entry",
            record_schema=MANIFEST_ENTRY_SCHEMAS[2],
            metadata=additional_metadata,
        ) as out:
            out.write_block([entry])

        with open(tmp_avro_file, "rb") as fo:
            r = reader(fo=fo)

            for k, v in additional_metadata.items():
                assert k in r.metadata
                assert v == r.metadata[k]

            it = iter(r)

            fa_entry = next(it)

        v2_entry = todict(entry)

        # These are not written in V1
        del v2_entry["sequence_number"]
        del v2_entry["file_sequence_number"]
        del v2_entry["data_file"]["content"]
        del v2_entry["data_file"]["equality_ids"]

        # Required in V1
        v2_entry["data_file"]["block_size_in_bytes"] = DEFAULT_BLOCK_SIZE

        assert v2_entry == fa_entry


def test_write_manifest_entry_with_iceberg_read_with_fastavro_v2() -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://some-path/some-file.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(),
        record_count=131327,
        file_size_in_bytes=220669226,
        column_sizes={1: 220661854},
        value_counts={1: 131327},
        null_value_counts={1: 0},
        nan_value_counts={},
        lower_bounds={1: b"aaaaaaaaaaaaaaaa"},
        upper_bounds={1: b"zzzzzzzzzzzzzzzz"},
        key_metadata=b"\xde\xad\xbe\xef",
        split_offsets=[4, 133697593],
        equality_ids=[],
        sort_order_id=4,
    )
    entry = ManifestEntry.from_args(
        status=ManifestEntryStatus.ADDED,
        snapshot_id=8638475580105682862,
        sequence_number=0,
        file_sequence_number=0,
        data_file=data_file,
    )

    additional_metadata = {"foo": "bar"}

    with TemporaryDirectory() as tmpdir:
        tmp_avro_file = tmpdir + "/manifest_entry.avro"

        with avro.AvroOutputFile[ManifestEntry](
            output_file=PyArrowFileIO().new_output(tmp_avro_file),
            file_schema=MANIFEST_ENTRY_SCHEMAS[2],
            schema_name="manifest_entry",
            metadata=additional_metadata,
        ) as out:
            out.write_block([entry])

        with open(tmp_avro_file, "rb") as fo:
            r = reader(fo=fo)

            for k, v in additional_metadata.items():
                assert k in r.metadata
                assert v == r.metadata[k]

            it = iter(r)

            fa_entry = next(it)

        assert todict(entry) == fa_entry


@pytest.mark.parametrize("format_version", [1, 2])
def test_write_manifest_entry_with_fastavro_read_with_iceberg(format_version: TableVersion) -> None:
    data_file_dict = {
        "content": DataFileContent.DATA,
        "file_path": "s3://some-path/some-file.parquet",
        "file_format": FileFormat.PARQUET,
        "partition": Record(),
        "record_count": 131327,
        "file_size_in_bytes": 220669226,
        "column_sizes": {1: 220661854},
        "value_counts": {1: 131327},
        "null_value_counts": {1: 0},
        "nan_value_counts": {},
        "lower_bounds": {1: b"aaaaaaaaaaaaaaaa"},
        "upper_bounds": {1: b"zzzzzzzzzzzzzzzz"},
        "key_metadata": b"\xde\xad\xbe\xef",
        "split_offsets": [4, 133697593],
        "equality_ids": [],
        "sort_order_id": 4,
        "spec_id": 3,
    }
    data_file_v2 = DataFile.from_args(**data_file_dict)  # type: ignore

    entry = ManifestEntry.from_args(
        status=ManifestEntryStatus.ADDED,
        snapshot_id=8638475580105682862,
        data_file=data_file_v2,
    )

    with TemporaryDirectory() as tmpdir:
        tmp_avro_file = tmpdir + "/manifest_entry.avro"

        schema = AvroSchemaConversion().iceberg_to_avro(MANIFEST_ENTRY_SCHEMAS[format_version], schema_name="manifest_entry")

        with open(tmp_avro_file, "wb") as out:
            writer(out, schema, [todict(entry)])

        # Read as V2
        with avro.AvroFile[ManifestEntry](
            input_file=PyArrowFileIO().new_input(tmp_avro_file),
            read_schema=MANIFEST_ENTRY_SCHEMAS[2],
            read_types={-1: ManifestEntry, 2: DataFile},
        ) as avro_reader:
            it = iter(avro_reader)
            avro_entry = next(it)

            assert entry == avro_entry

        # Read as the original version
        with avro.AvroFile[ManifestEntry](
            input_file=PyArrowFileIO().new_input(tmp_avro_file),
            read_schema=MANIFEST_ENTRY_SCHEMAS[format_version],
            read_types={-1: ManifestEntry, 2: DataFile},
        ) as avro_reader:
            it = iter(avro_reader)
            avro_entry = next(it)

            if format_version == 1:
                data_file_v1 = DataFile.from_args(**data_file_dict, _table_format_version=format_version)

                assert avro_entry == ManifestEntry.from_args(
                    status=1,
                    snapshot_id=8638475580105682862,
                    data_file=data_file_v1,
                    _table_format_version=format_version,
                )
            elif format_version == 2:
                assert entry == avro_entry
            else:
                raise ValueError(f"Unsupported version: {format_version}")


@pytest.mark.parametrize("is_required", [True, False])
def test_all_primitive_types(is_required: bool) -> None:
    all_primitives_schema = Schema(
        NestedField(field_id=1, name="field_fixed", field_type=FixedType(16), required=is_required),
        NestedField(field_id=2, name="field_decimal", field_type=DecimalType(6, 2), required=is_required),
        NestedField(field_id=3, name="field_bool", field_type=BooleanType(), required=is_required),
        NestedField(field_id=4, name="field_int", field_type=IntegerType(), required=True),
        NestedField(field_id=5, name="field_long", field_type=LongType(), required=is_required),
        NestedField(field_id=6, name="field_float", field_type=FloatType(), required=is_required),
        NestedField(field_id=7, name="field_double", field_type=DoubleType(), required=is_required),
        NestedField(field_id=8, name="field_date", field_type=DateType(), required=is_required),
        NestedField(field_id=9, name="field_time", field_type=TimeType(), required=is_required),
        NestedField(field_id=10, name="field_timestamp", field_type=TimestampType(), required=is_required),
        NestedField(field_id=11, name="field_timestamptz", field_type=TimestamptzType(), required=is_required),
        NestedField(field_id=12, name="field_string", field_type=StringType(), required=is_required),
        NestedField(field_id=13, name="field_uuid", field_type=UUIDType(), required=is_required),
        schema_id=1,
    )

    class AllPrimitivesRecord(Record):
        @property
        def field_fixed(self) -> bytes:
            return self._data[0]

        @property
        def field_decimal(self) -> Decimal:
            return self._data[1]

        @property
        def field_bool(self) -> bool:
            return self._data[2]

        @property
        def field_int(self) -> int:
            return self._data[3]

        @property
        def field_long(self) -> int:
            return self._data[4]

        @property
        def field_float(self) -> float:
            return self._data[5]

        @property
        def field_double(self) -> float:
            return self._data[6]

        @property
        def field_date(self) -> datetime:
            return self._data[7]

        @property
        def field_time(self) -> datetime:
            return self._data[8]

        @property
        def field_timestamp(self) -> datetime:
            return self._data[9]

        @property
        def field_timestamptz(self) -> datetime:
            return self._data[10]

        @property
        def field_string(self) -> str:
            return self._data[11]

        @property
        def field_uuid(self) -> UUID:
            return self._data[12]

    record = AllPrimitivesRecord(
        b"\x124Vx\x124Vx\x124Vx\x124Vx",
        Decimal("123.45"),
        True,
        123,
        429496729622,
        123.22000122070312,
        429496729622.314,
        19052,
        69922000000,
        1677629965000000,
        1677629965000000,
        "this is a sentence",
        UUID("12345678-1234-5678-1234-567812345678"),
    )

    with TemporaryDirectory() as tmpdir:
        tmp_avro_file = tmpdir + "/all_primitives.avro"
        # write to disk
        with avro.AvroOutputFile[AllPrimitivesRecord](
            PyArrowFileIO().new_output(tmp_avro_file), all_primitives_schema, "all_primitives_schema"
        ) as out:
            out.write_block([record])

        # read from disk
        with avro.AvroFile[AllPrimitivesRecord](
            PyArrowFileIO().new_input(tmp_avro_file),
            all_primitives_schema,
            {-1: AllPrimitivesRecord},
        ) as avro_reader:
            it = iter(avro_reader)
            avro_entry = next(it)

        # read with fastavro
        with open(tmp_avro_file, "rb") as fo:
            r = reader(fo=fo)
            it_fastavro = iter(r)
            avro_entry_read_with_fastavro = list(next(it_fastavro).values())

    for idx, field in enumerate(all_primitives_schema.as_struct()):
        assert record[idx] == avro_entry[idx], f"Invalid {field}"
        assert record[idx] == avro_entry_read_with_fastavro[idx], f"Invalid {field} read with fastavro"
