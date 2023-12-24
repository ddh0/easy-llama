# gguf_reader.py
# Python 3.11.6

import struct
from enum import IntEnum

class GGUFReader:
    """
    Peek at file header for GGUF metadata

    Raise ValueError if file is not GGUF or is outdated

    Credit to oobabooga for the parts of the code in this class

    Format spec: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
    """

    class GGUFValueType(IntEnum):
        UINT8 = 0
        INT8 = 1
        UINT16 = 2
        INT16 = 3
        UINT32 = 4
        INT32 = 5
        FLOAT32 = 6
        BOOL = 7
        STRING = 8
        ARRAY = 9
        UINT64 = 10
        INT64 = 11
        FLOAT64 = 12

    _simple_value_packing = {
        GGUFValueType.UINT8: "<B",
        GGUFValueType.INT8: "<b",
        GGUFValueType.UINT16: "<H",
        GGUFValueType.INT16: "<h",
        GGUFValueType.UINT32: "<I",
        GGUFValueType.INT32: "<i",
        GGUFValueType.FLOAT32: "<f",
        GGUFValueType.UINT64: "<Q",
        GGUFValueType.INT64: "<q",
        GGUFValueType.FLOAT64: "<d",
        GGUFValueType.BOOL: "?",
    }

    value_type_info = {
        GGUFValueType.UINT8: 1,
        GGUFValueType.INT8: 1,
        GGUFValueType.UINT16: 2,
        GGUFValueType.INT16: 2,
        GGUFValueType.UINT32: 4,
        GGUFValueType.INT32: 4,
        GGUFValueType.FLOAT32: 4,
        GGUFValueType.UINT64: 8,
        GGUFValueType.INT64: 8,
        GGUFValueType.FLOAT64: 8,
        GGUFValueType.BOOL: 1,
    }

    def get_single(self, value_type, file):
        if value_type == GGUFReader.GGUFValueType.STRING:
            value_length = struct.unpack("<Q", file.read(8))[0]
            value = file.read(value_length)
            value = value.decode("utf-8")
        else:
            type_str = GGUFReader._simple_value_packing.get(value_type)
            bytes_length = GGUFReader.value_type_info.get(value_type)
            value = struct.unpack(type_str, file.read(bytes_length))[0]

        return value

    def load_metadata(self, fname) -> dict:
        metadata = {}
        with open(fname, "rb") as file:
            GGUF_MAGIC = file.read(4)
            GGUF_VERSION = struct.unpack("<I", file.read(4))[0]
            # ti_data_count = struct.unpack("<Q", file.read(8))[0]
            file.read(8)
            kv_data_count = struct.unpack("<Q", file.read(8))[0]

            if GGUF_MAGIC != b"GGUF":
                raise ValueError(
                    "easy_llama: your model file is not a valid GGUF file " + \
                    f"(magic number mismatch, got {GGUF_MAGIC}, " + \
                    "expected b'GGUF')"
                )

            if GGUF_VERSION == 1:
                raise ValueError(
                    "easy_llama: your model file reports GGUF version 1, " + \
                    "but only versions 2 and above are supported. " + \
                    "re-convert your model or download a newer version"
                )

            for _ in range(kv_data_count):
                key_length = struct.unpack("<Q", file.read(8))[0]
                key = file.read(key_length)

                value_type = GGUFReader.GGUFValueType(
                    struct.unpack("<I", file.read(4))[0]
                )
                if value_type == GGUFReader.GGUFValueType.ARRAY:
                    ltype = GGUFReader.GGUFValueType(
                        struct.unpack("<I", file.read(4))[0]
                    )
                    length = struct.unpack("<Q", file.read(8))[0]
                    for _ in range(length):
                        _ = GGUFReader.get_single(self, ltype, file)
                else:
                    value = GGUFReader.get_single(self, value_type, file)
                    metadata[key.decode()] = value

        return metadata