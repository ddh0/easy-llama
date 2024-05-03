# utils.py
# https://github.com/ddh0/easy-llama/

"""Submodule containing convenience functions and GGUFReader"""

import sys

from time     import strftime
from enum     import IntEnum
from struct   import unpack
from colorama import Fore
from typing   import Any


# color codes used in Thread.interact()
RESET_ALL = Fore.RESET
USER_STYLE = RESET_ALL + Fore.GREEN
BOT_STYLE = RESET_ALL + Fore.CYAN
DIM_STYLE = RESET_ALL + Fore.LIGHTBLACK_EX
SPECIAL_STYLE = RESET_ALL + Fore.YELLOW

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

    def get_single(self, value_type, file) -> Any:
        if value_type == GGUFReader.GGUFValueType.STRING:
            value_length = unpack("<Q", file.read(8))[0]
            value = file.read(value_length)
            value = value.decode("utf-8")
        else:
            type_str = GGUFReader._simple_value_packing.get(value_type)
            bytes_length = GGUFReader.value_type_info.get(value_type)
            value = unpack(type_str, file.read(bytes_length))[0]

        return value

    def load_metadata(self, fname) -> dict:
        metadata = {}
        with open(fname, "rb") as file:
            GGUF_MAGIC = file.read(4)

            if GGUF_MAGIC != b"GGUF":
                raise ValueError(
                    "your model file is not a valid GGUF file " + \
                    f"(magic number mismatch, got {GGUF_MAGIC}, " + \
                    "expected b'GGUF')"
                )

            GGUF_VERSION = unpack("<I", file.read(4))[0]

            if GGUF_VERSION == 1:
                raise ValueError(
                    "your model file reports GGUF version 1, " + \
                    "but only versions 2 and above are supported. " + \
                    "re-convert your model or download a newer version"
                )

            # ti_data_count = struct.unpack("<Q", file.read(8))[0]
            file.read(8)
            kv_data_count = unpack("<Q", file.read(8))[0]

            for _ in range(kv_data_count):
                key_length = unpack("<Q", file.read(8))[0]
                key = file.read(key_length)

                value_type = GGUFReader.GGUFValueType(
                    unpack("<I", file.read(4))[0]
                )
                if value_type == GGUFReader.GGUFValueType.ARRAY:
                    ltype = GGUFReader.GGUFValueType(
                        unpack("<I", file.read(4))[0]
                    )
                    length = unpack("<Q", file.read(8))[0]
                    arr = [GGUFReader.get_single(self, ltype, file) for _ in range(length)]
                    metadata[key.decode()] = arr
                else:
                    value = GGUFReader.get_single(self, value_type, file)
                    metadata[key.decode()] = value

        return metadata

def cls() -> None:
    """Clear the terminal"""
    print("\033c\033[3J", end='', flush=True)

# no longer used in this module, but left for others to use
def get_timestamp_prefix_str() -> str:
    # helpful: https://strftime.net
    return strftime("[%Y, %b %e, %a %l:%M %p] ")

def truncate(text: str) -> str:
    return text if len(text) < 53 else f"{text[:50]}..."

def print_verbose(text: str) -> None:
    print("easy_llama: verbose:", text, file=sys.stderr, flush=True)

def print_info(text: str) -> None:
    print("easy_llama: info:", text, file=sys.stderr, flush=True)

def print_warning(text: str) -> None:
    print("easy_llama: warning:", text, file=sys.stderr, flush=True)
