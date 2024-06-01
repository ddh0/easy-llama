# utils.py
# https://github.com/ddh0/easy-llama/
from ._version import __version__, __llama_cpp_version__

"""Submodule containing convenience functions and GGUFReader"""

import os
import sys
import struct
import numpy as np

from typing   import Any, Iterable, TextIO, Optional, Union
from io       import BufferedReader
from time     import strftime
from enum     import IntEnum
from colorama import Fore


# color codes used in Thread.interact()
RESET_ALL = Fore.RESET
USER_STYLE = Fore.RESET + Fore.GREEN
BOT_STYLE = Fore.RESET + Fore.CYAN
DIM_STYLE = Fore.RESET + Fore.LIGHTBLACK_EX
SPECIAL_STYLE = Fore.RESET + Fore.YELLOW

class TypeAssertionFailedError(Exception):
    """`assert_type()` failed"""

# for typing of softmax parameter `z`
class _ArrayLike(Iterable):
    pass

# for typing of Model.stream_print() parameter `file`
class _SupportsWriteAndFlush(TextIO):
    pass

def softmax(z: _ArrayLike) -> np.ndarray:
    """
    Compute softmax over values in z, where z is array-like
    """
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()

def cls() -> None:
    """Clear the terminal"""
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
        print("\033c\033[3J", end='', flush=True)

# not used by default, but useful as a feature of an AdvancedFormat
def get_timestamp_str() -> str:
    # helpful: https://strftime.net
    return strftime("%Y, %b %e, %a %l:%M %p")

def truncate(text: str) -> str:
    return text if len(text) < 72 else f"{text[:69]}..."

def print_verbose(text: str) -> None:
    print("easy_llama: verbose:", text, file=sys.stderr, flush=True)

def print_info(text: str) -> None:
    print("easy_llama: info:", text, file=sys.stderr, flush=True)

def print_warning(text: str) -> None:
    print("easy_llama: warning:", text, file=sys.stderr, flush=True)

def assert_type(
    something: Any,
    expected_type: Union[type, tuple[type]],
    something_repr: str,
    code_location: str,
    hint: Optional[str] = None
):
    """
    Ensure that `something` is an instance of `expected_type`

    If `expected_type` is a tuple, ensure that `something` is an instance of
    some type in the tuple

    Raise `TypeAssertionFailedError` otherwise, using `something_repr` and
    `code_location` to make an informative exception message

    If specified, `hint` is added as a note to the exception
    """
    if isinstance(something, expected_type):
        return
    
    # represent `int` as 'int' instead of "<class 'int'>"
    type_something_repr = repr(type(something).__name__)

    if not isinstance(expected_type, tuple):
        expected_type_repr = repr(expected_type.__name__)
        exc = TypeAssertionFailedError(
            f"{code_location}: {something_repr} should be an instance of "
            f"{expected_type_repr}, not {type_something_repr}"
        )

        if hint is not None:
            exc.add_note(hint)
        
        raise exc
    
    # represent `(int, list)` as "('int', 'list')" instead of
    # "(<class 'int'>, <class 'list'>)"
    expected_type_repr = repr(tuple(t.__name__ for t in expected_type))
    exc = TypeAssertionFailedError(
        f"{code_location}: {something_repr} should be one of "
        f"{expected_type_repr}, not {type_something_repr}"
    )

    if hint is not None:
        exc.add_note(hint)

    raise exc

class GGUFValueType(IntEnum):
    # Occasionally check to ensure this class is consistent with gguf
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

class QuickGGUFReader:
    # NOTE: Officially, there is no way to determine if a GGUF file is little
    #       or big endian. The format specifcation directs us to assume that
    #       a file is little endian in all cases unless additional info is
    #       provided.
    #
    #       In addition to this, GGUF files cannot run on hosts with the
    #       opposite endianness. And, at this point in the code, the model
    #       is already loaded. Therefore, we can assume that the endianness
    #       of the file is the same as the endianness of the host.
    #
    # ref: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    if sys.byteorder == 'little':
        # LITTLE-endian format for struct.unpack() based on gguf value type
        value_packing: dict = {
            GGUFValueType.UINT8:   "<B",
            GGUFValueType.INT8:    "<b",
            GGUFValueType.UINT16:  "<H",
            GGUFValueType.INT16:   "<h",
            GGUFValueType.UINT32:  "<I",
            GGUFValueType.INT32:   "<i",
            GGUFValueType.FLOAT32: "<f",
            GGUFValueType.UINT64:  "<Q",
            GGUFValueType.INT64:   "<q",
            GGUFValueType.FLOAT64: "<d",
            GGUFValueType.BOOL:    "?"
        }
    else:
        # BIG-endian format for struct.unpack() based on gguf value type
        value_packing: dict = {
            GGUFValueType.UINT8:   ">B",
            GGUFValueType.INT8:    ">b",
            GGUFValueType.UINT16:  ">H",
            GGUFValueType.INT16:   ">h",
            GGUFValueType.UINT32:  ">I",
            GGUFValueType.INT32:   ">i",
            GGUFValueType.FLOAT32: ">f",
            GGUFValueType.UINT64:  ">Q",
            GGUFValueType.INT64:   ">q",
            GGUFValueType.FLOAT64: ">d",
            GGUFValueType.BOOL:    "?"
        }

    # length in bytes for each gguf value type
    value_lengths: dict = {
        GGUFValueType.UINT8:   1,
        GGUFValueType.INT8:    1,
        GGUFValueType.UINT16:  2,
        GGUFValueType.INT16:   2,
        GGUFValueType.UINT32:  4,
        GGUFValueType.INT32:   4,
        GGUFValueType.FLOAT32: 4,
        GGUFValueType.UINT64:  8,
        GGUFValueType.INT64:   8,
        GGUFValueType.FLOAT64: 8,
        GGUFValueType.BOOL:    1
    }

    @staticmethod
    def unpack(value_type: GGUFValueType, file: BufferedReader):
        return struct.unpack(
            QuickGGUFReader.value_packing.get(value_type),
            file.read(QuickGGUFReader.value_lengths.get(value_type))
        )[0]

    @staticmethod
    def get_single(
            value_type: GGUFValueType,
            file: BufferedReader
        ) -> Union[str, int, float, bool]:
        """Read a single value from an open file"""

        if value_type == GGUFValueType.STRING:
            value_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            value = file.read(value_length)
            value = value.decode("utf-8")
        else:
            value = QuickGGUFReader.unpack(value_type, file=file)
        return value
    
    @staticmethod
    def load_metadata(
            fn: Union[os.PathLike[str], str]
        ) -> dict[str, Union[str, int, float, bool, list]]:
        """
        Given a path to a GGUF file, peek at its header for metadata

        Return a dictionary where all keys are strings, and values can be
        strings, ints, floats, bools, or lists
        """

        metadata: dict[str, Union[str, int, float, bool, list]] = {}
        with open(fn, "rb") as file:
            GGUF_MAGIC = file.read(4)

            if GGUF_MAGIC != b"GGUF":
                raise ValueError(
                    "your model file is not a valid GGUF file "
                    f"(magic number mismatch, got {GGUF_MAGIC}, "
                    "expected b'GGUF')"
                )
            
            GGUF_VERSION = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)

            if GGUF_VERSION == 1:
                raise ValueError(
                    "your model file reports GGUF version 1, "
                    "but only versions 2 and above are supported. "
                    "re-convert your model or download a newer version"
                )
            
            # number of tensors in file - not used here
            ti_data_count = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            kv_data_count = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)

            for _ in range(kv_data_count):
                key_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
                key = file.read(key_length)
                value_type = GGUFValueType(
                    QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                )
                if value_type == GGUFValueType.ARRAY:
                    ltype = GGUFValueType(
                        QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                    )
                    length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
                    array = [
                        QuickGGUFReader.get_single(
                            ltype,
                            file
                        ) for _ in range(length)
                    ]
                    metadata[key.decode()] = array
                else:
                    value = QuickGGUFReader.get_single(value_type, file)
                    metadata[key.decode()] = value

        return metadata
