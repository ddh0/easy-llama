# utils.py
# https://github.com/ddh0/easy-llama/

"""Submodule containing convenience functions and GGUFReader"""

import os
import sys
import struct
import numpy as np

from typing   import Iterable, TextIO, Optional
from io       import BufferedReader
from enum     import IntEnum


# color codes used in Thread.interact()
RESET_ALL     = '\x1b[39m'
USER_STYLE    = '\x1b[39m\x1b[32m'
BOT_STYLE     = '\x1b[39m\x1b[36m'
DIM_STYLE     = '\x1b[39m\x1b[90m'
SPECIAL_STYLE = '\x1b[39m\x1b[33m'
ERROR_STYLE   = '\x1b[39m\x1b[91m'

NoneType: type = type(None)

class TypeAssertionError(Exception):
    """`assert_type()` failed"""

# for typing of softmax parameter `z`
class _ArrayLike(Iterable):
    pass

# for typing of Model.stream_print() parameter `file`
class _SupportsWriteAndFlush(TextIO):
    pass

def softmax(
    z: _ArrayLike,
    T: Optional[float] = None,
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    Compute softmax over values in z, where z is array-like.
    Also apply temperature `T`, if specified.

    Any floating-point value for temperature `T` is valid, including 0.0 and
    negative numbers.

    If `dtype` is not specified, the highest precision
    numpy `dtype` available will be used.
    """
    if dtype is None:
        if hasattr(np, 'float128'):
            _dtype = np.float128
        elif hasattr(np, 'float96'):
            _dtype = np.float96
        elif hasattr(np, 'float80'):
            _dtype = np.float80
        elif hasattr(np, 'float64'):
            _dtype = np.float64
        elif hasattr(np, 'float32'):
            _dtype = np.float32
        elif hasattr(np, 'float16'):
            _dtype = np.float16
        else:
            _dtype = float
    else:
        assert_type(
            dtype,
            type,
            'dtype',
            'softmax',
            'dtype should be a numpy floating type, such as `np.float32`'
        )
        _dtype = dtype
    
    _z = np.asarray(z, dtype=_dtype)
    if T is None or T == 1.0:
        # simple formula with no temperature
        e_z = np.exp(_z - np.max(_z), dtype=_dtype)
        return e_z / np.sum(e_z, axis=0, dtype=_dtype)
    assert_type(T, float, "temperature value 'T'", 'softmax')
    if T == 0.0:
        # Return an array where the maximum value in _z is 1.0 and all others are 0.0
        max_index = np.argmax(_z)
        result = np.zeros_like(_z, dtype=_dtype)
        result[max_index] = 1.0
        return result
    e_z = np.exp(np.divide(_z, T, dtype=_dtype), dtype=_dtype)
    return e_z / np.sum(e_z, axis=0, dtype=_dtype)

def cls() -> None:
    """Clear the terminal"""
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
        print("\033c\033[3J", end='', flush=True)

def truncate(text: str) -> str:
    return text if len(text) < 72 else f"{text[:69]}..."

def print_verbose(text: str) -> None:
    print("easy_llama: verbose:", text, file=sys.stderr, flush=True)

def print_info(text: str) -> None:
    print("easy_llama: info:", text, file=sys.stderr, flush=True)

def print_warning(text: str) -> None:
    print("easy_llama: warning:", text, file=sys.stderr, flush=True)

def _print_debug(
        obj: object,
        prefix: str='\t',
        file: _SupportsWriteAndFlush = sys.stderr
) -> None:
    print(f"{prefix}{type(obj)=}", file=file)
    print(f"{prefix}{repr(obj)=}", file=file)
    print(f"{prefix}{id(obj)=}", file=file)
    print(f"{prefix}{hex(id(obj))=}", file=file)
    print(f"{prefix}{sys.getsizeof(obj)=}", file=file)
    file.flush()

def assert_type(
    obj: object,
    expected_type: type | tuple[type],
    obj_name: str,
    code_location: str,
    hint: Optional[str] = None
):
    """
    Ensure that `obj` is an instance of `expected_type`

    If `expected_type` is a tuple, ensure that `obj` is an instance of
    some type in the tuple

    Raise `TypeAssertionError` otherwise, using `obj_name` and
    `code_location` to make an informative exception message

    If specified, `hint` is added as a note to the exception
    """
    if isinstance(obj, expected_type):
        return
    
    obj_type_repr = repr(type(obj).__name__)

    if not isinstance(expected_type, tuple):
        # represent `int` as 'int' instead of "<class 'int'>"
        expected_type_repr = repr(expected_type.__name__)
        exc = TypeAssertionError(
            f"{code_location}: {obj_name} should be an instance of "
            f"{expected_type_repr}, not {obj_type_repr}"
        )
    else:
        # represent `(int, list)` as "('int', 'list')" instead of
        # "(<class 'int'>, <class 'list'>)"
        expected_type_repr = repr(tuple(t.__name__ for t in expected_type))
        exc = TypeAssertionError(
            f"{code_location}: {obj_name} should be one of "
            f"{expected_type_repr}, not {obj_type_repr}"
        )
    if isinstance(hint, str):
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
    # ref: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

    # the GGUF format versions that this class supports
    SUPPORTED_GGUF_VERSIONS = [2, 3]

    # GGUF only supports execution on little or big endian machines
    if sys.byteorder not in ['little', 'big']:
        raise ValueError(
            "host is not little or big endian - GGUF is unsupported"
        )
    
    # arguments for struct.unpack() based on gguf value type
    value_packing: dict = {
        GGUFValueType.UINT8   : "=B",
        GGUFValueType.INT8    : "=b",
        GGUFValueType.UINT16  : "=H",
        GGUFValueType.INT16   : "=h",
        GGUFValueType.UINT32  : "=I",
        GGUFValueType.INT32   : "=i",
        GGUFValueType.FLOAT32 : "=f",
        GGUFValueType.UINT64  : "=Q",
        GGUFValueType.INT64   : "=q",
        GGUFValueType.FLOAT64 : "=d",
        GGUFValueType.BOOL    : "?"
    }

    # length in bytes for each gguf value type
    value_lengths: dict = {
        GGUFValueType.UINT8   : 1,
        GGUFValueType.INT8    : 1,
        GGUFValueType.UINT16  : 2,
        GGUFValueType.INT16   : 2,
        GGUFValueType.UINT32  : 4,
        GGUFValueType.INT32   : 4,
        GGUFValueType.FLOAT32 : 4,
        GGUFValueType.UINT64  : 8,
        GGUFValueType.INT64   : 8,
        GGUFValueType.FLOAT64 : 8,
        GGUFValueType.BOOL    : 1
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
        ) -> str | int | float | bool:
        """Read a single value from an open file"""
        if value_type == GGUFValueType.STRING:
            string_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            value = file.read(string_length)
            # officially, strings that cannot be decoded into utf-8 are invalid
            value = value.decode("utf-8")
        else:
            value = QuickGGUFReader.unpack(value_type, file=file)
        return value
    
    @staticmethod
    def load_metadata(
            fn: os.PathLike[str] | str
        ) -> dict[str, str | int | float | bool | list]:
        """
        Given a path to a GGUF file, peek at its header for metadata

        Return a dictionary where all keys are strings, and values can be
        strings, ints, floats, bools, or lists
        """

        metadata: dict[str, str | int | float | bool | list] = {}
        with open(fn, "rb") as file:
            magic = file.read(4)

            if magic != b'GGUF':
                raise ValueError(
                    "your model file is not a valid GGUF file "
                    f"(magic number mismatch, got {magic}, "
                    "expected b'GGUF')"
                )
            
            version = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)

            if version not in QuickGGUFReader.SUPPORTED_GGUF_VERSIONS:
                raise ValueError(
                    f"your model file reports GGUF version {version}, but "
                    f"only versions {QuickGGUFReader.SUPPORTED_GGUF_VERSIONS} "
                    "are supported. re-convert your model or download a newer "
                    "version"
                )
            
            tensor_count = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            if version == 3:
                metadata_kv_count = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            elif version == 2:
                metadata_kv_count = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)

            for _ in range(metadata_kv_count):
                if version == 3:
                    key_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
                elif version == 2:
                    key_length = 0
                    while key_length == 0:
                        # seek until next key is found
                        key_length = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                    file.read(4) # 4 byte offset for GGUFv2
                key = file.read(key_length)
                value_type = GGUFValueType(
                    QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                )
                if value_type == GGUFValueType.ARRAY:
                    array_value_type = GGUFValueType(
                        QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                    )
                    # array_length is the number of items in the array
                    if version == 3:
                        array_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
                    elif version == 2:
                        array_length = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                        file.read(4) # 4 byte offset for GGUFv2
                    array = [
                        QuickGGUFReader.get_single(
                            array_value_type,
                            file
                        ) for _ in range(array_length)
                    ]
                    metadata[key.decode()] = array
                else:
                    value = QuickGGUFReader.get_single(
                        value_type,
                        file
                    )
                    metadata[key.decode()] = value

        return metadata
