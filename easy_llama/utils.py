# utils.py
# https://github.com/ddh0/easy-llama/
from ._version import __version__, __llama_cpp_version__

"""Submodule containing convenience functions and QuickGGUFReader"""

import os
import sys
import struct
import numpy as np

from typing    import Iterable, TextIO, Optional
from io        import BufferedReader
from enum      import IntEnum


# color codes used in Thread.interact()
RESET_ALL     = '\x1b[39m'
USER_STYLE    = '\x1b[39m\x1b[32m' # green
BOT_STYLE     = '\x1b[39m\x1b[36m' # blue
DIM_STYLE     = '\x1b[39m\x1b[90m' # grey
SPECIAL_STYLE = '\x1b[39m\x1b[33m' # yellow
ERROR_STYLE   = '\x1b[39m\x1b[91m' # red

NoneType: type = type(None)

class TypeAssertionError(Exception):
    "A call to `assert_type()` has failed"

class _ArrayLike(Iterable):
    "Anything that can be interpreted as a numpy array"

class _SupportsWriteAndFlush(TextIO):
    "A file, stream, or buffer that supports writing and flushing"

class UnreachableException(Exception):
    "The code has reached an unreachable state"
    def __init__(self):
        super().__init__(
            "the code has reached a location that was thought to be "
            "unreachable. please report this issue to the developer at this "
            "link: https://github.com/ddh0/easy-llama/issues/new/choose"
        )

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

    If `dtype` is not specified, the highest precision numpy `dtype` available
    will be used.
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
            'dtype should be a floating type, such as `np.float32`'
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

def print_version_info(file: _SupportsWriteAndFlush = sys.stderr) -> None:
    print(f"easy_llama package version: {__version__}", file=file)
    print(f"llama_cpp package version: {__llama_cpp_version__}", file=file)

def print_verbose(text: str) -> None:
    print("easy_llama:", text, file=sys.stderr, flush=True)

def print_info(text: str) -> None:
    print("easy_llama: info:", text, file=sys.stderr, flush=True)

def print_warning(text: str) -> None:
    print(
        f"{RESET_ALL}easy_llama: {SPECIAL_STYLE}WARNING{RESET_ALL}:", text,
        file=sys.stderr, flush=True
    )

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

def assert_only_ints(iterable: Iterable) -> None:
    """
    Ensure that the given iterable contains only `int`s
    """
    if any(not isinstance(x, int) for x in iterable):
        raise TypeAssertionError(
            f"assert_only_ints: some item in the given iterable is not an int"
        )


class InferenceLock:
    """
    A context manager that can be used to prevent the model from accepting more
    than one generation at a time, which is not supported and can cause a hard
    crash.

    This is mostly useful in asychronous / multi-threaded contexts
    """

    class LockFailure(Exception):
        pass

    def __init__(self):
        self.locked = False

    def __enter__(self):
        return self.acquire()
    
    def __exit__(self, *_):
        return self.release()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *_):
       return self.__exit__()
    
    def acquire(self):
        if self.locked:
            raise self.LockFailure(
                'failed to acquire InferenceLock (already locked)'
            )
        self.locked = True
        return self
    
    def release(self):
        if not self.locked:
            raise self.LockFailure(
                'tried to release InferenceLock that is not acquired'
            )
        self.locked = False


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
