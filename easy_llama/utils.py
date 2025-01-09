# utils.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from _version import __version__

"""Submodule containing convenience functions and QuickGGUFReader"""

import os
import sys
import contextlib
import numpy as np

from typing import Iterable, TextIO, Optional, TypeVar, Generic, NoReturn

class Colors:
    """
    ANSI codes to set text foreground color in terminal output
    """
    RESET   = '\x1b[39m'
    GREEN   = '\x1b[39m\x1b[32m'
    BLUE    = '\x1b[39m\x1b[36m'
    GREY    = '\x1b[39m\x1b[90m'
    YELLOW  = '\x1b[39m\x1b[33m'
    RED     = '\x1b[39m\x1b[91m'
    MAGENTA = '\x1b[39m\x1b[35m'

# color codes used in print_info, print_warning, print_error, and
# Thread.interact()
RESET   = RESET_ALL     = Colors.RESET
GREEN   = USER_STYLE    = Colors.GREEN
BLUE    = BOT_STYLE     = Colors.BLUE
GREY    = DIM_STYLE     = Colors.GREY
YELLOW  = SPECIAL_STYLE = Colors.YELLOW
RED     = ERROR_STYLE   = Colors.RED
MAGENTA = TIMER_STYLE   = Colors.MAGENTA

NoneType: type = type(None)

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

class LlamaNullException(Exception):
    """Raised when a libllama function returns NULL or NULLPTR"""

T = TypeVar('T')

class ptr(Generic[T]):
    """
    Generic type hint representing any ctypes pointer
    
    Optionally subscriptable with any type
    """
        
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

    If `dtype` is not specified, `np.float32` is used if available.
    """
    if dtype is None:
        if hasattr(np, 'float32'):
            _dtype = np.float32
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

_open = open
_sys = sys
_os = os

@contextlib.contextmanager
def suppress_output(disable: bool = False):
    if disable:
        yield
    else:
        # save the original file descriptors
        original_stdout_fd = _sys.stdout.fileno()
        original_stderr_fd = _sys.stderr.fileno()

        saved_stdout_fd = _os.dup(original_stdout_fd)
        saved_stderr_fd = _os.dup(original_stderr_fd)

        with _open(_os.devnull, 'wb') as devnull:
            devnull_fd = devnull.fileno()

            _os.dup2(devnull_fd, original_stdout_fd)
            _os.dup2(devnull_fd, original_stderr_fd)

            try:
                yield
            finally:
                # restore the original file descriptors
                _os.dup2(saved_stdout_fd, original_stdout_fd)
                _os.dup2(saved_stderr_fd, original_stderr_fd)

                _os.close(saved_stdout_fd)
                _os.close(saved_stderr_fd)

def print_verbose(text: str) -> None:
    print(
        f"easy_llama:",
        text, file=sys.stderr, flush=True
    )

def print_info(text: str) -> None:
    print(
        f"{RESET}easy_llama: {GREEN}INFO{RESET}:",
        text, file=sys.stderr, flush=True
    )

def print_warning(text: str) -> None:
    print(
        f"{RESET}easy_llama: {YELLOW}WARNING{RESET}:",
        text, file=sys.stderr, flush=True
    )

def print_error(text: str) -> None:
    print(
        f"{RESET}easy_llama: {RED}ERROR{RESET}:",
        text, file=sys.stderr, flush=True
    )

def print_stopwatch(text: str) -> None:
    print(
        f"{RESET}easy_llama: {MAGENTA}STOPWATCH{RESET}:",
        text, file=sys.stderr, flush=True
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
        exc = TypeError(
            f"{code_location}: {obj_name} should be an instance of "
            f"{expected_type_repr}, not {obj_type_repr}"
        )
    else:
        # represent `(int, list)` as "('int', 'list')" instead of
        # "(<class 'int'>, <class 'list'>)"
        expected_type_repr = repr(tuple(t.__name__ for t in expected_type))
        exc = TypeError(
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
        raise TypeError(
            f"assert_only_ints: some item in the given iterable is not an int"
        )

def null_ptr_check(
    ptr: ptr, ptr_name: str, loc_hint: str
) -> None | NoReturn:
    """
    Ensure that the given object `ptr` is not NULL / NULLPTR

    Raise LlamaNullException on failure

    - ptr:
        The object to check
    - ptr_name:
        The name of the object (for error messages)
    - loc_hint:
        Code location hint used in easy-llama
    """
    if not bool(ptr):
        raise LlamaNullException(
            f"{loc_hint}: pointer `{ptr_name}` is null"
        )
