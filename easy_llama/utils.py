# utils.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""Submodule containing convenience functions used throughout easy_llama"""

from ._version import __version__

import os
import sys
import datetime
import contextlib

import numpy as np

from typing import Iterable, TextIO, Optional, TypeVar, Generic, NoReturn, Literal

class ANSI:
    """ANSI codes for terminal emulators"""

    BELL = '\a'

    CLEAR = '\x1bc\x1b[3J' # technically this is two ANSI codes in one

    # Standard colors
    FG_BLACK   = '\x1b[30m'
    BG_BLACK   = '\x1b[40m'
    FG_RED     = '\x1b[31m'
    BG_RED     = '\x1b[41m'
    FG_GREEN   = '\x1b[32m'
    BG_GREEN   = '\x1b[42m'
    FG_YELLOW  = '\x1b[33m'
    BG_YELLOW  = '\x1b[43m'
    FG_BLUE    = '\x1b[34m'
    BG_BLUE    = '\x1b[44m'
    FG_MAGENTA = '\x1b[35m'
    BG_MAGENTA = '\x1b[45m'
    FG_CYAN    = '\x1b[36m'
    BG_CYAN    = '\x1b[46m'
    FG_WHITE   = '\x1b[37m'
    BG_WHITE   = '\x1b[47m'

    # Bright colors
    FG_BRIGHT_BLACK   = '\x1b[90m'
    BG_BRIGHT_BLACK   = '\x1b[100m'
    FG_BRIGHT_RED     = '\x1b[91m'
    BG_BRIGHT_RED     = '\x1b[101m'
    FG_BRIGHT_GREEN   = '\x1b[92m'
    BG_BRIGHT_GREEN   = '\x1b[102m'
    FG_BRIGHT_YELLOW  = '\x1b[93m'
    BG_BRIGHT_YELLOW  = '\x1b[103m'
    FG_BRIGHT_BLUE    = '\x1b[94m'
    BG_BRIGHT_BLUE    = '\x1b[104m'
    FG_BRIGHT_MAGENTA = '\x1b[95m'
    BG_BRIGHT_MAGENTA = '\x1b[105m'
    FG_BRIGHT_CYAN    = '\x1b[96m'
    BG_BRIGHT_CYAN    = '\x1b[106m'
    FG_BRIGHT_WHITE   = '\x1b[97m'
    BG_BRIGHT_WHITE   = '\x1b[107m'

    # Modes
    MODE_RESET_ALL           = '\x1b[0m'
    MODE_BOLD_SET            = '\x1b[1m'
    MODE_BOLD_RESET          = '\x1b[22m'
    MODE_DIM_SET             = '\x1b[2m'
    MODE_DIM_RESET           = '\x1b[22m'
    MODE_ITALIC_SET          = '\x1b[3m'
    MODE_ITALIC_RESET        = '\x1b[23m'
    MODE_UNDERLINE_SET       = '\x1b[4m'
    MODE_UNDERLINE_RESET     = '\x1b[24m'
    MODE_BLINKING_SET        = '\x1b[5m'
    MODE_BLINKING_RESET      = '\x1b[25m'
    MODE_REVERSE_SET         = '\x1b[7m'
    MODE_REVERSE_RESET       = '\x1b[27m'
    MODE_HIDDEN_SET          = '\x1b[8m'
    MODE_HIDDEN_RESET        = '\x1b[28m'
    MODE_STRIKETHROUGH_SET   = '\x1b[9m'
    MODE_STRIKETHROUGH_RESET = '\x1b[29m'

NoneType: type = type(None)

class _ArrayLike(Iterable):
    """Anything that can be interpreted as a numpy array"""

class _SupportsWriteAndFlush(TextIO):
    """A file, stream, or buffer that supports writing and flushing"""

class UnreachableException(Exception):
    """The code has reached an unreachable state"""
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
    """Generic type hint representing any ctypes pointer. Optionally subscriptable with any
    type."""

def log(
    text: str,
    level: Literal[1, 2, 3, 4] = 1,
    disable: bool = False
) -> None:
    """Print the given text to the file, prefixed with a timestamp"""
    if disable:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %a %k:%M:%S.%f")[:-3]
    if level == 1:
        lvltxt = f"{ANSI.FG_BRIGHT_GREEN}INFO"
    elif level == 2:
        lvltxt = f"{ANSI.FG_BRIGHT_YELLOW}WARNING"
    elif level == 3:
        lvltxt = f"{ANSI.FG_BRIGHT_RED}ERROR"
    elif level == 4:
        lvltxt = f"{ANSI.FG_BRIGHT_MAGENTA}STOPWATCH"
    else:
        raise ValueError(f'parameter `level` must be one of `[1, 2, 3, 4]`, not {level}')
    print(
        f"{ANSI.MODE_RESET_ALL}{ANSI.MODE_BOLD_SET}{ANSI.FG_BRIGHT_BLACK}[{timestamp}]"
        f"{ANSI.MODE_RESET_ALL}{ANSI.MODE_BOLD_SET} {lvltxt}{ANSI.MODE_RESET_ALL}"
        f"{ANSI.MODE_BOLD_SET}{ANSI.FG_BRIGHT_BLACK}:{ANSI.MODE_RESET_ALL} {text}",
        end='\n',
        file=sys.stdout if level in [1,4] else sys.stderr,
        flush=True
    )

def softmax(
    z: _ArrayLike, T: Optional[float] = None, dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """Compute softmax over values in z, where z is array-like.
    Also apply temperature `T`, if specified.

    Any floating-point value for temperature `T` is valid, including 0.0 and
    negative numbers.

    If `dtype` is not specified, `np.float32` is used if available."""
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
        #os.system('clear')
        print(ANSI.CLEAR, end='', flush=True)

def truncate(text: str) -> str:
    return text if len(text) < 72 else f"{text[:69]}..."

def ez_encode(txt: str) -> bytes:
    """Encode the given text `txt` from string to UTF-8. If strict encoding fails, a warning
    will be shown and the offending character(s) will be replaced with `?`."""
    try:
        return txt.encode('utf-8', errors='strict')
    except UnicodeEncodeError:
        log(f'error encoding string to UTF-8. using ? replacement character.', level=3)
        return txt.encode('utf-8', errors='replace')

def ez_decode(txt: bytes) -> str:
    """Decode the given text `txt` from UTF-8 to string. If strict decoding fails, a warning
    will be shown and the offending character(s) will be replaced with `�` (U+FFFD)."""
    try:
        return txt.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        log(f'error decoding string from UTF-8. using � replacement character.', level=3)
        return txt.decode('utf-8', errors='replace')

_open = open
_sys = sys
_os = os

@contextlib.contextmanager
def suppress_output(disable: bool = False):
    """Suppress stdout and stderr."""

    # NOTE: simply changing sys.stdout and sys.stderr does not affect output from llama.cpp.
    #       this method (or similar) is required to suppress all undesired output, for example
    #       when `verbose=False`.

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

def assert_type(
    obj: object,
    expected_type: type | tuple[type],
    obj_name: str,
    code_location: str,
    hint: Optional[str] = None
):
    """Ensure that `obj` is an instance of `expected_type`.

    If `expected_type` is a tuple, ensure that `obj` is an instance of some type in the tuple.

    Raise `TypeError` otherwise, using `obj_name` and `code_location` to make an informative
    exception message.

    If specified, `hint` is added as a note to the exception."""

    if isinstance(obj, expected_type):
        return
    
    obj_type_repr = repr(type(obj).__name__)

    if not isinstance(expected_type, tuple):
        # represent `int` as 'int' instead of "<class 'int'>"
        expected_type_repr = repr(expected_type.__name__)
        exc = TypeError(
            f"{code_location}: {obj_name} should be an instance of {expected_type_repr}, "
            f"not {obj_type_repr}"
        )
    else:
        # represent `(int, list)` as "('int', 'list')" instead of
        # "(<class 'int'>, <class 'list'>)"
        expected_type_repr = repr(tuple(t.__name__ for t in expected_type))
        exc = TypeError(
            f"{code_location}: {obj_name} should be one of {expected_type_repr}, "
            f"not {obj_type_repr}"
        )
    if isinstance(hint, str):
        exc.add_note(hint)
    raise exc

def assert_only_ints(iterable: Iterable) -> None | NoReturn:
    """Ensure that the given iterable contains only `int`s"""
    if any(not isinstance(x, int) for x in iterable):
        raise TypeError(f"assert_only_ints: some item in the given iterable is not an int")

def null_ptr_check(ptr: ptr, ptr_name: str, loc_hint: str) -> None | NoReturn:
    """Ensure that the given object `ptr` is not NULL / NULLPTR

    Raise LlamaNullException on failure

    - ptr:
        The object to check
    - ptr_name:
        The name of the object (for error messages)
    - loc_hint:
        Code location hint used in easy-llama"""
    if not bool(ptr):
        raise LlamaNullException(f"{loc_hint}: pointer `{ptr_name}` is null")
