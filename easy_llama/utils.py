# utils.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""Submodule containing convenience functions used throughout easy_llama"""

from . import __version__

import os
import sys
import datetime
import contextlib

import numpy as np

from typing import Iterable, TextIO, TypeVar, Generic, NoReturn, Literal

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

NoneType = type(None)

_VERBOSE = True
"""Package-wide verbose flag"""

_DEBUG = False
"""Package-wide debug flag"""

class _ArrayLike(Iterable):
    """Anything that can be interpreted as a numpy array"""

class _SupportsWriteAndFlush(TextIO):
    """A file, stream, or buffer that supports writing and flushing"""

class LlamaNullException(Exception):
    """Raised when a libllama function returns NULL or NULLPTR"""

T = TypeVar('T')

class ptr(Generic[T]):
    """Generic type hint representing any ctypes pointer. Optionally subscriptable with any
    type."""

def set_verbose(state: bool) -> None:
    """Enable or disable verbose terminal output from easy-llama"""
    global _VERBOSE
    _VERBOSE = state

def get_verbose() -> bool:
    """Return `True` if verbose output is enabled in easy-llama, `False` otherwise. If debug
    output is enabled, this will always `True`."""
    global _VERBOSE, _DEBUG
    return _VERBOSE or _DEBUG # force-enable verbose output if debug output is enabled

def set_debug(state: bool) -> None:
    """Enable or disable debug output from easy-llama"""
    global _DEBUG
    _DEBUG = state

def get_debug() -> bool:
    """Return `True` if debug output is enabled in easy-llama, `False` otherwise"""
    global _DEBUG
    return _DEBUG

def log(
    text: str,
    level: Literal[1, 2, 3, 4] = 1,
    disable: bool = False
) -> None:
    """Print the given text, prefixed with a timestamp"""
    if disable:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %a %H:%M:%S.%f")[:-3]
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

def log_verbose(text: str, level: Literal[1,2,3,4] = 1) -> None:
    if get_verbose():
        log(text, level)

def log_debug(text: str, level: Literal[1,2,3,4] = 1) -> None:
    if _DEBUG:
        log('[DEBUG] ' + text, level)

@contextlib.contextmanager
def KeyboardInterruptHandler():
    log_verbose('Press CTRL+C to exit')
    try:
        yield
    except KeyboardInterrupt:
        print(ANSI.MODE_RESET_ALL, end='\n', flush=True)

def softmax(z: _ArrayLike, T: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over **all** elements of an arbitrarily shaped array in
    float32 precision. Supports temperature scaling for all real values of `T`."""
    z_arr = np.array(z, dtype=np.float32)
    if z_arr.size == 0:
        return z_arr
    if T == 0.0:
        result = np.zeros_like(z_arr)
        flat = z_arr.ravel()
        result.flat[np.argmax(flat)] = 1.0
        return result
    if T < 0:
        z_arr = -z_arr
        T = -T
    max_val = np.max(z_arr)
    scaled = (z_arr - max_val) / T
    exp_vals = np.exp(scaled)
    return exp_vals / np.sum(exp_vals)

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
    """Encode the given text `txt` from string to UTF-8. If strict encoding fails, an error
    will be shown and the offending character(s) will be replaced with `?`."""
    try:
        return txt.encode('utf-8', errors='strict')
    except UnicodeEncodeError:
        log(f'error encoding string to UTF-8. using ? replacement character.', level=3)
        return txt.encode('utf-8', errors='replace')

def ez_decode(txt: bytes) -> str:
    """Decode the given text `txt` from UTF-8 to string. If strict decoding fails, an error
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

def null_ptr_check(obj: ptr, ptr_name: str, loc_hint: str) -> None | NoReturn:
    """Ensure that the given object `obj` is not NULL / NULLPTR. Raise `LlamaNullException` on
    failure.

    - obj:
        The object to check (should be a ctypes pointer of some kind)
    - ptr_name:
        The name of the object, used for error message
    - loc_hint:
        Code location hint, used for error message"""
    if not bool(obj):
        raise LlamaNullException(f"{loc_hint}: pointer `{ptr_name}` is null")

def exc_to_str(exc: Exception) -> str:
    """Return `'ExceptionType: Exception message'`"""
    return f"{type(exc).__name__}: {exc}"
