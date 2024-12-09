# constants.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from enum import IntEnum

MAX_DEFAULT_CONTEXT_LENGTH = 8192

class Colors:
    """
    ANSI codes to set text foreground color in terminal output
    """
    RESET  = '\x1b[39m'
    GREEN  = '\x1b[39m\x1b[32m'
    BLUE   = '\x1b[39m\x1b[36m'
    GREY   = '\x1b[39m\x1b[90m'
    YELLOW = '\x1b[39m\x1b[33m'
    RED    = '\x1b[39m\x1b[91m'

# these values are from llama.cpp/gguf-py/gguf/constants.py
class GGUFValueType(IntEnum):
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