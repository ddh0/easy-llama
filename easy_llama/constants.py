# constants.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from enum import IntEnum

# "official" constants
LLAMA_DEFAULT_SEED = 0xFFFFFFFF
LLAMA_TOKEN_NULL = -1
LLAMA_FILE_MAGIC_GGLA = 0x67676C61
LLAMA_FILE_MAGIC_GGSN = 0x6767736E
LLAMA_FILE_MAGIC_GGSQ = 0x67677371
LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
LLAMA_SESSION_VERSION = 9
LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ
LLAMA_STATE_SEQ_VERSION = 2

# "unofficial" constants added by ddh0
MAX_LAYERS = 0x7FFFFFFF # maximum int32 value used for n_gpu_layers
GGUF_MAGIC = b'GGUF'
MAX_DEFAULT_CONTEXT_LENGTH = 8192

# enums and dataclasses

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