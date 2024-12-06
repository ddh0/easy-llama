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

class LlamaVocabType:
    NONE = 0
    SPM = 1
    BPE = 2
    WPM = 3
    UGM = 4
    RWKV = 5

class LlamaRopeType:
    NONE = -1
    NORM = 0
    NEOX = 2

class LlamaTokenType:
    UNDEFINED = 0
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6

class LlamaTokenAttr:
    UNDEFINED = 0
    UNKNOWN = 1 << 0
    UNUSED = 1 << 1
    NORMAL = 1 << 2
    CONTROL = 1 << 3
    USER_DEFINED = 1 << 4
    BYTE = 1 << 5
    NORMALIZED = 1 << 6
    LSTRIP = 1 << 7
    RSTRIP = 1 << 8
    SINGLE_WORD = 1 << 9

class LlamaAttentionType:
    UNSPECIFIED = -1
    CAUSAL = 0
    NON_CAUSAL = 1

class LlamaSplitMode:
    NONE = 0
    LAYER = 1
    ROW = 2

class LlamaFtype:
    ALL_F32 = 0
    MOSTLY_F16 = 1
    MOSTLY_Q4_0 = 2
    MOSTLY_Q4_1 = 3
    MOSTLY_Q8_0 = 7
    MOSTLY_Q5_0 = 8
    MOSTLY_Q5_1 = 9
    MOSTLY_Q2_K = 10
    MOSTLY_Q3_K_S = 11
    MOSTLY_Q3_K_M = 12
    MOSTLY_Q3_K_L = 13
    MOSTLY_Q4_K_S = 14
    MOSTLY_Q4_K_M = 15
    MOSTLY_Q5_K_S = 16
    MOSTLY_Q5_K_M = 17
    MOSTLY_Q6_K = 18
    MOSTLY_IQ2_XXS = 19
    MOSTLY_IQ2_XS = 20
    MOSTLY_Q2_K_S = 21
    MOSTLY_IQ3_XS = 22
    MOSTLY_IQ3_XXS = 23
    MOSTLY_IQ1_S = 24
    MOSTLY_IQ4_NL = 25
    MOSTLY_IQ3_S = 26
    MOSTLY_IQ3_M = 27
    MOSTLY_IQ2_S = 28
    MOSTLY_IQ2_M = 29
    MOSTLY_IQ4_XS = 30
    MOSTLY_IQ1_M = 31
    MOSTLY_BF16 = 32
    MOSTLY_Q4_0_4_4 = 33
    MOSTLY_Q4_0_4_8 = 34
    MOSTLY_Q4_0_8_8 = 35
    MOSTLY_TQ1_0 = 36
    MOSTLY_TQ2_0 = 37
    GUESSED = 1024

class LlamaRopeScalingType:
    UNSPECIFIED = -1
    NONE = 0
    LINEAR = 1
    YARN = 2
    MAX_VALUE = YARN

class LlamaPoolingType:
    UNSPECIFIED = -1
    NONE = 0
    MEAN = 1
    CLS = 2
    LAST = 3
    RANK = 4

class LlamaModelKVOverrideType:
    INT = 0
    FLOAT = 1
    BOOL = 2
    STR = 3

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