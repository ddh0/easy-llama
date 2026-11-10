# libllama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay
# -------------------------------------------------------------------------------------------------
# Manually update these constants to reflect the currently targeted version of llama.cpp:
# -------------------------------------------------------------------------------------------------
#
# llama.cpp commit OID
_TARGET_LLAMACPP_COMMIT = "e95426f16982054c379ca7bdc8186996dd22f0fe"
#
# git timestamp of the above commit
_TARGET_LLAMACPP_TIME   = "Sun Nov 8 00:20:58 2026 +0800"
#
# -------------------------------------------------------------------------------------------------

f"""
This file provides the low-level Python interface to LLAMA_API ("libllama"), which is originally defined
in `llama.cpp/include/llama.h`.

This file was last updated to match `llama.cpp/include/llama.h` as of this commit:
    - commit OID: `{_TARGET_LLAMACPP_COMMIT}`
    - commit timestamp: `{_TARGET_LLAMACPP_TIME}`

This file's status with respect to the above commit is:
    - Synchronized (pending testing)

Helpful references:
    - `libllama` API changelog:
    [llama.cpp/issues/9289](https://github.com/ggml-org/llama.cpp/issues/9289)
    - `llama.h` at master:
    [llama.cpp/blob/master/include/llama.h](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)
"""

from . import __version__

import os
import sys
import ctypes
import faulthandler

import numpy as np

from enum   import IntEnum
from typing import Optional, Callable
from .utils import ptr, log, str_decode, log_verbose

#
# Type hints and other constants
#

NULL = None
NULLPTR = ctypes.c_void_p(NULL)

C_FALSE = ctypes.c_int8(0)
C_TRUE = ctypes.c_int8(1)

MAX_TOKEN_LENGTH = 1024 # update this if you find a token that is > 1024 bytes
"""The maximum supported length of a single token's text, in bytes"""

# maximum value for int32. NOTE: newer llama.cpp versions treat any negative value
# for n_gpu_layers as "all layers" (so -1 is now the preferred way to offload
# everything); this constant is kept for backwards compatibility.
MAX_OFFLOAD_LAYERS = 0x7FFFFFFF

# the minimum value for int32. `llama_tokenize` returns this on overflow (see the
# `llama.h` docs); used by the helpers below to tell an overflow apart from a
# normal "buffer too small" negative return.
_INT32_MIN = -0x80000000

# keep state for backend
_BACKEND_INIT = False

# =================================================================================================
#
# Start of LLAMA_API
#
# =================================================================================================

#
# Stuff from llama.cpp/ggml/include/ggml.h
#

GGML_FILE_MAGIC   = 0x67676d6c # "ggml"
GGML_FILE_VERSION = 2

GGML_EXIT_SUCCESS = 0
GGML_EXIT_ABORTED = 1

GGML_ROPE_TYPE_NEOX   = 2
GGML_ROPE_TYPE_MROPE  = 8
GGML_ROPE_TYPE_VISION = 24
GGML_ROPE_TYPE_IMROPE = 40

GGUF_MAGIC = 0x46554747 # "GGUF"
GGUF_MAGIC_BYTES = b'GGUF'

GGUF_VERSION = 3

GGUF_DEFAULT_ALIGNMENT = 32

class GGMLType(IntEnum):
    GGML_TYPE_F32     = 0
    GGML_TYPE_F16     = 1
    GGML_TYPE_Q4_0    = 2
    GGML_TYPE_Q4_1    = 3
    # GGML_TYPE_Q4_2 = 4 -- support has been removed
    # GGML_TYPE_Q4_3 = 5 -- support has been removed
    GGML_TYPE_Q5_0    = 6
    GGML_TYPE_Q5_1    = 7
    GGML_TYPE_Q8_0    = 8
    GGML_TYPE_Q8_1    = 9
    GGML_TYPE_Q2_K    = 10
    GGML_TYPE_Q3_K    = 11
    GGML_TYPE_Q4_K    = 12
    GGML_TYPE_Q5_K    = 13
    GGML_TYPE_Q6_K    = 14
    GGML_TYPE_Q8_K    = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS  = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S   = 19
    GGML_TYPE_IQ4_NL  = 20
    GGML_TYPE_IQ3_S   = 21
    GGML_TYPE_IQ2_S   = 22
    GGML_TYPE_IQ4_XS  = 23
    GGML_TYPE_I8      = 24
    GGML_TYPE_I16     = 25
    GGML_TYPE_I32     = 26
    GGML_TYPE_I64     = 27
    GGML_TYPE_F64     = 28
    GGML_TYPE_IQ1_M   = 29
    GGML_TYPE_BF16    = 30
    # GGML_TYPE_Q4_0_4_4 = 31 -- support has been removed from gguf files
    # GGML_TYPE_Q4_0_4_8 = 32
    # GGML_TYPE_Q4_0_8_8 = 33
    GGML_TYPE_TQ1_0   = 34
    GGML_TYPE_TQ2_0   = 35
    # GGML_TYPE_IQ4_NL_4_4 = 36
    # GGML_TYPE_IQ4_NL_4_8 = 37
    # GGML_TYPE_IQ4_NL_8_8 = 38
    GGML_TYPE_MXFP4   = 39 # MXFP4 (1 block)
    GGML_TYPE_NVFP4   = 40 # NVFP4 (4 blocks, E4M3 scale)
    GGML_TYPE_Q1_0    = 41
    GGML_TYPE_Q2_0    = 42
    GGML_TYPE_F8_E4M3 = 43
    GGML_TYPE_COUNT   = 44

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

#
# GGML pointer types
#

ggml_backend_buffer_type_t = ctypes.c_void_p # struct ggml_backend_buffer_type *
ggml_backend_buffer_t      = ctypes.c_void_p # struct ggml_backend_buffer *
ggml_backend_dev_t         = ctypes.c_void_p # struct ggml_backend_device *
ggml_context_p             = ctypes.c_void_p # struct ggml_context *
ggml_cgraph_p              = ctypes.c_void_p # struct ggml_cgraph *
ggml_tensor_p              = ctypes.c_void_p # struct ggml_tensor *
ggml_threadpool_t          = ctypes.c_void_p # opaque

#
# Callback function types from ggml.h / llama.h
#

# bool (*)(float progress, void * user_data)
progress_callback_functype = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)

# bool (*)(const struct ggml_tensor * dst, const struct ggml_cgraph * finfo, void * user_data)
eval_callback_functype = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
)

# bool (*)(void * data)
abort_callback_functype = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

# void (*)(struct ggml_tensor * tensor, void * userdata)
model_set_tensor_data_functype = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)

# void (*)(enum ggml_log_level level, const char * text, void * user_data)
log_callback_functype = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

#
# Constants
#

LLAMA_DEFAULT_SEED = 0xFFFFFFFF

LLAMA_TOKEN_NULL = -1

LLAMA_FILE_MAGIC_GGLA = 0x67676c61 # 'ggla'
LLAMA_FILE_MAGIC_GGSN = 0x6767736e # 'ggsn'
LLAMA_FILE_MAGIC_GGSQ = 0x67677371 # 'ggsq'

LLAMA_SESSION_MAGIC   = LLAMA_FILE_MAGIC_GGSN
LLAMA_SESSION_VERSION = 11

LLAMA_STATE_SEQ_MAGIC   = LLAMA_FILE_MAGIC_GGSQ
LLAMA_STATE_SEQ_VERSION = 4

# maximum number of distinct DRY sequence breakers, duplicates not counted
LLAMA_DRY_MAX_SEQ_BREAKERS = 64

#
# State sequence save/load flags
#

LLAMA_STATE_SEQ_FLAGS_NONE = 0

# work only with partial states, e.g. SWA cache or recurrent cache part of the memory
LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY = 1

# keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster
# save/load). getting the state for a seq_id with this flag invalidates all prior states gotten
# for that seq_id with this flag.
LLAMA_STATE_SEQ_FLAGS_ON_DEVICE = 2

llama_state_seq_flags = ctypes.c_uint32

#
# Opaque pointer types
#

class gguf_context(ctypes.Structure):
    pass

gguf_context_p = ctypes.POINTER(gguf_context)

# Python has no native `FILE *`, so callers must obtain one from libc and pass it here as a void
# pointer, f.ex:
#
#     libc = ctypes.CDLL(None)
#     libc.fdopen.restype = ctypes.c_void_p
#     libc.fdopen.argtypes = [ctypes.c_int, ctypes.c_char_p]
#     file = libc.fdopen(os.open(path, os.O_RDONLY), b'rb')
#
# TODO: wrap this in a helper function?
FILE_p = ctypes.c_void_p

class llama_vocab(ctypes.Structure):
    pass

llama_vocab_p = ctypes.POINTER(llama_vocab)

class llama_model(ctypes.Structure):
    pass

llama_model_p = ctypes.POINTER(llama_model)

class llama_context(ctypes.Structure):
    pass

llama_context_p = ctypes.POINTER(llama_context)

class llama_sampler(ctypes.Structure):
    pass

llama_sampler_p = ctypes.POINTER(llama_sampler)

class llama_memory_i(ctypes.Structure):
    pass

llama_memory_i_p = ctypes.POINTER(llama_memory_i)

llama_memory_t = llama_memory_i_p

class llama_batch_ext(ctypes.Structure):
    pass

llama_batch_ext_p = ctypes.POINTER(llama_batch_ext)

llama_pos    = ctypes.c_int32 # typedefs
llama_token  = ctypes.c_int32
llama_seq_id = ctypes.c_int32

#
# Enums
#

class LlamaVocabType(IntEnum):
    LLAMA_VOCAB_TYPE_NONE   = 0 # For models without vocab
    LLAMA_VOCAB_TYPE_SPM    = 1 # LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE    = 2 # GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM    = 3 # BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM    = 4 # T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV   = 5 # RWKV tokenizer based on greedy tokenization
    LLAMA_VOCAB_TYPE_PLAMO2 = 6 # PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    LLAMA_VOCAB_TYPE_TEST   = 7 # Dummy tokenizer for testing: rolling hash of fixed-size chunks
                                # -> tokens, tokens -> hex
    LLAMA_VOCAB_TYPE_PLAMO3 = 8 # PLaMo-3 tokenizer with pre-segmentation and dynamic programming

class LlamaRopeType(IntEnum):
    LLAMA_ROPE_TYPE_NONE   = -1
    LLAMA_ROPE_TYPE_NORM   = 0
    LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX
    LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE
    LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE
    LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION

class LlamaTokenType(IntEnum):
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0
    LLAMA_TOKEN_TYPE_NORMAL       = 1
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2
    LLAMA_TOKEN_TYPE_CONTROL      = 3
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4
    LLAMA_TOKEN_TYPE_UNUSED       = 5
    LLAMA_TOKEN_TYPE_BYTE         = 6

class LlamaTokenAttr(IntEnum):
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9

# model file types
class LlamaFType(IntEnum):
    LLAMA_FTYPE_ALL_F32              = 0
    LLAMA_FTYPE_MOSTLY_F16           = 1  # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_0          = 2  # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_1          = 3  # except 1d tensors
    # LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4 -- support has been removed
    # LLAMA_FTYPE_MOSTLY_Q4_2          = 5 -- support has been removed
    # LLAMA_FTYPE_MOSTLY_Q4_3          = 6 -- support has been removed
    LLAMA_FTYPE_MOSTLY_Q8_0          = 7  # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_0          = 8  # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_1          = 9  # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K          = 10 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q6_K          = 18 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ1_S         = 24 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_S         = 26 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_M         = 27 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_S         = 28 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_M         = 29 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ1_M         = 31 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_BF16          = 32 # except 1d tensors
    # LLAMA_FTYPE_MOSTLY_Q4_0_4_4 = 33 -- removed from gguf files, use Q4_0 and runtime repack
    # LLAMA_FTYPE_MOSTLY_Q4_0_4_8 = 34 -- removed from gguf files, use Q4_0 and runtime repack
    # LLAMA_FTYPE_MOSTLY_Q4_0_8_8 = 35 -- removed from gguf files, use Q4_0 and runtime repack
    LLAMA_FTYPE_MOSTLY_TQ1_0         = 36 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_TQ2_0         = 37 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_NVFP4         = 39 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q1_0          = 40 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_0          = 41 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_F8_E4M3       = 42 # except 1d tensors

    LLAMA_FTYPE_GUESSED = 1024 # not specified in the model file

def _cstr(s: Optional[bytes]) -> str: # helper
    """Decode a `char *` returned by libllama. Returns '' if the pointer was NULL."""
    return s.decode('utf-8') if s else ''

def llama_ftype_name(ftype: int) -> str:
    """Get the model file type (quantization) as a string, e.g. "Q8_0" or "Q4_K - Medium\""""
    libllama.llama_ftype_name.argtypes = [ctypes.c_int]
    libllama.llama_ftype_name.restype = ctypes.c_char_p
    return _cstr(libllama.llama_ftype_name(ftype))

class LlamaRopeScalingType(IntEnum):
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    LLAMA_ROPE_SCALING_TYPE_NONE        = 0
    LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1
    LLAMA_ROPE_SCALING_TYPE_YARN        = 2
    LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE

class LlamaPoolingType(IntEnum):
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    LLAMA_POOLING_TYPE_NONE        = 0
    LLAMA_POOLING_TYPE_MEAN        = 1
    LLAMA_POOLING_TYPE_CLS         = 2
    LLAMA_POOLING_TYPE_LAST        = 3
    LLAMA_POOLING_TYPE_RANK        = 4

class LlamaAttentionType(IntEnum):
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    LLAMA_ATTENTION_TYPE_CAUSAL      = 0
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1

class LlamaFlashAttnType(IntEnum):
    LLAMA_FLASH_ATTN_TYPE_AUTO     = -1
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
    LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1

def llama_flash_attn_type_name(flash_attn_type: int) -> str:
    """Get the name of a flash attention type"""
    libllama.llama_flash_attn_type_name.argtypes = [ctypes.c_int]
    libllama.llama_flash_attn_type_name.restype = ctypes.c_char_p
    return _cstr(libllama.llama_flash_attn_type_name(flash_attn_type))

class LlamaSplitMode(IntEnum):
    LLAMA_SPLIT_MODE_NONE   = 0 # single GPU
    LLAMA_SPLIT_MODE_LAYER  = 1 # split layers and KV across GPUs
    LLAMA_SPLIT_MODE_ROW    = 2 # split layers and KV across GPUs, use tensor parallelism if supported
    LLAMA_SPLIT_MODE_TENSOR = 3

class LlamaLoadMode(IntEnum):
    LLAMA_LOAD_MODE_AUTO       = -1 # auto-detect based on device capabilities
    LLAMA_LOAD_MODE_NONE       = 0  # no special loading mode
    LLAMA_LOAD_MODE_MMAP       = 1  # memory map the model
    LLAMA_LOAD_MODE_MLOCK      = 2  # force system to keep model in RAM rather than swapping or compressing
    LLAMA_LOAD_MODE_MMAP_MLOCK = 3 # mmap + force system to keep model in RAM rather than swapping or compressing
    LLAMA_LOAD_MODE_DIRECT_IO  = 4  # use direct I/O if available

def llama_load_mode_name(load_mode: int) -> str:
    """Get the name of a load mode"""
    libllama.llama_load_mode_name.argtypes = [ctypes.c_int]
    libllama.llama_load_mode_name.restype = ctypes.c_char_p
    return _cstr(libllama.llama_load_mode_name(load_mode))

def llama_load_mode_from_str(s: str) -> int:
    """Get a load mode from its name. See `LlamaLoadMode`."""
    libllama.llama_load_mode_from_str.argtypes = [ctypes.c_char_p]
    libllama.llama_load_mode_from_str.restype = ctypes.c_int
    return libllama.llama_load_mode_from_str(s.encode('utf-8'))

class LlamaLazyMode(IntEnum):
    LLAMA_LAZY_MODE_OFF  = 0 # always read the whole tensor up front
    LLAMA_LAZY_MODE_AUTO = 1 # lazy only for marked tensors larger than 4 GiB (requires mmap)
    LLAMA_LAZY_MODE_ON   = 2 # read the rows of tensors marked by the arch on demand (requires mmap)

class LlamaContextType(IntEnum):
    LLAMA_CONTEXT_TYPE_DEFAULT = 0
    LLAMA_CONTEXT_TYPE_MTP     = 1

class LlamaModelMetaKey(IntEnum):
    LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE         = 0
    LLAMA_MODEL_META_KEY_SAMPLING_TOP_K            = 1
    LLAMA_MODEL_META_KEY_SAMPLING_TOP_P            = 2
    LLAMA_MODEL_META_KEY_SAMPLING_MIN_P            = 3
    LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY  = 4
    LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD    = 5
    LLAMA_MODEL_META_KEY_SAMPLING_TEMP             = 6
    LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N   = 7
    LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT   = 8
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT         = 9
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU     = 10
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA     = 11

def llama_model_meta_key_str(key: int) -> str:
    """Get the name of a sampling metadata key. Returns '' if the key is invalid."""
    libllama.llama_model_meta_key_str.argtypes = [ctypes.c_int]
    libllama.llama_model_meta_key_str.restype = ctypes.c_char_p
    return _cstr(libllama.llama_model_meta_key_str(key))

class LlamaProcessType(IntEnum):
    LLAMA_PROCESS_TYPE_ENCODE = 0
    LLAMA_PROCESS_TYPE_DECODE = 1

#
# Structs
#

class llama_token_data(ctypes.Structure):
    _fields_ = [
        ("id",    ctypes.c_int32), # token id
        ("logit", ctypes.c_float), # log-odds of the token
        ("p",     ctypes.c_float)  # probability of the token
    ]

llama_token_data_p = ctypes.POINTER(llama_token_data)

class llama_token_data_array(ctypes.Structure):
    _fields_ = [
        ("data",     ctypes.POINTER(llama_token_data)), # NOTE: this pointer can be modified by the samplers
        ("size",     ctypes.c_size_t                 ),
        ("selected", ctypes.c_int64                  ), # this is the index in the data array (i.e. not the token id)
        ("sorted",   ctypes.c_bool                   )  # NOTE: do not assume the data is sorted - always check this flag
    ]

llama_token_data_array_p = ctypes.POINTER(llama_token_data_array)

class llama_batch(ctypes.Structure):
    _fields_ = [
        # size of the batch
        ("n_tokens", ctypes.c_int32                              ),
        # the token ids of the input (used when embd is NULL)
        ("token",    ctypes.POINTER(llama_token)                 ),
        # token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
        ("embd",     ctypes.POINTER(ctypes.c_float)              ),
        # the positions of the respective token in the sequence
        # (if NULL, the token position is tracked automatically by llama_encode/llama_decode)
        ("pos",      ctypes.POINTER(llama_pos)                   ),
        # the number of sequences the respective token belongs to
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)              ),
        # the sequences to which the respective token belongs
        # (if NULL, the sequence ID is assumed to be 0)
        ("seq_id",   ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        # if C_FALSE: the logits (and/or the embeddings) for the respective token will not be output
        # if NULL:
        #     - if embd: all tokens are output
        #     - if not:  only the last token is output
        ("logits",   ctypes.POINTER(ctypes.c_int8)               ) # use C_FALSE / C_TRUE
    ]

llama_batch_p = ctypes.POINTER(llama_batch)

class llama_model_kv_override_val_type(ctypes.Union):
    _fields_ = [
        ("val_i64",  ctypes.c_int64     ),
        ("val_f64",  ctypes.c_double    ),
        ("val_bool", ctypes.c_bool      ),
        ("val_str",  ctypes.c_char * 128)
    ]

llama_model_kv_override_val_type_p = ctypes.POINTER(llama_model_kv_override_val_type)

class llama_model_kv_override(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_int                    ), # llama_model_kv_override_type
        ("key", ctypes.c_char * 128             ),
        ("val", llama_model_kv_override_val_type)
    ]
    _anonymous_ = ("val",) # make the union accessible anonymously, as in C

llama_model_kv_override_p = ctypes.POINTER(llama_model_kv_override)

class llama_model_kv_override_type(IntEnum):
    """Tag values for `llama_model_kv_override.tag`."""
    LLAMA_KV_OVERRIDE_TYPE_INT   = 0
    LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1
    LLAMA_KV_OVERRIDE_TYPE_BOOL  = 2
    LLAMA_KV_OVERRIDE_TYPE_STR   = 3

class llama_model_tensor_buft_override(ctypes.Structure):
    _fields_ = [
        ("pattern", ctypes.c_char_p            ), # regex pattern to match tensor names
        ("buft",    ggml_backend_buffer_type_t )  # opaque pointer to buffer type
    ]

llama_model_tensor_buft_override_p = ctypes.POINTER(llama_model_tensor_buft_override)

class llama_model_params(ctypes.Structure):
    # NOTE: These fields need to be aligned with struct llama_model_params {...} !!!
    #       If they are not aligned, you might get crashes that are very hard to debug !!!
    _fields_ = [
        # NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ("devices", ctypes.POINTER(ctypes.c_void_p)),

        # NULL-terminated list of buffer types to use for tensors that match a pattern
        ("tensor_buft_overrides", ctypes.POINTER(llama_model_tensor_buft_override)),

        ("n_gpu_layers", ctypes.c_int32), # number of layers to store in VRAM, a negative value means all layers
        ("split_mode",   ctypes.c_int  ), # LlamaSplitMode: how to split the model across multiple GPUs
        ("load_mode",    ctypes.c_int  ), # LlamaLoadMode: how to load the model
        ("lazy_mode",    ctypes.c_int  ), # LlamaLazyMode: on-demand reading of tensors marked by the arch

        # the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        ("main_gpu", ctypes.c_int32),

        # proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),

        # Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        # If the provided progress_callback returns true, model loading continues.
        # If it returns false, model loading is immediately aborted.
        ("progress_callback", progress_callback_functype),

        # context pointer passed to the progress callback
        ("progress_callback_user_data", ctypes.c_void_p),

        # override key-value pairs of the model meta data
        ("kv_overrides", ctypes.POINTER(llama_model_kv_override)),

        # Keep the booleans together to avoid misalignment during copy-by-value.
        ("vocab_only",      ctypes.c_bool), # only load the vocabulary, no weights
        ("check_tensors",   ctypes.c_bool), # validate model tensor data
        ("use_extra_bufts", ctypes.c_bool), # use extra buffer types (used for weight repacking)
        ("no_host",         ctypes.c_bool), # bypass host buffer allowing extra buffers to be used
        ("no_alloc",        ctypes.c_bool), # only load metadata and simulate memory allocations
        ("load_mtp",        ctypes.c_bool)  # whether to load MTP layers
    ]

llama_model_params_p = ctypes.POINTER(llama_model_params)

#
# Sampler interface (custom samplers)
#

llama_sampler_context_t = ctypes.c_void_p

# [EXPERIMENTAL] backend sampling data passed to llama_sampler_i.backend_apply
class llama_sampler_data(ctypes.Structure):
    _fields_ = [
        ("logits",     ggml_tensor_p), # struct ggml_tensor *
        ("probs",      ggml_tensor_p), # struct ggml_tensor *
        ("sampled",    ggml_tensor_p), # struct ggml_tensor *
        ("candidates", ggml_tensor_p)  # struct ggml_tensor *
    ]

llama_sampler_data_p = ctypes.POINTER(llama_sampler_data)

#
# llama_sampler_i (sampler interface) function pointer types
#
# NOTE: probably never need to use these externally unless you know what you're doing

# can be NULL
sampler_name_functype = ctypes.CFUNCTYPE(ctypes.c_char_p, llama_sampler_p)

# can be NULL
sampler_accept_functype = ctypes.CFUNCTYPE(None, llama_sampler_p, llama_token)

# REQUIRED
sampler_apply_functype = ctypes.CFUNCTYPE(None, llama_sampler_p, llama_token_data_array_p)

# can be NULL
sampler_reset_functype = ctypes.CFUNCTYPE(None, llama_sampler_p)

# can be NULL if ctx is NULL
sampler_clone_functype = ctypes.CFUNCTYPE(llama_sampler_p, llama_sampler_p)

# can be NULL if ctx is NULL
sampler_free_functype = ctypes.CFUNCTYPE(None, llama_sampler_p)

sampler_backend_init_functype = ctypes.CFUNCTYPE(
    ctypes.c_bool, llama_sampler_p, ggml_backend_buffer_type_t, ctypes.c_uint32
)

sampler_backend_accept_functype = ctypes.CFUNCTYPE(
    None, llama_sampler_p, ggml_context_p, ggml_cgraph_p, ggml_tensor_p
)

sampler_backend_apply_functype = ctypes.CFUNCTYPE(
    None, llama_sampler_p, ggml_context_p, ggml_cgraph_p, llama_sampler_data_p
)

sampler_backend_set_input_functype = ctypes.CFUNCTYPE(None, llama_sampler_p)

sampler_backend_reset_functype = ctypes.CFUNCTYPE(None, llama_sampler_p)

sampler_copy_state_functype = ctypes.CFUNCTYPE(None, llama_sampler_p, llama_sampler_p)

class llama_sampler_i(ctypes.Structure):
    _fields_ = [
        # classic (host) sampling interface
        ("name",              sampler_name_functype             ), # can be NULL
        ("accept",            sampler_accept_functype           ), # can be NULL
        ("apply",             sampler_apply_functype            ), # REQUIRED
        ("reset",             sampler_reset_functype            ), # can be NULL
        ("clone",             sampler_clone_functype            ), # can be NULL if ctx is NULL
        ("free",              sampler_free_functype             ), # can be NULL if ctx is NULL
        # [EXPERIMENTAL] backend sampling interface
        ("backend_init",      sampler_backend_init_functype     ),
        ("backend_accept",    sampler_backend_accept_functype   ),
        ("backend_apply",     sampler_backend_apply_functype    ),
        ("backend_set_input", sampler_backend_set_input_functype),
        ("backend_reset",     sampler_backend_reset_functype    ),
        ("copy_state",        sampler_copy_state_functype       )
    ]

llama_sampler_i_p = ctypes.POINTER(llama_sampler_i)

llama_sampler._fields_ = [ # NOTE: opaque until this point
    ("iface", llama_sampler_i_p      ),
    ("ctx",   llama_sampler_context_t)
]

class llama_sampler_seq_config(ctypes.Structure):
    _fields_ = [
        ("seq_id",   llama_seq_id    ), # the sequence this sampler is bound to
        ("sampler",  llama_sampler_p )  # NOTE: must be a sampler chain (llama_sampler_chain_init)
    ]

llama_sampler_seq_config_p = ctypes.POINTER(llama_sampler_seq_config)

class llama_context_params(ctypes.Structure):
    # NOTE: these fields need to be aligned with struct llama_context_params {...} !!!
    #       if they are not aligned, you might get crashes that are very hard to debug !!!
    _fields_ = [
        ("n_ctx",                 ctypes.c_uint32), # text context, 0 = from model
        ("n_batch",               ctypes.c_uint32), # logical maximum batch size that can be submitted to llama_decode
        ("n_ubatch",              ctypes.c_uint32), # physical maximum batch size (micro-batch size)
        ("n_seq_max",             ctypes.c_uint32), # max number of sequences (i.e. distinct states for recurrent models)
        ("n_rs_seq",              ctypes.c_uint32), # recurrent-state snapshots per seq for rollback, 0 = no rollback [EXPERIMENTAL]
        ("n_outputs_max",         ctypes.c_uint32), # max outputs in a ubatch (0 = n_batch)
        ("n_outputs_max_per_seq", ctypes.c_uint32), # max outputs per sequence (0 = n_outputs_max)
        ("n_threads",             ctypes.c_int32 ), # number of threads to use for generation
        ("n_threads_batch",       ctypes.c_int32 ), # number of threads to use for batch processing

        ("ctx_type",          ctypes.c_int), # LlamaContextType: set the context type (e.g. MTP)
        ("rope_scaling_type", ctypes.c_int), # LlamaRopeScalingType: RoPE scaling type
        ("pooling_type",      ctypes.c_int), # LlamaPoolingType: whether to pool (sum) embedding results by sequence id
        ("attention_type",    ctypes.c_int), # LlamaAttentionType: attention type to use for embeddings
        ("flash_attn_type",   ctypes.c_int), # LlamaFlashAttnType: when to enable Flash Attention

        # ref: https://github.com/ggml-org/llama.cpp/pull/2054
        ("rope_freq_base",   ctypes.c_float ), # RoPE base frequency, 0 = from model
        ("rope_freq_scale",  ctypes.c_float ), # RoPE frequency scaling factor, 0 = from model
        ("yarn_ext_factor",  ctypes.c_float ), # YaRN extrapolation mix factor, negative = from model
        ("yarn_attn_factor", ctypes.c_float ), # YaRN magnitude scaling factor
        ("yarn_beta_fast",   ctypes.c_float ), # YaRN low correction dim
        ("yarn_beta_slow",   ctypes.c_float ), # YaRN high correction dim
        ("yarn_orig_ctx",    ctypes.c_uint32), # YaRN original context size
        ("defrag_thold",     ctypes.c_float ), # [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ("cb_eval",           eval_callback_functype), # callback for eval
        ("cb_eval_user_data", ctypes.c_void_p       ), # user data for eval callback

        ("type_k", ctypes.c_int), # data type for K cache [EXPERIMENTAL] (GGMLType)
        ("type_v", ctypes.c_int), # data type for V cache [EXPERIMENTAL] (GGMLType)

        ("moe_cache_size",      ctypes.c_size_t), # device cache in bytes for the experts kept in host memory, split among the devices like the layers, 0 = disabled [EXPERIMENTAL]
        ("moe_cache_cpu_share", ctypes.c_float ), # share of the MoE cache misses computed on the CPU instead of uploaded, 0 = all uploaded [EXPERIMENTAL]

        # if the abort callback returns true, execution of llama_decode() will be aborted
        # currently works only with CPU execution
        ("abort_callback",      abort_callback_functype), # callback for abort
        ("abort_callback_data", ctypes.c_void_p        ), # user data for abort callback

        # keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        ("embeddings",  ctypes.c_bool), # if true, extract embeddings (together with logits, except for a pooling type other than NONE)
        ("offload_kqv", ctypes.c_bool), # whether to offload the KQV ops (including the KV cache) to GPU
        ("no_perf",     ctypes.c_bool), # whether to measure performance timings
        ("op_offload",  ctypes.c_bool), # whether to offload host tensor operations to device
        ("swa_full",    ctypes.c_bool), # use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                                        # NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                                        #       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
        ("kv_unified",  ctypes.c_bool), # use a unified buffer across the input sequences when computing the attention
                                        # try to disable when n_seq_max > 1 for improved performance when the sequences
                                        # do not share a large prefix (ref: https://github.com/ggml-org/llama.cpp/pull/14363)

        # [EXPERIMENTAL] backend sampler chain configuration (make sure the caller keeps the
        # sampler chains alive). note: the samplers must be sampler chains (i.e. use
        # llama_sampler_chain_init)
        ("samplers",    llama_sampler_seq_config_p),
        ("n_samplers",  ctypes.c_size_t           ),

        # source/target/parent context
        ("ctx_other",   llama_context_p)
    ]

llama_context_params_p = ctypes.POINTER(llama_context_params)

class llama_model_tensor_override(ctypes.Structure):
    _fields_ = [
        ("pattern", ctypes.c_char_p), # regex pattern to match tensor names
        ("type",    ctypes.c_int   )  # GGMLType
    ]

llama_model_tensor_override_p = ctypes.POINTER(llama_model_tensor_override)

class llama_model_imatrix_data(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p               ), # tensor name
        ("data", ctypes.POINTER(ctypes.c_float)), # imatrix data
        ("size", ctypes.c_size_t               )  # number of floats in data
    ]

llama_model_imatrix_data_p = ctypes.POINTER(llama_model_imatrix_data)

class llama_model_quantize_params(ctypes.Structure):
    _fields_ = [
        ("nthread",                ctypes.c_int32), # number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        ("ftype",                  ctypes.c_int  ), # LlamaFType: quantize to this llama_ftype
        ("output_tensor_type",     ctypes.c_int  ), # GGMLType: output tensor type
        ("token_embedding_type",   ctypes.c_int  ), # GGMLType: token embeddings tensor type
        ("allow_requantize",       ctypes.c_bool ), # allow quantizing non-f32/f16 tensors
        ("quantize_output_tensor", ctypes.c_bool ), # quantize output.weight
        ("only_copy",              ctypes.c_bool ), # only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        ("pure",                   ctypes.c_bool ), # quantize all tensors to the default type
        ("keep_split",             ctypes.c_bool ), # quantize to the same number of shards
        ("dry_run",                ctypes.c_bool ), # calculate and show the final quantization size without performing quantization
        ("imatrix",                llama_model_imatrix_data_p     ), # pointer to importance matrix data
        ("kv_overrides",           llama_model_kv_override_p      ), # pointer to kv overrides
        ("tt_overrides",           llama_model_tensor_override_p  ), # pointer to tensor overrides
        ("prune_layers",           ctypes.POINTER(ctypes.c_int32) ), # pointer to layer indices to prune
        ("max_buf_size",           ctypes.c_size_t                )  # max bytes of tensor rows kept in memory at once, 0 = default (8 GiB)
    ]

llama_model_quantize_params_p = ctypes.POINTER(llama_model_quantize_params)

class llama_logit_bias(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_int32),
        ("bias",  ctypes.c_float)
    ]

llama_logit_bias_p = ctypes.POINTER(llama_logit_bias)

class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool) # whether to measure performance timings
    ]

llama_sampler_chain_params_p = ctypes.POINTER(llama_sampler_chain_params)

class llama_chat_message(ctypes.Structure):
    _fields_ = [
        ("role",    ctypes.c_char_p),
        ("content", ctypes.c_char_p)
    ]

llama_chat_message_p = ctypes.POINTER(llama_chat_message)

class llama_adapter_lora(ctypes.Structure):
    pass

llama_adapter_lora_p = ctypes.POINTER(llama_adapter_lora)

#
# libllama initialization
#

_PATH: Optional[str] = None
"""If libllama is loaded, this is the path it was loaded from."""

def llama_backend_init() -> None:
    """Initialize the libllama backend. This function needs to be called before any other
    libllama functions can be used."""
    global _BACKEND_INIT, libllama, _PATH

    if _BACKEND_INIT:
        return

    faulthandler.enable() # prints more helpful info if python crashes

    log_verbose(
        f'easy_llama v{__version__} '
        f'targeting llama.cpp@{_TARGET_LLAMACPP_COMMIT[:8]} ({_TARGET_LLAMACPP_TIME})'
    )

    lib_path = os.environ.get('LIBLLAMA')

    if lib_path is None:
        log(f'failed to load libllama: $LIBLLAMA is not set', 3)
        raise OSError(
            f"The llama.cpp shared library could not be loaded because the LIBLLAMA "
            f"environment variable is not set. You must set LIBLLAMA to the path of your "
            f"libllama shared library file (`.so`, `.dll`, or `.dylib`). For example, on "
            f"Linux: `export LIBLLAMA=/path/to/your/libllama.so`."
        )

    if not os.path.exists(lib_path):
        log(f'failed to load libllama: $LIBLLAMA does not exist: {lib_path}', 3)
        raise FileNotFoundError(
            f"The llama.cpp shared library could not be loaded because the LIBLLAMA "
            f"environment variable points to a file that does not actually exist. You must "
            f"set LIBLLAMA to the path of your llama.cpp shared library file (`.so`, `.dll`, "
            f"or `.dylib`). For example, on Linux: `export LIBLLAMA=/path/to/your/libllama.so`."
        )

    if os.path.isdir(lib_path):
        log(f'failed to load libllama: $LIBLLAMA is a directory, not a file: {lib_path}', 3)
        raise IsADirectoryError(
            f"The llama.cpp shared library could not be loaded because the LIBLLAMA "
            f"environment variable points to a directory, but it should point to a file "
            f"instead. You must set LIBLLAMA to the path of your llama.cpp shared library file "
            f"(`.so`, `.dll`, or `.dylib`). For example, on Linux: "
            f"`export LIBLLAMA=/path/to/your/libllama.so`."
        )

    try:
        libllama = ctypes.CDLL(lib_path)
    except Exception as exc:
        log(f'failed to load libllama from {lib_path}: {type(exc).__name__}: {exc}', 3)
        raise exc
    else:
        log(f'loaded libllama from {lib_path}')

    # actually initialize the backend
    libllama.llama_backend_init.argtypes = []
    libllama.llama_backend_init.restype = None
    libllama.llama_backend_init()

    _PATH = lib_path

    # initialize the built-in greedy sampler
    _internals.greedy_sampler = llama_sampler_init_greedy()

    # don't let this function be called again
    _BACKEND_INIT = True

def llama_version() -> str:
    """Get the version of the loaded libllama build"""
    libllama.llama_version.argtypes = []
    libllama.llama_version.restype = ctypes.c_char_p
    return _cstr(libllama.llama_version())

#
# *_default_params
#

def llama_model_default_params() -> llama_model_params:
    """Get the default parameters for a llama model"""
    libllama.llama_model_default_params.argtypes = []
    libllama.llama_model_default_params.restype = llama_model_params
    return libllama.llama_model_default_params()

def llama_context_default_params() -> llama_context_params:
    """Get the default parameters for a llama context"""
    libllama.llama_context_default_params.argtypes = []
    libllama.llama_context_default_params.restype = llama_context_params
    return libllama.llama_context_default_params()

def llama_sampler_chain_default_params() -> llama_sampler_chain_params:
    """Get the default parameters for a sampler chain"""
    libllama.llama_sampler_chain_default_params.argtypes = []
    libllama.llama_sampler_chain_default_params.restype = llama_sampler_chain_params
    return libllama.llama_sampler_chain_default_params()

def llama_model_quantize_default_params() -> llama_model_quantize_params:
    """Get the default parameters for model quantization"""
    libllama.llama_model_quantize_default_params.argtypes = []
    libllama.llama_model_quantize_default_params.restype = llama_model_quantize_params
    return libllama.llama_model_quantize_default_params()

#
# Setup and teardown
#

def llama_backend_free() -> None:
    """Free the libllama backend

    Call once at the end of the program - currently only used for MPI"""
    global _BACKEND_INIT
    if not _BACKEND_INIT:
        return

    libllama.llama_backend_free.argtypes = []
    libllama.llama_backend_free.restype = None
    libllama.llama_backend_free()

    # free the built-in greedy sampler; a subsequent llama_backend_init() will create a new one
    if _internals.greedy_sampler is not None:
        llama_sampler_free(_internals.greedy_sampler)
        _internals.greedy_sampler = None

    _BACKEND_INIT = False

def llama_numa_init(numa: int) -> None:
    """Initialize NUMA optimizations globally. See `ggml_numa_strategy`."""
    libllama.llama_numa_init.argtypes = [ctypes.c_int]
    libllama.llama_numa_init.restype = None
    libllama.llama_numa_init(numa)

def llama_attach_threadpool(
    ctx: ptr[llama_context],
    threadpool: ggml_threadpool_t,
    threadpool_batch: ggml_threadpool_t
) -> None:
    """Attach a threadpool to a llama_context"""
    libllama.llama_attach_threadpool.argtypes = [
        llama_context_p, ggml_threadpool_t, ggml_threadpool_t
    ]
    libllama.llama_attach_threadpool.restype = None
    libllama.llama_attach_threadpool(ctx, threadpool, threadpool_batch)

def llama_detach_threadpool(ctx: ptr[llama_context]) -> None:
    """Detach a threadpool from a llama_context"""
    libllama.llama_detach_threadpool.argtypes = [llama_context_p]
    libllama.llama_detach_threadpool.restype = None
    libllama.llama_detach_threadpool(ctx)

def llama_model_init_from_user(
    metadata: ptr[gguf_context],
    set_tensor_data: Callable,
    set_tensor_data_ud: ptr,
    params: llama_model_params
) -> ptr[llama_model]:
    """Create a new model from GGUF metadata plus a function used to set the tensor data.

    Tensors are created as GGML_TYPE_F32 by default; override by adding a tensor with the same
    name but a different type to the context."""
    libllama.llama_model_init_from_user.argtypes = [
        gguf_context_p, model_set_tensor_data_functype, ctypes.c_void_p, llama_model_params
    ]
    libllama.llama_model_init_from_user.restype = llama_model_p
    return libllama.llama_model_init_from_user(metadata, set_tensor_data, set_tensor_data_ud, params)

def llama_model_load_from_file(
    path_model: str,
    params: llama_model_params
) -> ptr[llama_model]:
    """Load a llama model from a file. Returns a pointer.

    If the file is split into multiple parts, the file name must follow this pattern:
    `<name>-%05d-of-%05d.gguf`. If the split file name does not follow this pattern, use
    `llama_model_load_from_splits`."""
    libllama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    libllama.llama_model_load_from_file.restype = llama_model_p
    return libllama.llama_model_load_from_file(path_model.encode('utf-8'), params)

def llama_model_load_from_file_ptr(
    file: FILE_p,
    params: llama_model_params
) -> ptr[llama_model]:
    """Load a llama model from an open FILE pointer.

    The GGUF is read from the current position, so it can be embedded in a larger file. mmap
    needs the GGUF data section at a file offset to be aligned to the CPU tensor alignment
    (32 bytes)."""
    libllama.llama_model_load_from_file_ptr.argtypes = [FILE_p, llama_model_params]
    libllama.llama_model_load_from_file_ptr.restype = llama_model_p
    return libllama.llama_model_load_from_file_ptr(file, params)

def llama_model_load_from_splits(
    paths: list[str],
    n_paths: int,
    params: llama_model_params
) -> ptr[llama_model]:
    """Load a llama model from multiple splits (support custom naming scheme). Returns a pointer.

    The paths must be in the correct order."""
    libllama.llama_model_load_from_splits.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t, llama_model_params
    ]
    libllama.llama_model_load_from_splits.restype = llama_model_p
    c_paths = (ctypes.c_char_p * n_paths)(*[p.encode('utf-8') for p in paths])
    return libllama.llama_model_load_from_splits(c_paths, n_paths, params)

def llama_model_save_to_file(model: ptr[llama_model], path_model: str) -> None:
    """Save a llama model to a file."""
    libllama.llama_model_save_to_file.argtypes = [llama_model_p, ctypes.c_char_p]
    libllama.llama_model_save_to_file.restype = None
    libllama.llama_model_save_to_file(model, path_model.encode('utf-8'))

def llama_model_free(model: ptr[llama_model]) -> None:
    """Free a model"""
    libllama.llama_model_free.argtypes = [llama_model_p]
    libllama.llama_model_free.restype = None
    libllama.llama_model_free(model)

def llama_init_from_model(
    model: ptr[llama_model],
    params: llama_context_params
) -> ptr[llama_context]:
    """Create a new llama context with a loaded model"""
    libllama.llama_init_from_model.argtypes = [llama_model_p, llama_context_params]
    libllama.llama_init_from_model.restype = llama_context_p
    return libllama.llama_init_from_model(model, params)

def llama_free(ctx: ptr[llama_context]) -> None:
    """Frees all allocated memory"""
    libllama.llama_free.argtypes = [llama_context_p]
    libllama.llama_free.restype = None
    libllama.llama_free(ctx)

#
# llama backend helper functions
#

def llama_time_us() -> int:
    """Get the current time in microseconds"""
    libllama.llama_time_us.argtypes = []
    libllama.llama_time_us.restype = ctypes.c_int64
    return libllama.llama_time_us()

def llama_max_devices() -> int:
    """Get the maximum number of devices"""
    libllama.llama_max_devices.argtypes = []
    libllama.llama_max_devices.restype = ctypes.c_size_t
    return libllama.llama_max_devices()

def llama_max_parallel_sequences() -> int:
    """Get the maximum number of parallel sequences supported by the backend."""
    libllama.llama_max_parallel_sequences.argtypes = []
    libllama.llama_max_parallel_sequences.restype = ctypes.c_size_t
    return libllama.llama_max_parallel_sequences()

def llama_max_tensor_buft_overrides() -> int:
    """Get the maximum number of tensor buffer type overrides supported by the backend."""
    libllama.llama_max_tensor_buft_overrides.argtypes = []
    libllama.llama_max_tensor_buft_overrides.restype = ctypes.c_size_t
    return libllama.llama_max_tensor_buft_overrides()

def llama_supports_mmap() -> bool:
    """Check if mmap is supported"""
    libllama.llama_supports_mmap.argtypes = []
    libllama.llama_supports_mmap.restype = ctypes.c_bool
    return libllama.llama_supports_mmap()

def llama_supports_mlock() -> bool:
    """Check if mlock is supported"""
    libllama.llama_supports_mlock.argtypes = []
    libllama.llama_supports_mlock.restype = ctypes.c_bool
    return libllama.llama_supports_mlock()

def llama_supports_gpu_offload() -> bool:
    """Check if GPU offload is supported"""
    libllama.llama_supports_gpu_offload.argtypes = []
    libllama.llama_supports_gpu_offload.restype = ctypes.c_bool
    return libllama.llama_supports_gpu_offload()

def llama_supports_rpc() -> bool:
    """Check if RPC is supported"""
    libllama.llama_supports_rpc.argtypes = []
    libllama.llama_supports_rpc.restype = ctypes.c_bool
    return libllama.llama_supports_rpc()

#
# getters for llama_context
#
# NOTE: After creating a llama_context, it is recommended to query the actual values using these
#       functions. In some cases the requested values via llama_context_params may differ from
#       the actual values used by the context.
#       ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
#

def llama_n_ctx(ctx: ptr[llama_context]) -> int:
    """Get the context size"""
    libllama.llama_n_ctx.argtypes = [llama_context_p]
    libllama.llama_n_ctx.restype = ctypes.c_uint32
    return libllama.llama_n_ctx(ctx)

def llama_n_ctx_seq(ctx: ptr[llama_context]) -> int:
    """Get the context size per sequence"""
    libllama.llama_n_ctx_seq.argtypes = [llama_context_p]
    libllama.llama_n_ctx_seq.restype = ctypes.c_uint32
    return libllama.llama_n_ctx_seq(ctx)

def llama_n_batch(ctx: ptr[llama_context]) -> int:
    """Get the logical maximum batch size"""
    libllama.llama_n_batch.argtypes = [llama_context_p]
    libllama.llama_n_batch.restype = ctypes.c_uint32
    return libllama.llama_n_batch(ctx)

def llama_n_ubatch(ctx: ptr[llama_context]) -> int:
    """Get the physical maximum batch size"""
    libllama.llama_n_ubatch.argtypes = [llama_context_p]
    libllama.llama_n_ubatch.restype = ctypes.c_uint32
    return libllama.llama_n_ubatch(ctx)

def llama_n_seq_max(ctx: ptr[llama_context]) -> int:
    """Get the maximum number of sequences"""
    libllama.llama_n_seq_max.argtypes = [llama_context_p]
    libllama.llama_n_seq_max.restype = ctypes.c_uint32
    return libllama.llama_n_seq_max(ctx)

def llama_n_rs_seq(ctx: ptr[llama_context]) -> int:
    """Get the number of recurrent-state snapshots per sequence for rollback (0 = none)"""
    libllama.llama_n_rs_seq.argtypes = [llama_context_p]
    libllama.llama_n_rs_seq.restype = ctypes.c_uint32
    return libllama.llama_n_rs_seq(ctx)

def llama_get_model(ctx: ptr[llama_context]) -> ptr[llama_model]:
    """Get the model associated with a context"""
    libllama.llama_get_model.argtypes = [llama_context_p]
    libllama.llama_get_model.restype = llama_model_p
    return libllama.llama_get_model(ctx)

def llama_get_memory(ctx: ptr[llama_context]) -> llama_memory_t:
    """Get the memory (formerly KV cache) of a context"""
    libllama.llama_get_memory.argtypes = [llama_context_p]
    libllama.llama_get_memory.restype = llama_memory_t
    return libllama.llama_get_memory(ctx)

def llama_pooling_type(ctx: ptr[llama_context]) -> int:
    """Get the pooling type used by a context. See `LlamaPoolingType`."""
    libllama.llama_pooling_type.argtypes = [llama_context_p]
    libllama.llama_pooling_type.restype = ctypes.c_int
    return libllama.llama_pooling_type(ctx)

#
# llama_model getters
#

def llama_model_get_vocab(model: ptr[llama_model]) -> ptr[llama_vocab]:
    """Get a pointer to the llama_vocab struct"""
    libllama.llama_model_get_vocab.argtypes = [llama_model_p]
    libllama.llama_model_get_vocab.restype = llama_vocab_p
    return libllama.llama_model_get_vocab(model)

def llama_model_rope_type(model: ptr[llama_model]) -> int:
    """Get the RoPE type used by a model. See `LlamaRopeType`."""
    libllama.llama_model_rope_type.argtypes = [llama_model_p]
    libllama.llama_model_rope_type.restype = ctypes.c_int
    return libllama.llama_model_rope_type(model)

def llama_model_n_ctx_train(model: ptr[llama_model]) -> int:
    """Get the context size used during training"""
    libllama.llama_model_n_ctx_train.argtypes = [llama_model_p]
    libllama.llama_model_n_ctx_train.restype = ctypes.c_int32
    return libllama.llama_model_n_ctx_train(model)

def llama_model_n_embd(model: ptr[llama_model]) -> int:
    """Get the embedding size"""
    libllama.llama_model_n_embd.argtypes = [llama_model_p]
    libllama.llama_model_n_embd.restype = ctypes.c_int32
    return libllama.llama_model_n_embd(model)

def llama_model_n_embd_inp(model: ptr[llama_model]) -> int:
    """Get the size of the input embeddings"""
    libllama.llama_model_n_embd_inp.argtypes = [llama_model_p]
    libllama.llama_model_n_embd_inp.restype = ctypes.c_int32
    return libllama.llama_model_n_embd_inp(model)

def llama_model_n_embd_out(model: ptr[llama_model]) -> int:
    """Get the size of the output embeddings"""
    libllama.llama_model_n_embd_out.argtypes = [llama_model_p]
    libllama.llama_model_n_embd_out.restype = ctypes.c_int32
    return libllama.llama_model_n_embd_out(model)

def llama_model_n_layer(model: ptr[llama_model]) -> int:
    """Get the number of layers"""
    libllama.llama_model_n_layer.argtypes = [llama_model_p]
    libllama.llama_model_n_layer.restype = ctypes.c_int32
    return libllama.llama_model_n_layer(model)

def llama_model_n_layer_nextn(model: ptr[llama_model]) -> int:
    """Get the number of MTP (next-n) layers"""
    libllama.llama_model_n_layer_nextn.argtypes = [llama_model_p]
    libllama.llama_model_n_layer_nextn.restype = ctypes.c_int32
    return libllama.llama_model_n_layer_nextn(model)

def llama_model_n_head(model: ptr[llama_model]) -> int:
    """Get the number of attention heads"""
    libllama.llama_model_n_head.argtypes = [llama_model_p]
    libllama.llama_model_n_head.restype = ctypes.c_int32
    return libllama.llama_model_n_head(model)

def llama_model_n_head_kv(model: ptr[llama_model]) -> int:
    """Get the number of KV heads"""
    libllama.llama_model_n_head_kv.argtypes = [llama_model_p]
    libllama.llama_model_n_head_kv.restype = ctypes.c_int32
    return libllama.llama_model_n_head_kv(model)

def llama_model_n_swa(model: ptr[llama_model]) -> int:
    """Get the size of the sliding window for models which use SWA."""
    libllama.llama_model_n_swa.argtypes = [llama_model_p]
    libllama.llama_model_n_swa.restype = ctypes.c_int32
    return libllama.llama_model_n_swa(model)

def llama_model_rope_freq_scale_train(model: ptr[llama_model]) -> float:
    """Get the RoPE frequency scaling factor used during training"""
    libllama.llama_model_rope_freq_scale_train.argtypes = [llama_model_p]
    libllama.llama_model_rope_freq_scale_train.restype = ctypes.c_float
    return libllama.llama_model_rope_freq_scale_train(model)

def llama_model_n_cls_out(model: ptr[llama_model]) -> int:
    """Returns the number of classifier outputs (only valid for classifier models).
    Undefined behavior for non-classifier models."""
    libllama.llama_model_n_cls_out.argtypes = [llama_model_p]
    libllama.llama_model_n_cls_out.restype = ctypes.c_uint32
    return libllama.llama_model_n_cls_out(model)

def llama_model_cls_label(model: ptr[llama_model], i: int) -> Optional[str]:
    """Returns label of classifier output by index (<n_cls_out).
    Returns None if no label provided."""
    libllama.llama_model_cls_label.argtypes = [llama_model_p, ctypes.c_uint32]
    libllama.llama_model_cls_label.restype = ctypes.c_char_p
    c_label = libllama.llama_model_cls_label(model, i)
    return _cstr(c_label) if c_label else None

#
# llama_vocab getters
#

def llama_vocab_type(vocab: ptr[llama_vocab]) -> int:
    """Get the LlamaVocabType of this llama_vocab"""
    libllama.llama_vocab_type.argtypes = [llama_vocab_p]
    libllama.llama_vocab_type.restype = ctypes.c_int
    return libllama.llama_vocab_type(vocab)

def llama_vocab_n_tokens(vocab: ptr[llama_vocab]) -> int:
    """Get the number of tokens in this llama_vocab"""
    libllama.llama_vocab_n_tokens.argtypes = [llama_vocab_p]
    libllama.llama_vocab_n_tokens.restype = ctypes.c_int32
    return libllama.llama_vocab_n_tokens(vocab)

#
# GGUF metadata functions
#
# - The functions return the length of the string on success, or -1 on failure
# - The output string is always null-terminated and cleared on failure
# - When retrieving a string, an extra byte must be allocated to account for the null terminator
# - GGUF array values are not supported by these functions
#

def llama_model_meta_val_str(
    model: ptr[llama_model],
    key: str,
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    buf_size: int
) -> int:
    """Get a metadata value as a string by key name"""
    libllama.llama_model_meta_val_str.argtypes = [
        llama_model_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t
    ]
    libllama.llama_model_meta_val_str.restype = ctypes.c_int32
    return libllama.llama_model_meta_val_str(model, key.encode('utf-8'), buf, buf_size)

def llama_model_meta_count(model: ptr[llama_model]) -> int:
    """Get the number of metadata key-value pairs"""
    libllama.llama_model_meta_count.argtypes = [llama_model_p]
    libllama.llama_model_meta_count.restype = ctypes.c_int32
    return libllama.llama_model_meta_count(model)

def llama_model_meta_key_by_index(
    model: ptr[llama_model],
    i: int,
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    buf_size: int
) -> int:
    """Get a metadata key by index"""
    libllama.llama_model_meta_key_by_index.argtypes = [
        llama_model_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t
    ]
    libllama.llama_model_meta_key_by_index.restype = ctypes.c_int32
    return libllama.llama_model_meta_key_by_index(model, i, buf, buf_size)

def llama_model_meta_val_str_by_index(
    model: ptr[llama_model],
    i: int,
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    buf_size: int
) -> int:
    """Get a metadata value by index"""
    libllama.llama_model_meta_val_str_by_index.argtypes = [
        llama_model_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t
    ]
    libllama.llama_model_meta_val_str_by_index.restype = ctypes.c_int32
    return libllama.llama_model_meta_val_str_by_index(model, i, buf, buf_size)

def llama_model_desc(
    model: ptr[llama_model],
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    buf_size: int
) -> int:
    """Get a string describing the model type"""
    libllama.llama_model_desc.argtypes = [llama_model_p, ctypes.c_char_p, ctypes.c_size_t]
    libllama.llama_model_desc.restype = ctypes.c_int32
    return libllama.llama_model_desc(model, buf, buf_size)

def llama_model_ftype(model: ptr[llama_model]) -> int:
    """Get the model file type (quantization). See `LlamaFType`."""
    libllama.llama_model_ftype.argtypes = [llama_model_p]
    libllama.llama_model_ftype.restype = ctypes.c_int
    return libllama.llama_model_ftype(model)

def llama_model_size(model: ptr[llama_model]) -> int:
    """Get the total size of all tensors in the model in bytes"""
    libllama.llama_model_size.argtypes = [llama_model_p]
    libllama.llama_model_size.restype = ctypes.c_uint64
    return libllama.llama_model_size(model)

def llama_model_chat_template(
    model: ptr[llama_model],
    name: Optional[str] = None
) -> Optional[str]:
    """Get the default chat template. Returns None if not available.
    If `name` is None, return the default chat template."""
    libllama.llama_model_chat_template.argtypes = [llama_model_p, ctypes.c_char_p]
    libllama.llama_model_chat_template.restype = ctypes.c_char_p
    c_name = name.encode('utf-8') if name is not None else None
    c_template = libllama.llama_model_chat_template(model, c_name)
    return _cstr(c_template) if c_template else None

def llama_model_n_params(model: ptr[llama_model]) -> int:
    """Get the total number of parameters in the model"""
    libllama.llama_model_n_params.argtypes = [llama_model_p]
    libllama.llama_model_n_params.restype = ctypes.c_uint64
    return libllama.llama_model_n_params(model)

def llama_model_has_encoder(model: ptr[llama_model]) -> bool:
    """Check if the model has an encoder"""
    libllama.llama_model_has_encoder.argtypes = [llama_model_p]
    libllama.llama_model_has_encoder.restype = ctypes.c_bool
    return libllama.llama_model_has_encoder(model)

def llama_model_has_decoder(model: ptr[llama_model]) -> bool:
    """Check if the model has a decoder"""
    libllama.llama_model_has_decoder.argtypes = [llama_model_p]
    libllama.llama_model_has_decoder.restype = ctypes.c_bool
    return libllama.llama_model_has_decoder(model)

def llama_model_decoder_start_token(model: ptr[llama_model]) -> int:
    """Get the start token for the decoder. Returns -1 for non encoder-decoder models."""
    libllama.llama_model_decoder_start_token.argtypes = [llama_model_p]
    libllama.llama_model_decoder_start_token.restype = llama_token
    return libllama.llama_model_decoder_start_token(model)

def llama_model_is_recurrent(model: ptr[llama_model]) -> bool:
    """Check if the model is recurrent (like Mamba, RWKV, etc.)"""
    libllama.llama_model_is_recurrent.argtypes = [llama_model_p]
    libllama.llama_model_is_recurrent.restype = ctypes.c_bool
    return libllama.llama_model_is_recurrent(model)

def llama_model_is_hybrid(model: ptr[llama_model]) -> bool:
    """Check if the model is hybrid (like Jamba, Granite, etc.)"""
    libllama.llama_model_is_hybrid.argtypes = [llama_model_p]
    libllama.llama_model_is_hybrid.restype = ctypes.c_bool
    return libllama.llama_model_is_hybrid(model)

def llama_model_is_diffusion(model: ptr[llama_model]) -> bool:
    """Check if the model is diffusion-based (like LLaDA, Dream, etc.)"""
    libllama.llama_model_is_diffusion.argtypes = [llama_model_p]
    libllama.llama_model_is_diffusion.restype = ctypes.c_bool
    return libllama.llama_model_is_diffusion(model)

#
# quantization
#

def llama_model_quantize(
    fname_inp: str,
    fname_out: str,
    params: llama_model_quantize_params
) -> int:
    """Quantize a model. Returns 0 on success"""
    libllama.llama_model_quantize.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(llama_model_quantize_params)
    ]
    libllama.llama_model_quantize.restype = ctypes.c_uint32
    return libllama.llama_model_quantize(
        fname_inp.encode('utf-8'),
        fname_out.encode('utf-8'),
        ctypes.byref(params)
    )

#
# llama_adapter_*
#

def llama_adapter_lora_init(
    model: ptr[llama_model],
    path_lora: str
) -> ptr[llama_adapter_lora]:
    """Load a LoRA adapter from file.

    The adapter is valid as long as the associated model is not freed."""
    libllama.llama_adapter_lora_init.argtypes = [llama_model_p, ctypes.c_char_p]
    libllama.llama_adapter_lora_init.restype = llama_adapter_lora_p
    return libllama.llama_adapter_lora_init(model, path_lora.encode('utf-8'))

def llama_adapter_lora_init_from_file_ptr(
    model: ptr[llama_model],
    file: FILE_p
) -> ptr[llama_adapter_lora]:
    """Load a LoRA adapter from an open FILE pointer, reading from its current position"""
    libllama.llama_adapter_lora_init_from_file_ptr.argtypes = [llama_model_p, FILE_p]
    libllama.llama_adapter_lora_init_from_file_ptr.restype = llama_adapter_lora_p
    return libllama.llama_adapter_lora_init_from_file_ptr(model, file)

def llama_adapter_meta_val_str(
    adapter: ptr[llama_adapter_lora],
    key: str,
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    buf_size: int
) -> int:
    """Get metadata value as a string by key name"""
    libllama.llama_adapter_meta_val_str.argtypes = [
        llama_adapter_lora_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t
    ]
    libllama.llama_adapter_meta_val_str.restype = ctypes.c_int32
    return libllama.llama_adapter_meta_val_str(adapter, key.encode('utf-8'), buf, buf_size)

def llama_adapter_meta_count(adapter: ptr[llama_adapter_lora]) -> int:
    """Get the number of metadata key/value pairs"""
    libllama.llama_adapter_meta_count.argtypes = [llama_adapter_lora_p]
    libllama.llama_adapter_meta_count.restype = ctypes.c_int32
    return libllama.llama_adapter_meta_count(adapter)

def llama_adapter_meta_key_by_index(
    adapter: ptr[llama_adapter_lora],
    i: int,
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    buf_size: int
) -> int:
    """Get metadata key name by index"""
    libllama.llama_adapter_meta_key_by_index.argtypes = [
        llama_adapter_lora_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t
    ]
    libllama.llama_adapter_meta_key_by_index.restype = ctypes.c_int32
    return libllama.llama_adapter_meta_key_by_index(adapter, i, buf, buf_size)

def llama_adapter_meta_val_str_by_index(
    adapter: ptr[llama_adapter_lora],
    i: int,
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    buf_size: int
) -> int:
    """Get metadata value as a string by index"""
    libllama.llama_adapter_meta_val_str_by_index.argtypes = [
        llama_adapter_lora_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t
    ]
    libllama.llama_adapter_meta_val_str_by_index.restype = ctypes.c_int32
    return libllama.llama_adapter_meta_val_str_by_index(adapter, i, buf, buf_size)

def llama_adapter_lora_free(adapter: ptr[llama_adapter_lora]) -> None:
    """Manually free a LoRA adapter

    NOTE: loaded adapters that are not manually freed will be freed when the associated model
    is deleted"""
    libllama.llama_adapter_lora_free.argtypes = [llama_adapter_lora_p]
    libllama.llama_adapter_lora_free.restype = None
    libllama.llama_adapter_lora_free(adapter)

def llama_adapter_get_alora_n_invocation_tokens(adapter: ptr[llama_adapter_lora]) -> int:
    """Get the number of invocation tokens if the current lora is an alora"""
    libllama.llama_adapter_get_alora_n_invocation_tokens.argtypes = [llama_adapter_lora_p]
    libllama.llama_adapter_get_alora_n_invocation_tokens.restype = ctypes.c_uint64
    return libllama.llama_adapter_get_alora_n_invocation_tokens(adapter)

def llama_adapter_get_alora_invocation_tokens(adapter: ptr[llama_adapter_lora]) -> list[int]:
    """Get the invocation tokens if the current lora is an alora"""
    libllama.llama_adapter_get_alora_invocation_tokens.argtypes = [llama_adapter_lora_p]
    libllama.llama_adapter_get_alora_invocation_tokens.restype = ctypes.POINTER(llama_token)
    n_tokens = llama_adapter_get_alora_n_invocation_tokens(adapter)
    c_tokens = libllama.llama_adapter_get_alora_invocation_tokens(adapter)
    return [c_tokens[i] for i in range(n_tokens)]

def llama_set_adapters_lora(
    ctx: ptr[llama_context],
    adapters: ptr[llama_adapter_lora],
    n_adapters: int,
    scales: ptr[ctypes.c_float]
) -> int:
    """Set LoRA adapters on the context.

    Will only modify the adapters if the adapters currently in the context are different.
    `adapters` is an array of `n_adapters` `llama_adapter_lora_p` entries (build one with
    `_internals.get_adapter_array`) and `scales` is an array of `n_adapters` floats (build one
    with `_internals.get_float_array`). Both arrays must outlive this call."""
    libllama.llama_set_adapters_lora.argtypes = [
        llama_context_p, ctypes.POINTER(llama_adapter_lora_p), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float)
    ]
    libllama.llama_set_adapters_lora.restype = ctypes.c_int32
    return libllama.llama_set_adapters_lora(ctx, adapters, n_adapters, scales)

#
# control vector (activation steering)
#

def llama_set_adapter_cvec(
    ctx: ptr[llama_context],
    data: ptr[ctypes.c_float],
    length: int,
    n_embd: int,
    il_start: int,
    il_end: int
) -> int:
    """Apply a control vector to a llama_context, or if `data` is None, clear the currently
    loaded vector.

    `length` should be the size of a single layer's control, and `data` should point to a
    `n_embd x n_layers` buffer starting from layer 1. `il_start` and `il_end` are the layer
    range the vector should apply to (both inclusive)."""
    libllama.llama_set_adapter_cvec.argtypes = [
        llama_context_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32
    ]
    libllama.llama_set_adapter_cvec.restype = ctypes.c_int32
    return libllama.llama_set_adapter_cvec(ctx, data, length, n_embd, il_start, il_end)

#
# llama_memory
#

def llama_memory_clear(mem: llama_memory_t, data: bool) -> None:
    """Clear the memory contents. If `data` is `True`, the data buffers will also be cleared
    together with the metadata."""
    libllama.llama_memory_clear.argtypes = [llama_memory_t, ctypes.c_bool]
    libllama.llama_memory_clear.restype = None
    libllama.llama_memory_clear(mem, data)

def llama_memory_seq_rm(
    mem: llama_memory_t,
    seq_id: llama_seq_id,
    p0: llama_pos,
    p1: llama_pos
) -> bool:
    """Removes all tokens that belong to the specified sequence and have positions in [p0, p1).
    Returns False if a partial sequence cannot be removed. Removing a whole sequence never fails.
    seq_id < 0 : match any sequence [TAG_LLAMA_SEQ_ID_NEG]
    p0 < 0     : [0,  p1]
    p1 < 0     : [p0, inf)"""
    libllama.llama_memory_seq_rm.argtypes = [llama_memory_t, llama_seq_id, llama_pos, llama_pos]
    libllama.llama_memory_seq_rm.restype = ctypes.c_bool
    return libllama.llama_memory_seq_rm(mem, seq_id, p0, p1)

def llama_memory_seq_cp(
    mem: llama_memory_t,
    seq_id_src: llama_seq_id,
    seq_id_dst: llama_seq_id,
    p0: llama_pos,
    p1: llama_pos
) -> None:
    """Copy all tokens that belong to the specified sequence to another sequence.
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)"""
    libllama.llama_memory_seq_cp.argtypes = [
        llama_memory_t, llama_seq_id, llama_seq_id, llama_pos, llama_pos
    ]
    libllama.llama_memory_seq_cp.restype = None
    libllama.llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1)

def llama_memory_seq_keep(mem: llama_memory_t, seq_id: llama_seq_id) -> None:
    """Removes all tokens that do not belong to the specified sequence."""
    libllama.llama_memory_seq_keep.argtypes = [llama_memory_t, llama_seq_id]
    libllama.llama_memory_seq_keep.restype = None
    libllama.llama_memory_seq_keep(mem, seq_id)

def llama_memory_seq_add(
    mem: llama_memory_t,
    seq_id: llama_seq_id,
    p0: llama_pos,
    p1: llama_pos,
    delta: llama_pos
) -> None:
    """Adds relative position "delta" to all tokens that belong to the specified sequence and
    have positions in [p0, p1).
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)"""
    libllama.llama_memory_seq_add.argtypes = [
        llama_memory_t, llama_seq_id, llama_pos, llama_pos, llama_pos
    ]
    libllama.llama_memory_seq_add.restype = None
    libllama.llama_memory_seq_add(mem, seq_id, p0, p1, delta)

def llama_memory_seq_div(
    mem: llama_memory_t,
    seq_id: llama_seq_id,
    p0: llama_pos,
    p1: llama_pos,
    d: int
) -> None:
    """Integer division of the positions by factor of `d > 1`.
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)"""
    libllama.llama_memory_seq_div.argtypes = [
        llama_memory_t, llama_seq_id, llama_pos, llama_pos, ctypes.c_int
    ]
    libllama.llama_memory_seq_div.restype = None
    libllama.llama_memory_seq_div(mem, seq_id, p0, p1, d)

def llama_memory_seq_pos_min(mem: llama_memory_t, seq_id: llama_seq_id) -> int:
    """Returns the smallest position present in the memory for the specified sequence.
    This is typically non-zero only for SWA caches.
    Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the
    memory. Return -1 if the sequence is empty."""
    libllama.llama_memory_seq_pos_min.argtypes = [llama_memory_t, llama_seq_id]
    libllama.llama_memory_seq_pos_min.restype = llama_pos
    return libllama.llama_memory_seq_pos_min(mem, seq_id)

def llama_memory_seq_pos_max(mem: llama_memory_t, seq_id: llama_seq_id) -> int:
    """Returns the largest position present in the memory for the specified sequence.
    Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the
    memory. Return -1 if the sequence is empty."""
    libllama.llama_memory_seq_pos_max.argtypes = [llama_memory_t, llama_seq_id]
    libllama.llama_memory_seq_pos_max.restype = llama_pos
    return libllama.llama_memory_seq_pos_max(mem, seq_id)

def llama_memory_can_shift(mem: llama_memory_t) -> bool:
    """Check if the memory supports shifting."""
    libllama.llama_memory_can_shift.argtypes = [llama_memory_t]
    libllama.llama_memory_can_shift.restype = ctypes.c_bool
    return libllama.llama_memory_can_shift(mem)

#
# state and session mgmt.
#
# NOTE: Session files are versioned. A session written by a libllama with
#       LLAMA_SESSION_VERSION / LLAMA_STATE_SEQ_VERSION != the values compiled into the
#       currently loaded libllama will fail to load. see the constants near the top of this
#       file (currently LLAMA_SESSION_VERSION == 11, LLAMA_STATE_SEQ_VERSION == 4).
#       the state-seq flags (LLAMA_STATE_SEQ_FLAGS_*) are also defined near the top.
#

def llama_state_get_size(ctx: ptr[llama_context]) -> int:
    """Get the *actual* size in bytes of the state (logits, embedding and memory).

    Only use when saving the state, not when restoring it, otherwise the size may be too
    small."""
    libllama.llama_state_get_size.argtypes = [llama_context_p]
    libllama.llama_state_get_size.restype = ctypes.c_size_t
    return libllama.llama_state_get_size(ctx)

def llama_state_get_data(ctx: ptr[llama_context], dst: ctypes.c_void_p, size: int) -> int:
    """Copy the state to the specified destination address.

    `dst` needs to have allocated enough memory. Returns the number of bytes copied."""
    libllama.llama_state_get_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_size_t]
    libllama.llama_state_get_data.restype = ctypes.c_size_t
    return libllama.llama_state_get_data(ctx, dst, size)

def llama_state_set_data(ctx: ptr[llama_context], src: ctypes.c_void_p, size: int) -> int:
    """Set the state reading from the specified address. Returns the number of bytes read."""
    libllama.llama_state_set_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_size_t]
    libllama.llama_state_set_data.restype = ctypes.c_size_t
    return libllama.llama_state_set_data(ctx, src, size)

def llama_state_load_file(
    ctx: ptr[llama_context],
    path_session: str,
    tokens_out: ptr[llama_token],
    n_token_capacity: int,
    n_token_count_out: ptr[ctypes.c_size_t]
) -> bool:
    """Load the context state and session tokens from a file.

    :param path_session: The path to the session file
    :param tokens_out: A buffer to receive the loaded session tokens
    :param n_token_capacity: The maximum capacity of the `tokens_out` buffer
    :param n_token_count_out: A pointer to store the actual number of tokens loaded
    :return: True on success, False on failure"""
    libllama.llama_state_load_file.argtypes = [
        llama_context_p, ctypes.c_char_p, ctypes.POINTER(llama_token), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    libllama.llama_state_load_file.restype = ctypes.c_bool
    return libllama.llama_state_load_file(
        ctx,
        path_session.encode('utf-8'),
        tokens_out,
        ctypes.c_size_t(n_token_capacity),
        n_token_count_out
    )

def llama_state_save_file(ctx: ptr[llama_context], path_session: str, tokens: list[int]) -> bool:
    """Save the context state and current tokens to a file.

    :param path_session: The path to save the session file
    :param tokens: A list of tokens representing the current prompt/session
    :return: True on success, False on failure"""
    n_token_count = len(tokens)
    libllama.llama_state_save_file.argtypes = [
        llama_context_p, ctypes.c_char_p, ctypes.POINTER(llama_token), ctypes.c_size_t
    ]
    libllama.llama_state_save_file.restype = ctypes.c_bool
    return libllama.llama_state_save_file(
        ctx,
        path_session.encode('utf-8'),
        (llama_token * n_token_count)(*tokens),
        ctypes.c_size_t(n_token_count)
    )

def llama_state_seq_get_size(ctx: ptr[llama_context], seq_id: int) -> int:
    """Get the exact size needed to copy the state of a single sequence"""
    libllama.llama_state_seq_get_size.argtypes = [llama_context_p, llama_seq_id]
    libllama.llama_state_seq_get_size.restype = ctypes.c_size_t
    return libllama.llama_state_seq_get_size(ctx, seq_id)

def llama_state_seq_get_data(
    ctx: ptr[llama_context],
    dst: ctypes.c_void_p,
    size: int,
    seq_id: int
) -> int:
    """Copy the state of a single sequence into the specified buffer"""
    libllama.llama_state_seq_get_data.argtypes = [
        llama_context_p, ctypes.c_void_p, ctypes.c_size_t, llama_seq_id
    ]
    libllama.llama_state_seq_get_data.restype = ctypes.c_size_t
    return libllama.llama_state_seq_get_data(ctx, dst, size, seq_id)

def llama_state_seq_set_data(
    ctx: ptr[llama_context],
    src: ctypes.c_void_p,
    size: int,
    dest_seq_id: int
) -> int:
    """Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the
    specified sequence

    Returns:
    - Positive: Ok
    - Zero: Failed to load"""
    libllama.llama_state_seq_set_data.argtypes = [
        llama_context_p, ctypes.c_void_p, ctypes.c_size_t, llama_seq_id
    ]
    libllama.llama_state_seq_set_data.restype = ctypes.c_size_t
    return libllama.llama_state_seq_set_data(ctx, src, size, dest_seq_id)

def llama_state_seq_save_file(
    ctx: ptr[llama_context],
    filepath: str,
    seq_id: int,
    tokens: list[int]
) -> int:
    """Save the state of a single sequence, along with its tokens, to a file."""
    n_token_count = len(tokens)
    libllama.llama_state_seq_save_file.argtypes = [
        llama_context_p, ctypes.c_char_p, llama_seq_id, ctypes.POINTER(llama_token),
        ctypes.c_size_t
    ]
    libllama.llama_state_seq_save_file.restype = ctypes.c_size_t
    return libllama.llama_state_seq_save_file(
        ctx,
        filepath.encode('utf-8'),
        seq_id,
        (llama_token * n_token_count)(*tokens),
        ctypes.c_size_t(n_token_count)
    )

def llama_state_seq_load_file(
    ctx: ptr[llama_context],
    filepath: str,
    dest_seq_id: int,
    tokens_out: ptr[llama_token],
    n_token_capacity: int,
    n_token_count_out: ptr[ctypes.c_size_t]
) -> int:
    """Load the state of a single sequence from a file.

    If `tokens_out` is None, only the token count is reported through `n_token_count_out` and
    no state is loaded."""
    libllama.llama_state_seq_load_file.argtypes = [
        llama_context_p, ctypes.c_char_p, llama_seq_id, ctypes.POINTER(llama_token),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    libllama.llama_state_seq_load_file.restype = ctypes.c_size_t
    return libllama.llama_state_seq_load_file(
        ctx,
        filepath.encode('utf-8'),
        dest_seq_id,
        tokens_out,
        ctypes.c_size_t(n_token_capacity),
        n_token_count_out
    )

def llama_state_seq_get_size_ext(
    ctx: ptr[llama_context],
    seq_id: llama_seq_id,
    flags: llama_state_seq_flags
) -> int:
    """Get the exact size needed to copy part of the state of a single sequence.

    See the `LLAMA_STATE_SEQ_FLAGS_*` constants near the top of this file."""
    libllama.llama_state_seq_get_size_ext.argtypes = [
        llama_context_p, llama_seq_id, llama_state_seq_flags
    ]
    libllama.llama_state_seq_get_size_ext.restype = ctypes.c_size_t
    return libllama.llama_state_seq_get_size_ext(ctx, seq_id, flags)

def llama_state_seq_get_data_ext(
    ctx: ptr[llama_context],
    dst: ctypes.c_void_p,
    size: int,
    seq_id: llama_seq_id,
    flags: llama_state_seq_flags
) -> int:
    """Copy part of the state of a single sequence into the specified buffer.

    See the `LLAMA_STATE_SEQ_FLAGS_*` constants near the top of this file."""
    libllama.llama_state_seq_get_data_ext.argtypes = [
        llama_context_p, ctypes.c_void_p, ctypes.c_size_t, llama_seq_id, llama_state_seq_flags
    ]
    libllama.llama_state_seq_get_data_ext.restype = ctypes.c_size_t
    return libllama.llama_state_seq_get_data_ext(ctx, dst, size, seq_id, flags)

def llama_state_seq_set_data_ext(
    ctx: ptr[llama_context],
    src: ctypes.c_void_p,
    size: int,
    dest_seq_id: llama_seq_id,
    flags: llama_state_seq_flags
) -> int:
    """Copy part of the state of a single sequence from the specified buffer.

    See the `LLAMA_STATE_SEQ_FLAGS_*` constants near the top of this file."""
    libllama.llama_state_seq_set_data_ext.argtypes = [
        llama_context_p, ctypes.c_void_p, ctypes.c_size_t, llama_seq_id, llama_state_seq_flags
    ]
    libllama.llama_state_seq_set_data_ext.restype = ctypes.c_size_t
    return libllama.llama_state_seq_set_data_ext(ctx, src, size, dest_seq_id, flags)

#
# llama_batch
#

# NOTE: we're not supposed to use `llama_batch_get_one(...)`, so it's omitted

def llama_batch_init(n_tokens: int, embd: int, n_seq_max: int) -> llama_batch:
    """Allocate a batch of tokens on the heap that can hold a maximum of n_tokens.

    Each token can be assigned up to n_seq_max sequence ids. The batch has to be freed with
    `llama_batch_free()`. If `embd` != 0, `llama_batch.embd` will be allocated with size of
    `n_tokens * embd * sizeof(float)`. Otherwise, `llama_batch.token` will be allocated to store
    `n_tokens` llama_token. The rest of the llama_batch members are allocated with size
    n_tokens. All members are left uninitialized."""
    libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    libllama.llama_batch_init.restype = llama_batch
    return libllama.llama_batch_init(n_tokens, embd, n_seq_max)

def llama_batch_free(batch: llama_batch) -> None:
    """Frees a batch of tokens allocated with `llama_batch_init()`"""
    libllama.llama_batch_free.argtypes = [llama_batch]
    libllama.llama_batch_free.restype = None
    libllama.llama_batch_free(batch)

#
# llama_encode / llama_decode
#

def llama_encode(ctx: ptr[llama_context], batch: llama_batch) -> int:
    """Process a batch of tokens.

    In contrast to `llama_decode()` - this call does not use KV cache.
    For encoder-decoder contexts, processes the batch using the encoder.
    Can store the encoder output internally for later use by the decoder's cross-attention
    layers.
    Returns:
    - 0:
        success
    - < 0:
        error. the memory state is restored to the state before this call"""
    libllama.llama_encode.argtypes = [llama_context_p, llama_batch]
    libllama.llama_encode.restype = ctypes.c_int32
    return libllama.llama_encode(ctx, batch)

def llama_decode(ctx: ptr[llama_context], batch: llama_batch) -> int:
    """Process a batch of tokens.

    Requires the context to have a memory.
    For encoder-decoder contexts, processes the batch using the decoder.
    Positive return values does not mean a fatal error, but rather a warning.

    Upon fatal error or abort, the ubatches that managed to be been processed will remain in
    the memory state of the context.

    To handle this correctly, query the memory state using `llama_memory_seq_pos_min()` and
    `llama_memory_seq_pos_max()`.

    Upon other return values, the memory state is restored to the state before this call.
        - 0: success
        - 1: could not find a KV slot for the batch (try reducing the size of the batch or
          increase the context)
        - 2: aborted (processed ubatches will remain in the context's memory)
        - -1: invalid input batch
        - < -1: fatal error (processed ubatches will remain in the context's memory)
    """
    # TODO: wrap the function call in try/except for KeyboardInterrupt?
    libllama.llama_decode.argtypes = [llama_context_p, llama_batch]
    libllama.llama_decode.restype = ctypes.c_int32
    return libllama.llama_decode(ctx, batch)

class llama_embd(ctypes.Structure):
    _fields_ = [
        ("data",   ctypes.POINTER(ctypes.c_float)), # pointer to embedding data
        ("n_rows", ctypes.c_size_t               ), # number of embedding rows in data
        ("n_embd", ctypes.c_size_t               )  # size of one row
    ]

llama_embd_p = ctypes.POINTER(llama_embd)

#
# llama_batch_ext
#

def llama_batch_ext_init(ctx: ptr[llama_context]) -> ptr[llama_batch_ext]:
    """Initialize an extended batch for use with a context"""
    libllama.llama_batch_ext_init.argtypes = [llama_context_p]
    libllama.llama_batch_ext_init.restype = llama_batch_ext_p
    return libllama.llama_batch_ext_init(ctx)

def llama_batch_ext_free(batch: ptr[llama_batch_ext]) -> None:
    """Free an extended batch"""
    libllama.llama_batch_ext_free.argtypes = [llama_batch_ext_p]
    libllama.llama_batch_ext_free.restype = None
    libllama.llama_batch_ext_free(batch)

def llama_batch_ext_clear(batch: ptr[llama_batch_ext]) -> None:
    """Clear an extended batch (without freeing it)"""
    libllama.llama_batch_ext_clear.argtypes = [llama_batch_ext_p]
    libllama.llama_batch_ext_clear.restype = None
    libllama.llama_batch_ext_clear(batch)

def llama_batch_ext_add(batch: ptr[llama_batch_ext], seq_id: llama_seq_id) -> int:
    """Add an input token to the batch, with default values:
        id   = LLAMA_TOKEN_NULL
        embd = NULL
        pos  = not set, the caller must set it with `llama_batch_ext_set_pos()`

    Returns the batch index (>= 0). On error:
        -1: batch is full
        -2: token is invalid (id == LLAMA_TOKEN_NULL or invalid embd)
        -3: invalid sequence id"""
    libllama.llama_batch_ext_add.argtypes = [llama_batch_ext_p, llama_seq_id]
    libllama.llama_batch_ext_add.restype = ctypes.c_int32
    return libllama.llama_batch_ext_add(batch, seq_id)

def llama_batch_ext_add_token(
    batch: ptr[llama_batch_ext],
    seq_id: llama_seq_id,
    id: llama_token
) -> int:
    """Add an input token with a specified token ID to the batch.

    Returns the batch index (>= 0), or a negative value on error (see `llama_batch_ext_add`)."""
    libllama.llama_batch_ext_add_token.argtypes = [llama_batch_ext_p, llama_seq_id, llama_token]
    libllama.llama_batch_ext_add_token.restype = ctypes.c_int32
    return libllama.llama_batch_ext_add_token(batch, seq_id, id)

def llama_batch_ext_add_embd(
    batch: ptr[llama_batch_ext],
    seq_id: llama_seq_id,
    embd: llama_embd
) -> int:
    """Add an input token with a specified token embedding to the batch.

    Returns the batch index (>= 0), or a negative value on error (see `llama_batch_ext_add`)."""
    libllama.llama_batch_ext_add_embd.argtypes = [llama_batch_ext_p, llama_seq_id, llama_embd]
    libllama.llama_batch_ext_add_embd.restype = ctypes.c_int32
    return libllama.llama_batch_ext_add_embd(batch, seq_id, embd)

def llama_batch_ext_add_seq(
    batch: ptr[llama_batch_ext],
    idx: int,
    seq_id: llama_seq_id
) -> bool:
    """Add the token at index `idx` in the batch to another sequence id. The position will stay
    the same.

    NOTE: this should be called before other `_set_*()` functions."""
    libllama.llama_batch_ext_add_seq.argtypes = [llama_batch_ext_p, ctypes.c_int32, llama_seq_id]
    libllama.llama_batch_ext_add_seq.restype = ctypes.c_bool
    return libllama.llama_batch_ext_add_seq(batch, idx, seq_id)

def llama_batch_ext_set_embd_token(
    batch: ptr[llama_batch_ext],
    idx: int,
    embd: llama_embd
) -> bool:
    """Set the token embedding for the token at index `idx` in the batch.

    Use it after `llama_batch_ext_add_token()` to have an entry with both a token id and an
    embedding."""
    libllama.llama_batch_ext_set_embd_token.argtypes = [
        llama_batch_ext_p, ctypes.c_int32, llama_embd
    ]
    libllama.llama_batch_ext_set_embd_token.restype = ctypes.c_bool
    return libllama.llama_batch_ext_set_embd_token(batch, idx, embd)

def llama_batch_ext_set_embd_state(
    batch: ptr[llama_batch_ext],
    idx: int,
    embd: llama_embd
) -> bool:
    """Set the "state" embedding for the token at index `idx` in the batch.

    "state" here means extra hidden state carried over from a previous stage, e.g.:
      - MTP: state from N layers of the target model
      - Qwen3 VL (deepstack): state from N layers of the vision encoder
      - LiliCorr: context embeddings for the transformers

    Returns False if the context does not take a state embedding (MTP and LiliCorr contexts do)."""
    libllama.llama_batch_ext_set_embd_state.argtypes = [
        llama_batch_ext_p, ctypes.c_int32, llama_embd
    ]
    libllama.llama_batch_ext_set_embd_state.restype = ctypes.c_bool
    return libllama.llama_batch_ext_set_embd_state(batch, idx, embd)

def llama_batch_ext_set_output_embd(
    batch: ptr[llama_batch_ext],
    idx: int,
    value: bool
) -> bool:
    """Set whether output embeddings should be available for the token at index `idx` in the
    batch.

    NOTE: for now, this is equivalent to setting the output logits."""
    libllama.llama_batch_ext_set_output_embd.argtypes = [
        llama_batch_ext_p, ctypes.c_int32, ctypes.c_bool
    ]
    libllama.llama_batch_ext_set_output_embd.restype = ctypes.c_bool
    return libllama.llama_batch_ext_set_output_embd(batch, idx, value)

def llama_batch_ext_set_output_logits(
    batch: ptr[llama_batch_ext],
    idx: int,
    value: bool
) -> bool:
    """Set output logits for the token at index `idx` in the batch.

    NOTE: for now, this is equivalent to setting the output embd."""
    libllama.llama_batch_ext_set_output_logits.argtypes = [
        llama_batch_ext_p, ctypes.c_int32, ctypes.c_bool
    ]
    libllama.llama_batch_ext_set_output_logits.restype = ctypes.c_bool
    return libllama.llama_batch_ext_set_output_logits(batch, idx, value)

def llama_batch_ext_set_pos(
    batch: ptr[llama_batch_ext],
    idx: int,
    pos: ptr[llama_pos]
) -> bool:
    """Set a custom position for the token at index `idx` in the batch.

    For M-RoPE models:
        - embedding tokens must have n_pos_per_embd positions per token, in order [t, y, x, z];
          t is also the KV cache position
        - text tokens only require one single position per token"""
    libllama.llama_batch_ext_set_pos.argtypes = [
        llama_batch_ext_p, ctypes.c_int32, ctypes.POINTER(llama_pos)
    ]
    libllama.llama_batch_ext_set_pos.restype = ctypes.c_bool
    return libllama.llama_batch_ext_set_pos(batch, idx, pos)

def llama_process(
    ctx: ptr[llama_context],
    ptype: int,
    batch: ptr[llama_batch_ext]
) -> int:
    """Process an extended batch. Return values are the same as `llama_decode()`.

    :param ptype: See `LlamaProcessType`."""
    libllama.llama_process.argtypes = [llama_context_p, ctypes.c_int, llama_batch_ext_p]
    libllama.llama_process.restype = ctypes.c_int32
    return libllama.llama_process(ctx, ptype, batch)

#
# misc. setters and getters
#

def llama_set_n_threads(
    ctx: ptr[llama_context],
    n_threads: int,
    n_threads_batch: int
) -> None:
    """Set the number of threads used for decoding"""
    libllama.llama_set_n_threads.argtypes = [llama_context_p, ctypes.c_int32, ctypes.c_int32]
    libllama.llama_set_n_threads.restype = None
    libllama.llama_set_n_threads(ctx, n_threads, n_threads_batch)

def llama_n_threads(ctx: ptr[llama_context]) -> int:
    """Get the number of threads used for generation of a single token"""
    libllama.llama_n_threads.argtypes = [llama_context_p]
    libllama.llama_n_threads.restype = ctypes.c_int32
    return libllama.llama_n_threads(ctx)

def llama_n_threads_batch(ctx: ptr[llama_context]) -> int:
    """Get the number of threads used for prompt and batch processing"""
    libllama.llama_n_threads_batch.argtypes = [llama_context_p]
    libllama.llama_n_threads_batch.restype = ctypes.c_int32
    return libllama.llama_n_threads_batch(ctx)

def llama_set_embeddings(ctx: ptr[llama_context], embeddings: bool) -> None:
    """Set whether the context outputs embeddings or not"""
    libllama.llama_set_embeddings.argtypes = [llama_context_p, ctypes.c_bool]
    libllama.llama_set_embeddings.restype = None
    libllama.llama_set_embeddings(ctx, embeddings)

def llama_set_causal_attn(ctx: ptr[llama_context], causal_attn: bool) -> None:
    """Set whether to use causal attention or not.

    If set to True, the model will only attend to the past tokens."""
    libllama.llama_set_causal_attn.argtypes = [llama_context_p, ctypes.c_bool]
    libllama.llama_set_causal_attn.restype = None
    libllama.llama_set_causal_attn(ctx, causal_attn)

def llama_get_causal_attn(ctx: ptr[llama_context]) -> bool:
    """Returns whether the context is currently using causal attention"""
    libllama.llama_get_causal_attn.argtypes = [llama_context_p]
    libllama.llama_get_causal_attn.restype = ctypes.c_bool
    return libllama.llama_get_causal_attn(ctx)

def llama_set_abort_callback(
    ctx: ptr[llama_context],
    abort_callback: ctypes._CFunctionType, # abort_callback_functype
    abort_callback_data: ptr
) -> None:
    """Set an abort callback. If it returns True, execution of `llama_decode()` will be aborted.

    Currently works only with CPU execution."""
    libllama.llama_set_abort_callback.argtypes = [
        llama_context_p, abort_callback_functype, ctypes.c_void_p
    ]
    libllama.llama_set_abort_callback.restype = None
    libllama.llama_set_abort_callback(ctx, abort_callback, abort_callback_data)

def llama_synchronize(ctx: ptr[llama_context]) -> None:
    """Wait until all computations are finished

    This is automatically done when using one of the functions below to obtain the computation
    results and is not necessary to call explicitly in most cases"""
    libllama.llama_synchronize.argtypes = [llama_context_p]
    libllama.llama_synchronize.restype = None
    libllama.llama_synchronize(ctx)

#
# getters for logits and embeddings
#
# NOTE: `llama_get_logits` and `llama_get_embeddings` may eventually be deprecated in favor of
#       `llama_get_logits_ith()` / `llama_get_embeddings_ith()`
#
#       ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522

def llama_get_logits(ctx: ptr[llama_context]) -> ptr[ctypes.c_float]:
    """Get the token logits obtained from the last call to `llama_decode()`

    The logits for which `llama_batch.logits[i] != 0` are stored contiguously in the order they
    have appeared in the batch.

    Rows: number of tokens for which `llama_batch.logits[i] != 0`
    Cols: n_vocab"""
    libllama.llama_get_logits.argtypes = [llama_context_p]
    libllama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_logits(ctx)

def llama_get_logits_ith(ctx: ptr[llama_context], i: int) -> ptr[ctypes.c_float]:
    """Get the logits for the ith token.

    For positive indices, equivalent to `llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab`.
    Negative indices can be used to access logits in reverse order, -1 is the last logit.
    Returns None for invalid indices."""
    libllama.llama_get_logits_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_logits_ith(ctx, i)

def llama_get_embeddings(ctx: ptr[llama_context]) -> ptr[ctypes.c_float]:
    """Get all output token embeddings.

    When `pooling_type == LLAMA_POOLING_TYPE_NONE` or when using a generative model, the
    embeddings for which `llama_batch.logits[i] != 0` are stored contiguously in the order they
    have appeared in the batch. Shape: [n_outputs*n_embd]. Otherwise, returns None."""
    libllama.llama_get_embeddings.argtypes = [llama_context_p]
    libllama.llama_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_embeddings(ctx)

def llama_get_embeddings_ith(ctx: ptr[llama_context], i: int) -> ptr[ctypes.c_float]:
    """Get the embeddings for the ith token.

    For positive indices, equivalent to `llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd`.
    Negative indices can be used to access embeddings in reverse order, -1 is the last
    embedding. Shape: [n_embd] (1-dimensional). Returns None for invalid indices."""
    libllama.llama_get_embeddings_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_embeddings_ith.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_embeddings_ith(ctx, i)

def llama_get_embeddings_seq(ctx: ptr[llama_context], seq_id: llama_seq_id) -> ptr[ctypes.c_float]:
    """Get the embeddings for a sequence id.

    Returns None if `pooling_type` is `LLAMA_POOLING_TYPE_NONE`.
    When `pooling_type == LLAMA_POOLING_TYPE_RANK`, returns float[n_cls_out] with the rank(s) of
    the sequence. Otherwise: float[n_embd] (1-dimensional)."""
    libllama.llama_get_embeddings_seq.argtypes = [llama_context_p, llama_seq_id]
    libllama.llama_get_embeddings_seq.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_embeddings_seq(ctx, seq_id)

#
# [EXPERIMENTAL] backend sampling
#
# NOTE: use only if the llama_context was created with at least one llama_sampler_seq_config
#       (ref: `llama_context_params.samplers` and `llama_context_params.n_samplers`).
#

def llama_get_sampled_token_ith(ctx: ptr[llama_context], i: int) -> int:
    """Get the backend sampled token for the ith token.

    With multiple outputs, sampler state advances when the token is accepted, not when it is
    read through this function. When accepting multiple outputs, accept a contiguous prefix in
    output order. Returns LLAMA_TOKEN_NULL if no token was sampled."""
    libllama.llama_get_sampled_token_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_sampled_token_ith.restype = llama_token
    return libllama.llama_get_sampled_token_ith(ctx, i)

def llama_get_sampled_probs_ith(ctx: ptr[llama_context], i: int) -> ptr[ctypes.c_float]:
    """Get the backend sampled probabilities for the ith token.

    The index matches `llama_get_sampled_token_ith()`. Returns None if no probabilities were
    generated."""
    libllama.llama_get_sampled_probs_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_sampled_probs_ith.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_sampled_probs_ith(ctx, i)

def llama_get_sampled_probs_count_ith(ctx: ptr[llama_context], i: int) -> int:
    """Get the number of backend sampled probabilities for the ith token"""
    libllama.llama_get_sampled_probs_count_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_sampled_probs_count_ith.restype = ctypes.c_uint32
    return libllama.llama_get_sampled_probs_count_ith(ctx, i)

def llama_get_sampled_logits_ith(ctx: ptr[llama_context], i: int) -> ptr[ctypes.c_float]:
    """Get the backend sampled logits for the ith token.

    Returns None if no logits were sampled."""
    libllama.llama_get_sampled_logits_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_sampled_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_sampled_logits_ith(ctx, i)

def llama_get_sampled_logits_count_ith(ctx: ptr[llama_context], i: int) -> int:
    """Get the number of backend sampled logits for the ith token"""
    libllama.llama_get_sampled_logits_count_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_sampled_logits_count_ith.restype = ctypes.c_uint32
    return libllama.llama_get_sampled_logits_count_ith(ctx, i)

def llama_get_sampled_candidates_ith(ctx: ptr[llama_context], i: int) -> ptr[llama_token]:
    """Get the backend sampled candidates (token ids) for the ith token.

    These are needed to map probability/logit indices to vocab token ids. Returns None if no
    candidates were sampled."""
    libllama.llama_get_sampled_candidates_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_sampled_candidates_ith.restype = ctypes.POINTER(llama_token)
    return libllama.llama_get_sampled_candidates_ith(ctx, i)

def llama_get_sampled_candidates_count_ith(ctx: ptr[llama_context], i: int) -> int:
    """Get the number of backend sampled candidates (token ids) for the ith token"""
    libllama.llama_get_sampled_candidates_count_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_sampled_candidates_count_ith.restype = ctypes.c_uint32
    return libllama.llama_get_sampled_candidates_count_ith(ctx, i)

#
# llama_vocab
#

def llama_vocab_get_text(vocab: ptr[llama_vocab], token: int) -> bytes:
    """Get the text representation of a token (as bytes)"""
    libllama.llama_vocab_get_text.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_get_text.restype = ctypes.c_char_p
    return libllama.llama_vocab_get_text(vocab, token)

def llama_vocab_get_score(vocab: ptr[llama_vocab], token: int) -> float:
    """Get the score of a token"""
    libllama.llama_vocab_get_score.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_get_score.restype = ctypes.c_float
    return libllama.llama_vocab_get_score(vocab, token)

def llama_vocab_get_attr(vocab: ptr[llama_vocab], token: int) -> int:
    """Get the attributes of a token. See `LlamaTokenAttr`."""
    libllama.llama_vocab_get_attr.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_get_attr.restype = ctypes.c_int # LlamaTokenAttr
    return libllama.llama_vocab_get_attr(vocab, token)

def llama_vocab_is_eog(vocab: ptr[llama_vocab], token: int) -> bool:
    """Check if the token is supposed to end generation (end-of-generation, eg. EOS,
    EOT, etc.)"""
    libllama.llama_vocab_is_eog.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_is_eog.restype = ctypes.c_bool
    return libllama.llama_vocab_is_eog(vocab, token)

def llama_vocab_is_control(vocab: ptr[llama_vocab], token: int) -> bool:
    """Identify if token ID is a control token or a render-able token"""
    libllama.llama_vocab_is_control.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_is_control.restype = ctypes.c_bool
    return libllama.llama_vocab_is_control(vocab, token)

def llama_vocab_bos(vocab: ptr[llama_vocab]) -> int:
    """Get the BOS token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_bos.argtypes = [llama_vocab_p]
    libllama.llama_vocab_bos.restype = llama_token
    return libllama.llama_vocab_bos(vocab)

def llama_vocab_eos(vocab: ptr[llama_vocab]) -> int:
    """Get the EOS token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_eos.argtypes = [llama_vocab_p]
    libllama.llama_vocab_eos.restype = llama_token
    return libllama.llama_vocab_eos(vocab)

def llama_vocab_eot(vocab: ptr[llama_vocab]) -> int:
    """Get the end-of-turn token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_eot.argtypes = [llama_vocab_p]
    libllama.llama_vocab_eot.restype = llama_token
    return libllama.llama_vocab_eot(vocab)

def llama_vocab_sep(vocab: ptr[llama_vocab]) -> int:
    """Get the sentence separator token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_sep.argtypes = [llama_vocab_p]
    libllama.llama_vocab_sep.restype = llama_token
    return libllama.llama_vocab_sep(vocab)

def llama_vocab_nl(vocab: ptr[llama_vocab]) -> int:
    """Get the newline token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_nl.argtypes = [llama_vocab_p]
    libllama.llama_vocab_nl.restype = llama_token
    return libllama.llama_vocab_nl(vocab)

def llama_vocab_pad(vocab: ptr[llama_vocab]) -> int:
    """Get the padding token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_pad.argtypes = [llama_vocab_p]
    libllama.llama_vocab_pad.restype = llama_token
    return libllama.llama_vocab_pad(vocab)

def llama_vocab_mask(vocab: ptr[llama_vocab]) -> int:
    """Get the mask token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_mask.argtypes = [llama_vocab_p]
    libllama.llama_vocab_mask.restype = llama_token
    return libllama.llama_vocab_mask(vocab)

def llama_vocab_get_add_bos(vocab: ptr[llama_vocab]) -> bool:
    """Whether BOS token should be added to tokenizations."""
    libllama.llama_vocab_get_add_bos.argtypes = [llama_vocab_p]
    libllama.llama_vocab_get_add_bos.restype = ctypes.c_bool
    return libllama.llama_vocab_get_add_bos(vocab)

def llama_vocab_get_add_eos(vocab: ptr[llama_vocab]) -> bool:
    """Whether EOS token should be added to tokenizations."""
    libllama.llama_vocab_get_add_eos.argtypes = [llama_vocab_p]
    libllama.llama_vocab_get_add_eos.restype = ctypes.c_bool
    return libllama.llama_vocab_get_add_eos(vocab)

def llama_vocab_get_add_sep(vocab: ptr[llama_vocab]) -> bool: # is this only for classifier models?
    """Whether SEP token should be added to tokenizations."""
    libllama.llama_vocab_get_add_sep.argtypes = [llama_vocab_p]
    libllama.llama_vocab_get_add_sep.restype = ctypes.c_bool
    return libllama.llama_vocab_get_add_sep(vocab)

def llama_vocab_get_suppress_tokens(vocab: ptr[llama_vocab]) -> tuple[int, list[int]]:
    """Get the model-specific suppress tokens (gguf key: tokenizer.ggml.suppress_tokens).

    Returns a `(n_suppress_tokens, tokens)` tuple. `n_suppress_tokens` is 0 and `tokens` is
    empty if the model defines none."""
    n_suppress_tokens = ctypes.c_int32(0)
    libllama.llama_vocab_get_suppress_tokens.argtypes = [
        llama_vocab_p, ctypes.POINTER(ctypes.c_int32)
    ]
    libllama.llama_vocab_get_suppress_tokens.restype = ctypes.POINTER(llama_token)
    c_tokens = libllama.llama_vocab_get_suppress_tokens(
        vocab, ctypes.byref(n_suppress_tokens)
    )
    n = n_suppress_tokens.value
    if not c_tokens or n <= 0:
        return (0, [])
    return (n, [c_tokens[i] for i in range(n)])

def llama_vocab_fim_pre(vocab: ptr[llama_vocab]) -> int:
    """Get the infill prefix token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_fim_pre.argtypes = [llama_vocab_p]
    libllama.llama_vocab_fim_pre.restype = llama_token
    return libllama.llama_vocab_fim_pre(vocab)

def llama_vocab_fim_suf(vocab: ptr[llama_vocab]) -> int:
    """Get the infill suffix token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_fim_suf.argtypes = [llama_vocab_p]
    libllama.llama_vocab_fim_suf.restype = llama_token
    return libllama.llama_vocab_fim_suf(vocab)

def llama_vocab_fim_mid(vocab: ptr[llama_vocab]) -> int:
    """Get the infill middle token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_fim_mid.argtypes = [llama_vocab_p]
    libllama.llama_vocab_fim_mid.restype = llama_token
    return libllama.llama_vocab_fim_mid(vocab)

def llama_vocab_fim_pad(vocab: ptr[llama_vocab]) -> int:
    """Get the infill pad token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_fim_pad.argtypes = [llama_vocab_p]
    libllama.llama_vocab_fim_pad.restype = llama_token
    return libllama.llama_vocab_fim_pad(vocab)

def llama_vocab_fim_rep(vocab: ptr[llama_vocab]) -> int:
    """Get the infill repo token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_fim_rep.argtypes = [llama_vocab_p]
    libllama.llama_vocab_fim_rep.restype = llama_token
    return libllama.llama_vocab_fim_rep(vocab)

def llama_vocab_fim_sep(vocab: ptr[llama_vocab]) -> int:
    """Get the infill separator token ID. Returns the value of `LLAMA_TOKEN_NULL` if not found."""
    libllama.llama_vocab_fim_sep.argtypes = [llama_vocab_p]
    libllama.llama_vocab_fim_sep.restype = llama_token
    return libllama.llama_vocab_fim_sep(vocab)

#
# tokenization etc.
#
# NOTE: the tokenization API is thread-safe.

def llama_tokenize(
    vocab: ptr[llama_vocab],
    text: bytes,
    text_len: int,
    tokens: ptr[llama_token],
    n_tokens_max: int,
    add_special: bool,
    parse_special: bool
) -> int:
    """Convert the provided text into tokens

    - vocab:
        The llama_vocab to use.
    - text:
        The text to convert (as bytes)
    - text_len:
        The length of the text in bytes
    - tokens:
        The tokens pointer must be large enough to hold the resulting tokens.
    - n_tokens_max:
        Maximum number of tokens to return (fail if text is too long)
    - add_special:
        Allow to add BOS and EOS tokens if model is configured to do so.
    - parse_special:
        Allow tokenizing special and/or control tokens which otherwise are not
        exposed and treated as plaintext. Does not insert a leading space.

    Returns the number of tokens on success, no more than n_tokens_max. Returns
    a negative number on failure - the number of tokens that would have been
    returned.

    Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)."""
    libllama.llama_tokenize.argtypes = [
        llama_vocab_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(llama_token),
        ctypes.c_int32, ctypes.c_bool, ctypes.c_bool
    ]
    libllama.llama_tokenize.restype = ctypes.c_int32
    return libllama.llama_tokenize(
        vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special
    )

def llama_token_to_piece(
    vocab: ptr[llama_vocab],
    token: int,
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    length: int,
    lstrip: int,
    special: bool
) -> int:
    """Convert a single token to a piece of text.

    Does not write a null terminator to the buffer. The user can skip up to `lstrip` leading
    spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix').

    If `special` is True, special tokens are rendered in the output."""
    libllama.llama_token_to_piece.argtypes = [
        llama_vocab_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
    ]
    libllama.llama_token_to_piece.restype = ctypes.c_int32
    return libllama.llama_token_to_piece(vocab, token, buf, length, lstrip, special)

def llama_detokenize(
    vocab: ptr[llama_vocab],
    tokens: ptr[llama_token],
    n_tokens: int,
    text: ctypes.Array, # created via ctypes.create_string_buffer()
    text_len_max: int,
    remove_special: bool,
    unparse_special: bool
) -> int:
    """Convert the provided tokens into text (inverse of `llama_tokenize()`).

    - Returns the number of chars/bytes on success, no more than text_len_max.
    - Returns a negative number on failure - the number of chars/bytes that would have been
      returned.
    - remove_special: allow to remove BOS and EOS tokens if model is configured to do so.
    - unparse_special: if True, special tokens are rendered in the output."""
    libllama.llama_detokenize.argtypes = [
        llama_vocab_p, ctypes.POINTER(llama_token), ctypes.c_int32, ctypes.c_char_p,
        ctypes.c_int32, ctypes.c_bool, ctypes.c_bool
    ]
    libllama.llama_detokenize.restype = ctypes.c_int32
    return libllama.llama_detokenize(
        vocab, tokens, n_tokens, text, text_len_max, remove_special, unparse_special
    )

#
# chat templating
#
# XXX: do we really need this? use jinja.

def llama_chat_apply_template(
    tmpl: Optional[str],
    chat: ptr[llama_chat_message],
    n_msg: int,
    add_ass: bool,
    buf: ctypes.Array, # created via ctypes.create_string_buffer()
    length: int
) -> int:
    """Apply a chat template.

    NOTE: this function does not use a jinja parser. It only supports a pre-defined list of
    templates. See:
    https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template

    :param tmpl: A Jinja template to use for this chat. Pass None to use the model's default
        chat template.
    :param chat: Pointer to a list of multiple llama_chat_message
    :param n_msg: Number of llama_chat_message in this chat
    :param add_ass: Whether to end the prompt with the token(s) that indicate the start of an
        assistant message.
    :param buf: A buffer to hold the output formatted prompt. The recommended alloc size is
        2 * (total number of characters of all messages)
    :param length: The size of the allocated buffer
    :return: The total number of bytes of the formatted prompt. If it is larger than the size of
        the buffer, you may need to re-alloc it and then re-apply the template."""
    libllama.llama_chat_apply_template.argtypes = [
        ctypes.c_char_p, llama_chat_message_p, ctypes.c_size_t, ctypes.c_bool, ctypes.c_char_p,
        ctypes.c_int32
    ]
    libllama.llama_chat_apply_template.restype = ctypes.c_int32
    c_tmpl = tmpl.encode('utf-8') if tmpl is not None else None
    return libllama.llama_chat_apply_template(c_tmpl, chat, n_msg, add_ass, buf, length)

def llama_chat_builtin_templates(output: ptr[ctypes.c_char_p], n: int) -> int:
    """Get the list of built-in chat templates.

    :param output: Buffer of at least `n` `const char *` entries to receive the template names
    :param n: The number of entries available in `output`
    :return: The number of built-in templates (may exceed `n`, in which case `output` was
        filled only partially)"""
    libllama.llama_chat_builtin_templates.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
    libllama.llama_chat_builtin_templates.restype = ctypes.c_int32
    return libllama.llama_chat_builtin_templates(output, n)

#
# llama_sampler
#

def llama_set_sampler(
    ctx: ptr[llama_context],
    seq_id: llama_seq_id,
    smpl: ptr[llama_sampler]
) -> bool:
    """[EXPERIMENTAL] Attach a sampler to the context.

    NOTE: prefer initializing the context with `llama_context_params.samplers` when possible."""
    libllama.llama_set_sampler.argtypes = [llama_context_p, llama_seq_id, llama_sampler_p]
    libllama.llama_set_sampler.restype = ctypes.c_bool
    return libllama.llama_set_sampler(ctx, seq_id, smpl)

def llama_sampler_init(
    iface: ptr[llama_sampler_i],
    ctx: llama_sampler_context_t
) -> ptr[llama_sampler]:
    """Create a sampler from a `llama_sampler_i` interface and a context pointer.

    IMPORTANT: the caller must keep both `iface` (the `llama_sampler_i` instance) and any
    `ctypes.CFUNCTYPE` callback objects assigned into it alive for as long as the returned
    sampler is in use. ctypes frees the C trampolines when the Python objects are collected,
    and libllama holds a raw pointer to them -- calling through a collected callback is a hard
    crash, not an exception."""
    libllama.llama_sampler_init.argtypes = [llama_sampler_i_p, llama_sampler_context_t]
    libllama.llama_sampler_init.restype = llama_sampler_p
    return libllama.llama_sampler_init(iface, ctx)

def llama_sampler_name(smpl: ptr[llama_sampler]) -> str:
    """Get the name of a sampler"""
    libllama.llama_sampler_name.argtypes = [llama_sampler_p]
    libllama.llama_sampler_name.restype = ctypes.c_char_p
    return _cstr(libllama.llama_sampler_name(smpl))

def llama_sampler_accept(smpl: ptr[llama_sampler], token: llama_token) -> None:
    """Accept a token sampled by a sampler.

    Accepting a token updates the internal state of certain samplers (e.g. grammar, repetition,
    etc.)"""
    libllama.llama_sampler_accept.argtypes = [llama_sampler_p, llama_token]
    libllama.llama_sampler_accept.restype = None
    libllama.llama_sampler_accept(smpl, token)

def llama_sampler_apply(smpl: ptr[llama_sampler], cur_p: ptr[llama_token_data_array]) -> None:
    """Apply a sampler to a token data array (pass `ctypes.byref(...)` for `cur_p`)"""
    libllama.llama_sampler_apply.argtypes = [llama_sampler_p, llama_token_data_array_p]
    libllama.llama_sampler_apply.restype = None
    libllama.llama_sampler_apply(smpl, cur_p)

def llama_sampler_reset(smpl: ptr[llama_sampler]) -> None:
    """Reset a sampler"""
    libllama.llama_sampler_reset.argtypes = [llama_sampler_p]
    libllama.llama_sampler_reset.restype = None
    libllama.llama_sampler_reset(smpl)

def llama_sampler_clone(smpl: ptr[llama_sampler]) -> ptr[llama_sampler]:
    """Clone a sampler"""
    libllama.llama_sampler_clone.argtypes = [llama_sampler_p]
    libllama.llama_sampler_clone.restype = llama_sampler_p
    return libllama.llama_sampler_clone(smpl)

def llama_sampler_copy(src: ptr[llama_sampler], dst: ptr[llama_sampler]) -> None:
    """Copy mutable state from `src` into `dst` while keeping `dst`'s references to the current
    sampling graph.

    `src` and `dst` must have the same type and configuration."""
    libllama.llama_sampler_copy.argtypes = [llama_sampler_p, llama_sampler_p]
    libllama.llama_sampler_copy.restype = None
    libllama.llama_sampler_copy(src, dst)

def llama_sampler_free(smpl: ptr[llama_sampler]) -> None:
    """Free a sampler

    IMPORTANT: do not free if the sampler has been added to a llama_sampler_chain (via
    `llama_sampler_chain_add`)"""
    libllama.llama_sampler_free.argtypes = [llama_sampler_p]
    libllama.llama_sampler_free.restype = None
    libllama.llama_sampler_free(smpl)

#
# llama_sampler_chain_*
#

def llama_sampler_chain_init(params: llama_sampler_chain_params) -> ptr[llama_sampler]:
    """Initialize a sampler chain

    NOTE: `params` is passed by value, matching the C signature."""
    libllama.llama_sampler_chain_init.argtypes = [llama_sampler_chain_params]
    libllama.llama_sampler_chain_init.restype = llama_sampler_p
    return libllama.llama_sampler_chain_init(params)

def llama_sampler_chain_add(chain: ptr[llama_sampler], smpl: ptr[llama_sampler]) -> None:
    """Add a sampler to a sampler chain

    IMPORTANT: takes ownership of the sampler object and will free it when `llama_sampler_free`
    is called"""
    libllama.llama_sampler_chain_add.argtypes = [llama_sampler_p, llama_sampler_p]
    libllama.llama_sampler_chain_add.restype = None
    libllama.llama_sampler_chain_add(chain, smpl)

def llama_sampler_chain_get(chain: ptr[llama_sampler], i: int) -> ptr[llama_sampler]:
    """Get a sampler from a sampler chain.

    Returns None if:
      - the sampler is NULL
      - the sampler is not a llama_sampler_chain
      - the index is out of bounds, unless i == -1
      - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)"""
    libllama.llama_sampler_chain_get.argtypes = [llama_sampler_p, ctypes.c_int32]
    libllama.llama_sampler_chain_get.restype = llama_sampler_p
    return libllama.llama_sampler_chain_get(chain, i)

def llama_sampler_chain_n(chain: ptr[llama_sampler]) -> int:
    """Get the number of samplers in a sampler chain"""
    libllama.llama_sampler_chain_n.argtypes = [llama_sampler_p]
    libllama.llama_sampler_chain_n.restype = ctypes.c_int32
    return libllama.llama_sampler_chain_n(chain)

def llama_sampler_chain_remove(chain: ptr[llama_sampler], i: int) -> ptr[llama_sampler]:
    """Remove a sampler from a sampler chain

    After removing a sampler, the chain will no longer own it, and it will not be freed when the
    chain is freed."""
    libllama.llama_sampler_chain_remove.argtypes = [llama_sampler_p, ctypes.c_int32]
    libllama.llama_sampler_chain_remove.restype = llama_sampler_p
    return libllama.llama_sampler_chain_remove(chain, i)

#
# llama_sampler cont.
#

def llama_sampler_init_greedy() -> ptr[llama_sampler]:
    """Initialize a greedy sampler (always picks the most likely token)"""
    libllama.llama_sampler_init_greedy.argtypes = []
    libllama.llama_sampler_init_greedy.restype = llama_sampler_p
    return libllama.llama_sampler_init_greedy()

def llama_sampler_init_dist(seed: int) -> ptr[llama_sampler]:
    """Initialize a distribution sampler

    Pass `seed == LLAMA_DEFAULT_SEED` to use a random seed."""
    libllama.llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
    libllama.llama_sampler_init_dist.restype = llama_sampler_p
    return libllama.llama_sampler_init_dist(seed)

def llama_sampler_init_top_k(k: int) -> ptr[llama_sampler]:
    """Initialize a top-K sampler. Setting k <= 0 makes this a no-op.

    Top-K sampling as described in "The Curious Case of Neural Text Degeneration"
    https://arxiv.org/abs/1904.09751"""
    libllama.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    libllama.llama_sampler_init_top_k.restype = llama_sampler_p
    return libllama.llama_sampler_init_top_k(k)

def llama_sampler_init_top_p(p: float, min_keep: int) -> ptr[llama_sampler]:
    """Initialize a top-p (nucleus) sampler

    Nucleus sampling as described in "The Curious Case of Neural Text Degeneration"
    https://arxiv.org/abs/1904.09751"""
    libllama.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    libllama.llama_sampler_init_top_p.restype = llama_sampler_p
    return libllama.llama_sampler_init_top_p(p, min_keep)

def llama_sampler_init_min_p(p: float, min_keep: int) -> ptr[llama_sampler]:
    """Initialize a min-p sampler

    Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841"""
    libllama.llama_sampler_init_min_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    libllama.llama_sampler_init_min_p.restype = llama_sampler_p
    return libllama.llama_sampler_init_min_p(p, min_keep)

def llama_sampler_init_typical(p: float, min_keep: int) -> ptr[llama_sampler]:
    """Initialize a locally typical sampler

    Locally Typical Sampling as described in https://arxiv.org/abs/2202.00666"""
    libllama.llama_sampler_init_typical.argtypes = [ctypes.c_float, ctypes.c_size_t]
    libllama.llama_sampler_init_typical.restype = llama_sampler_p
    return libllama.llama_sampler_init_typical(p, min_keep)

def llama_sampler_init_temp(t: float) -> ptr[llama_sampler]:
    """Initialize a temperature sampler

    Updates the logits `l_i = l_i / t`. When `t` <= 0.0, the maximum logit is kept at its
    original value, the rest are set to -inf"""
    libllama.llama_sampler_init_temp.argtypes = [ctypes.c_float]
    libllama.llama_sampler_init_temp.restype = llama_sampler_p
    return libllama.llama_sampler_init_temp(t)

def llama_sampler_init_temp_ext(t: float, delta: float, exponent: float) -> ptr[llama_sampler]:
    """Initialize a dynamic temperature / entropy sampler

    Dynamic temperature implementation (a.k.a. entropy) as described in
    https://arxiv.org/abs/2309.02772"""
    libllama.llama_sampler_init_temp_ext.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    libllama.llama_sampler_init_temp_ext.restype = llama_sampler_p
    return libllama.llama_sampler_init_temp_ext(t, delta, exponent)

def llama_sampler_init_xtc(p: float, t: float, min_keep: int, seed: int) -> ptr[llama_sampler]:
    """Initialize an XTC sampler

    XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335"""
    libllama.llama_sampler_init_xtc.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_size_t, ctypes.c_uint32
    ]
    libllama.llama_sampler_init_xtc.restype = llama_sampler_p
    return libllama.llama_sampler_init_xtc(p, t, min_keep, seed)

def llama_sampler_init_top_n_sigma(n: float) -> ptr[llama_sampler]:
    """Initialize a top-n-sigma sampler

    Top-n-sigma sampling as described in
    [2025.acl-long.528](https://aclanthology.org/2025.acl-long.528/)

    (this reference is better than the one in llama.h, which is
    ["Top-nσ: Not All Logits Are You Need"](https://arxiv.org/pdf/2411.07641))"""
    libllama.llama_sampler_init_top_n_sigma.argtypes = [ctypes.c_float]
    libllama.llama_sampler_init_top_n_sigma.restype = llama_sampler_p
    return libllama.llama_sampler_init_top_n_sigma(n)

def llama_sampler_init_mirostat(
    n_vocab: int,
    seed: int,
    tau: float,
    eta: float,
    m: int
) -> ptr[llama_sampler]:
    """Initialize a Mirostat v1 sampler

    Mirostat 1.0 as described in https://arxiv.org/abs/2007.14966. Uses tokens instead of words.

    :param tau: The target cross-entropy (or surprise) value you want to achieve for the
        generated text. A higher value corresponds to more surprising or less predictable text,
        while a lower value corresponds to less surprising or more predictable text.
    :param eta: The learning rate used to update `mu` based on the error between the target and
        observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
        updated more quickly, while a smaller learning rate will result in slower updates.
    :param m: The number of tokens considered in the estimation of `s_hat`. This is an arbitrary
        value that is used to calculate `s_hat`, which in turn helps to calculate the value of
        `k`. In the paper, they use `m = 100`, but you can experiment with different values to
        see how it affects the performance of the algorithm.
    :param mu: Maximum cross-entropy. This value is initialized to be twice the target
        cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the
        target and observed surprisal."""
    libllama.llama_sampler_init_mirostat.argtypes = [
        ctypes.c_int32, ctypes.c_uint32, ctypes.c_float, ctypes.c_float, ctypes.c_int32
    ]
    libllama.llama_sampler_init_mirostat.restype = llama_sampler_p
    return libllama.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)

def llama_sampler_init_mirostat_v2(seed: int, tau: float, eta: float) -> ptr[llama_sampler]:
    """Initialize a Mirostat v2 sampler

    Mirostat 2.0 as described in https://arxiv.org/abs/2007.14966. Uses tokens instead of words.

    :param tau: The target cross-entropy (or surprise) value you want to achieve for the
        generated text. A higher value corresponds to more surprising or less predictable text,
        while a lower value corresponds to less surprising or more predictable text.
    :param eta: The learning rate used to update `mu` based on the error between the target and
        observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
        updated more quickly, while a smaller learning rate will result in slower updates."""
    libllama.llama_sampler_init_mirostat_v2.argtypes = [
        ctypes.c_uint32, ctypes.c_float, ctypes.c_float
    ]
    libllama.llama_sampler_init_mirostat_v2.restype = llama_sampler_p
    return libllama.llama_sampler_init_mirostat_v2(seed, tau, eta)

def llama_sampler_init_grammar(
    vocab: ptr[llama_vocab],
    grammar_str: str,
    grammar_root: str
) -> ptr[llama_sampler]:
    """Initialize a GBNF grammar. See grammars/README.md in llama.cpp for details.

    :param vocab: The vocabulary that this grammar will be used with.
    :param grammar_str: The production rules for the grammar, encoded as a string. Returns an
        empty grammar if empty. Returns None if parsing of grammar_str fails.
    :param grammar_root: The name of the start symbol for the grammar."""
    libllama.llama_sampler_init_grammar.argtypes = [
        llama_vocab_p, ctypes.c_char_p, ctypes.c_char_p
    ]
    libllama.llama_sampler_init_grammar.restype = llama_sampler_p
    return libllama.llama_sampler_init_grammar(
        vocab, grammar_str.encode('utf-8'), grammar_root.encode('utf-8')
    )

def llama_sampler_init_grammar_lazy_patterns(
    vocab: ptr[llama_vocab],
    grammar_str: str,
    grammar_root: str,
    trigger_patterns: list[str],
    num_trigger_patterns: int,
    trigger_tokens: ptr[llama_token],
    num_trigger_tokens: int
) -> ptr[llama_sampler]:
    """Initialize a lazy grammar sampler, introduced in
    https://github.com/ggml-org/llama.cpp/pull/9639

    :param trigger_patterns:
        A list of patterns that will trigger the grammar sampler.
        Pattern will be matched from the start of the generation output, and grammar sampler will be
        fed content starting from its first match group.
    :param trigger_tokens:
        A list of tokens that will trigger the grammar sampler.
        Grammar sampler will be fed content starting from the trigger token included.

    NOTE: `num_trigger_patterns` is ignored; the actual count is derived from
          `len(trigger_patterns)` so that the ctypes array size always matches the C side.
          The parameter is kept for signature compatibility."""
    libllama.llama_sampler_init_grammar_lazy_patterns.argtypes = [
        llama_vocab_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t, ctypes.POINTER(llama_token), ctypes.c_size_t
    ]
    libllama.llama_sampler_init_grammar_lazy_patterns.restype = llama_sampler_p
    num_trigger_patterns = len(trigger_patterns)
    c_trigger_patterns = (ctypes.c_char_p * num_trigger_patterns)(
        *[pattern.encode('utf-8') for pattern in trigger_patterns]
    )
    return libllama.llama_sampler_init_grammar_lazy_patterns(
        vocab,
        grammar_str.encode('utf-8'),
        grammar_root.encode('utf-8'),
        c_trigger_patterns,
        num_trigger_patterns,
        trigger_tokens,
        num_trigger_tokens
    )

def llama_sampler_init_penalties(
    n_vocab: int,
    penalty_last_n: int,
    penalty_repeat: float,
    penalty_freq: float,
    penalty_present: float
) -> ptr[llama_sampler]:
    """Initialize a penalties sampler

    NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow.
          For example, apply top-k or top-p sampling first.

    :param n_vocab:         vocabulary size (used to size internal buffers)
    :param penalty_last_n:  last n tokens to penalize (0 = disable penalty)
    :param penalty_repeat:  must be > 0.0, 1.0 = disabled
    :param penalty_freq:    must be finite, 0.0 = disabled
    :param penalty_present: must be finite, 0.0 = disabled"""
    libllama.llama_sampler_init_penalties.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    libllama.llama_sampler_init_penalties.restype = llama_sampler_p
    return libllama.llama_sampler_init_penalties(
        n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present
    )

def llama_sampler_init_dry(
    vocab: ptr[llama_vocab],
    dry_multiplier: float,
    dry_base: float,
    dry_allowed_length: int,
    dry_penalty_last_n: int,
    seq_breakers: ptr[ctypes.c_char_p],
    num_breakers: int
) -> ptr[llama_sampler]:
    """Initialize a DRY sampler

    DRY sampler, designed by p-e-w, as described in:
    https://github.com/oobabooga/text-generation-webui/pull/5677, porting the Koboldcpp
    implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982

    NOTE: `num_breakers` must not exceed `LLAMA_DRY_MAX_SEQ_BREAKERS` (64). Duplicates are not
    counted, and processing each breaker scans the entire vocabulary, so the init cost is
    O(num_breakers x n_vocab).

    NOTE: `n_ctx_train` was removed from this function's parameters -- it is now taken from the
    vocabulary."""
    libllama.llama_sampler_init_dry.argtypes = [
        llama_vocab_p, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t
    ]
    libllama.llama_sampler_init_dry.restype = llama_sampler_p
    return libllama.llama_sampler_init_dry(
        vocab,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_penalty_last_n,
        seq_breakers,
        num_breakers
    )

def llama_sampler_init_adaptive_p(
    target: float,
    decay: float,
    seed: int
) -> ptr[llama_sampler]:
    """Initialize an adaptive-p sampler

    Adaptive-p selects tokens near a configurable target probability over time. Internally the
    sampler maintains an exponential moving average of the *original* probabilities of selected
    tokens at each sampling step and uses that EMA to compute an adapted target probability at
    each step, thus maintaining the desired target probability over time.

    Adaptive-p selects a token ID rather than just mutating candidates, so it must be last in
    the sampler chain (like mirostat, dist, greedy).

    Only mild truncation before this sampler is recommended the authors suggest applying min-p
    before adaptive-p as the only other active sampler in the chain.

    :param target: select tokens near this probability (valid range 0.0 to 1.0; negative =
        disabled)
    :param decay: EMA decay for adaptation; history is approximately 1/(1-decay) tokens (valid
        range 0.0 - 0.99)
    :param seed: RNG seed

    ref: https://github.com/ggml-org/llama.cpp/pull/17927

    addtl. ref: https://github.com/MrJackSpade/adaptive-p-docs"""
    libllama.llama_sampler_init_adaptive_p.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_uint32
    ]
    libllama.llama_sampler_init_adaptive_p.restype = llama_sampler_p
    return libllama.llama_sampler_init_adaptive_p(target, decay, seed)

def llama_sampler_init_logit_bias(
    n_vocab: int,
    n_logit_bias: int,
    logit_bias: ptr[llama_logit_bias]
) -> ptr[llama_sampler]:
    """Initialize a logit bias sampler"""
    libllama.llama_sampler_init_logit_bias.argtypes = [
        ctypes.c_int32, ctypes.c_int32, llama_logit_bias_p
    ]
    libllama.llama_sampler_init_logit_bias.restype = llama_sampler_p
    return libllama.llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias)

def llama_sampler_init_infill(vocab: ptr[llama_vocab]) -> ptr[llama_sampler]:
    """Initialize an infill sampler

    This sampler is meant to be used for fill-in-the-middle infilling. It's supposed to be used
    after top_k + top_p sampling.

    1. if the sum of the EOG probs times the number of candidates is higher than the sum of the
       other probs -> pick EOG
    2. combine probs of tokens that have the same prefix
    3. discard non-EOG tokens with low prob
    4. if no tokens are left -> pick EOT"""
    libllama.llama_sampler_init_infill.argtypes = [llama_vocab_p]
    libllama.llama_sampler_init_infill.restype = llama_sampler_p
    return libllama.llama_sampler_init_infill(vocab)

def llama_sampler_get_seed(smpl: ptr[llama_sampler]) -> int:
    """Get the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise"""
    libllama.llama_sampler_get_seed.argtypes = [llama_sampler_p]
    libllama.llama_sampler_get_seed.restype = ctypes.c_uint32
    return libllama.llama_sampler_get_seed(smpl)

def llama_sampler_sample(
    smpl: ptr[llama_sampler],
    ctx: ptr[llama_context],
    idx: int
) -> int:
    """Sample and accept a token from the idx-th output of the last evaluation.

    For multiple outputs from one sampler, call this function in output order without gaps.

    Shorthand for:
        logits = llama_get_logits_ith(ctx, idx)
        cur_p = <llama_token_data_array initialized from logits>
        llama_sampler_apply(smpl, byref(cur_p))
        token = cur_p.data[cur_p.selected].id
        llama_sampler_accept(smpl, token)
        return token

    Returns the sampled token."""
    libllama.llama_sampler_sample.argtypes = [llama_sampler_p, llama_context_p, ctypes.c_int32]
    libllama.llama_sampler_sample.restype = llama_token
    return libllama.llama_sampler_sample(smpl, ctx, idx)

#
# split GGUF handling
#

def llama_split_path(
    split_path: ctypes.Array, # created via ctypes.create_string_buffer()
    maxlen: int,
    path_prefix: str,
    split_no: int,
    split_count: int
) -> int:
    """Build a split GGUF final path for this chunk.

    Example:
        llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4)
        => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"

    Returns the split_path length."""
    libllama.llama_split_path.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32
    ]
    libllama.llama_split_path.restype = ctypes.c_int32
    return libllama.llama_split_path(
        split_path, maxlen, path_prefix.encode('utf-8'), split_no, split_count
    )

def llama_split_prefix(
    split_prefix: ctypes.Array, # created via ctypes.create_string_buffer()
    maxlen: int,
    split_path: str,
    split_no: int,
    split_count: int
) -> int:
    """Extract the path prefix from the split_path if and only if the split_no and split_count
    match.

    Example:
        llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4)
        => split_prefix = "/models/ggml-model-q4_0"

    Returns the split_prefix length."""
    libllama.llama_split_prefix.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32
    ]
    libllama.llama_split_prefix.restype = ctypes.c_int32
    return libllama.llama_split_prefix(
        split_prefix, maxlen, split_path.encode('utf-8'), split_no, split_count
    )

def llama_print_system_info() -> str:
    """Print system information to stderr and return it as a string"""
    libllama.llama_print_system_info.argtypes = []
    libllama.llama_print_system_info.restype = ctypes.c_char_p
    text = _cstr(libllama.llama_print_system_info())
    print(text, file=sys.stderr, flush=True)
    return text

#
# llama_log_*
#

def llama_log_set(
    log_callback: ctypes._CFunctionType,
    user_data: ptr
) -> None:
    """Set a callback for all future logging events.

    If this is not called, or None is supplied, everything is output on stderr.

    NOTE: the logger state is global, so these functions are NOT thread safe."""
    libllama.llama_log_set.argtypes = [log_callback_functype, ctypes.c_void_p]
    libllama.llama_log_set.restype = None
    libllama.llama_log_set(log_callback, user_data)

def llama_log_get(log_callback: ptr, user_data: ptr) -> None:
    """Get the current logging callback and its user data, via out-params.

    NOTE: the logger state is global, so these functions are NOT thread safe."""
    libllama.llama_log_get.argtypes = [ctypes.POINTER(log_callback_functype), ctypes.POINTER(ctypes.c_void_p)]
    libllama.llama_log_get.restype = None
    libllama.llama_log_get(log_callback, user_data)

# NOTE: llama_perf_* should not be used externally, so we don't include those here

# NOTE: training functions are not included

# =================================================================================================
#
# End of LLAMA_API
#
# =================================================================================================

class _internals:
    """### INTERNAL

    Helper functions used by the `llama` and `sampling` modules."""

    # this is initialized when libllama is initialized
    greedy_sampler: Optional[ptr[llama_sampler]] = None

    class LogitBiasArrayType:
        """Type hint for a `ctypes.Array[llama_logit_bias]` of arbitrary length"""

    class CharPArrayType:
        """Type hint for a `ctypes.Array[ctypes.c_char_p]` of arbitrary length"""

    class FloatArrayType:
        """Type hint for a `ctypes.Array[ctypes.c_float]` of arbitrary length"""

    class AdapterArrayType:
        """Type hint for a `ctypes.Array[llama_adapter_lora_p]` of arbitrary length"""

    #
    # Decoding
    #

    @staticmethod
    def _fill_batch(
        batch: llama_batch,
        pos: int,
        tokens: list[int],
        logits_all: bool,
        seq_id: int = 0,
    ) -> None:
        """### INTERNAL

        Fill a batch allocated by `llama_batch_init()` with single-sequence tokens.

        All tokens are assigned to the same `seq_id`, with consecutive positions starting at
        `pos`. The last token always has its `logits` flag set. If `logits_all` is True, every
        token gets the flag (needed to read logits for every position).

        NOTE: like `_fill_batch_multi`, this assumes `n_pos_per_embd == 1` and is
        therefore unsuitable for M-RoPE models. Use the `llama_batch_ext_*` API for those."""
        n_tokens = len(tokens)
        if n_tokens == 0:
            batch.n_tokens = 0
            return
        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            batch.token[i]     = tokens[i]
            batch.pos[i]       = pos + i
            batch.n_seq_id[i]  = 1
            batch.seq_id[i][0] = seq_id
            batch.logits[i]    = C_TRUE if logits_all else C_FALSE
        batch.logits[n_tokens - 1] = C_TRUE

    @staticmethod
    def _fill_batch_multi(
        batch: llama_batch,
        tokens: list[int],
        positions: list[int],
        seq_ids: list[list[int]],
        logits: Optional[list[bool]] = None,
        max_seq_per_token: int = 1,
        use_embd: bool = False,
    ) -> None:
        """### INTERNAL

        Fill a batch allocated by `llama_batch_init()` with explicit per-token data.

        Each token `i` may belong to multiple sequences (`seq_ids[i]`). The batch must have
        been allocated with `n_seq_max >= max_seq_per_token`.

        If `logits` is None, only the last token gets its `logits` flag set. If `use_embd` is
        True, the token IDs are NOT written into the batch (the caller is expected to have
        populated `batch.embd` and set `batch.token = NULL`).

        LIMITATION: M-RoPE models are not supported. This helper writes exactly one entry of
        `batch.pos` per token, which matches `n_pos_per_embd == 1` (the norm for text-only models).

        M-RoPE models (Qwen2-VL and friends) require `n_pos_per_embd` positions per token, in order
        `[t, y, x, z]` where `t` is also the KV cache position; in other words, `batch.pos` must
        contain `n_tokens * n_pos_per_embd` entries, and the `pos` array below would be misaligned
        if `n_pos_per_embd > 1`.

        Callers for M-RoPE models must build their own `llama_batch` (or preferably use the
        `llama_batch_ext_*` API, which models M-RoPE natively via `llama_batch_ext_set_pos`)."""
        n_tokens = len(tokens)
        if n_tokens == 0:
            batch.n_tokens = 0
            return
        if len(positions) != n_tokens or len(seq_ids) != n_tokens:
            raise ValueError(
                f'_fill_batch_multi: length mismatch '
                f'(tokens={n_tokens}, positions={len(positions)}, seq_ids={len(seq_ids)})'
            )
        if logits is None:
            logits = [False] * n_tokens
            logits[-1] = True
        elif len(logits) != n_tokens:
            raise ValueError(
                f'_fill_batch_multi: logits length {len(logits)} != n_tokens {n_tokens}'
            )

        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            sids = seq_ids[i]
            if not sids:
                raise ValueError(
                    f'_fill_batch_multi: token {i} has no sequence ids'
                )
            if len(sids) > max_seq_per_token:
                raise ValueError(
                    f'_fill_batch_multi: token {i} belongs to {len(sids)} sequences, '
                    f'but n_seq_max={max_seq_per_token}'
                )
            if not use_embd:
                batch.token[i] = tokens[i]
            batch.pos[i]      = positions[i]
            batch.n_seq_id[i] = len(sids)
            for j, sid in enumerate(sids):
                batch.seq_id[i][j] = sid
            batch.logits[i] = C_TRUE if logits[i] else C_FALSE

    @staticmethod
    def decode_pp(
        ctx: ptr[llama_context],
        pos: int,
        tokens: list[int],
        n_tokens: int,
        seq_id: int = 0,
    ) -> None:
        """### INTERNAL

        Decode with batch size > 1 (prompt processing) for a single sequence."""
        batch = llama_batch_init(n_tokens=n_tokens, embd=0, n_seq_max=1)
        try:
            _internals._fill_batch(
                batch, pos, tokens, logits_all=False, seq_id=seq_id
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_pp: llama_decode failed with status code {ret}')

    @staticmethod
    def decode_tg(
        ctx: ptr[llama_context],
        pos: int,
        token: int,
        seq_id: int = 0,
    ) -> None:
        """### INTERNAL

        Decode with batch size == 1 (text generation) for a single sequence."""
        batch = llama_batch_init(n_tokens=1, embd=0, n_seq_max=1)
        try:
            _internals._fill_batch(
                batch, pos, [token], logits_all=True, seq_id=seq_id
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_tg: llama_decode failed with status code {ret}')

    @staticmethod
    def decode_pp_with_logits(
        ctx: ptr[llama_context],
        pos: int,
        tokens: list[int],
        n_tokens: int,
        n_vocab: int,
        seq_id: int = 0,
    ) -> np.ndarray:
        """### INTERNAL

        Decode a prompt for a single sequence, returning logits for all tokens.

        Shape: `(n_tokens, n_vocab)`.

        NOTE: the returned array is a view into the context's logits buffer and is invalidated by
              the next `llama_decode()` call."""
        batch = llama_batch_init(n_tokens=n_tokens, embd=0, n_seq_max=1)
        try:
            _internals._fill_batch(
                batch, pos, tokens, logits_all=True, seq_id=seq_id
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_pp_with_logits: llama_decode failed with status code {ret}')
        c_logits = llama_get_logits(ctx)
        return np.ctypeslib.as_array(c_logits, shape=(n_tokens, n_vocab))

    @staticmethod
    def decode_tg_with_logits(
        ctx: ptr[llama_context],
        pos: int,
        token: int,
        n_vocab: int,
        seq_id: int = 0,
    ) -> np.ndarray:
        """### INTERNAL

        Decode a single token for a single sequence, returning the next-token logits.

        Shape: `(n_vocab,)`. Same view caveat as `decode_pp_with_logits`.

        NOTE: the returned array is a view into the context's logits buffer and is invalidated by
              the next `llama_decode()` call."""
        batch = llama_batch_init(n_tokens=1, embd=0, n_seq_max=1)
        try:
            _internals._fill_batch(
                batch, pos, [token], logits_all=True, seq_id=seq_id
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_tg_with_logits: llama_decode failed with status code {ret}')
        c_logits = llama_get_logits(ctx)
        return np.ctypeslib.as_array(c_logits, shape=(1, n_vocab))[0]

    @staticmethod
    def decode_multi_seq(
        ctx: ptr[llama_context],
        tokens: list[int],
        positions: list[int],
        seq_ids: list[list[int]],
        logits: Optional[list[bool]] = None,
    ) -> None:
        """### INTERNAL

        Low-level multi-sequence decode.

        Tokens `i` is emitted at `positions[i]` and belongs to every sequence in
        `seq_ids[i]` (so a token may be shared between sequences). This is the primitive the
        higher-level helpers below are built on.

        `n_seq_max` is derived from `max(len(s) for s in seq_ids)`, so any `seq_id` valid in
        the context's `n_seq_max` budget is accepted.

        NOTE: total tokens per sequence must fit in `llama_n_ctx_seq(ctx)`; if you overrun it
        `llama_decode` returns 1 ("could not find a KV slot")."""
        n_tokens = len(tokens)
        if n_tokens == 0:
            return
        max_seq = max(len(s) for s in seq_ids) if seq_ids else 1
        batch = llama_batch_init(n_tokens=n_tokens, embd=0, n_seq_max=max_seq)
        try:
            _internals._fill_batch_multi(
                batch, tokens, positions, seq_ids, logits,
                max_seq_per_token=max_seq,
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_multi_seq: llama_decode failed with status code {ret}')

    @staticmethod
    def decode_pp_multi_seq(
        ctx: ptr[llama_context],
        seqs: list[tuple[int, list[int], int]],
    ) -> None:
        """### INTERNAL

        Decode multiple independent sequences' prompts in a single batch.

        Each item is `(seq_id, tokens, start_pos)`. Each token belongs to exactly one
        sequence. Logits are enabled on the last token of each non-empty sequence, in input
        order — i.e. `llama_get_logits_ith(ctx, k)` gives the next-token logits for the
        `k`-th sequence that has at least one token.

        For shared prefixes, use `decode_shared_prefix_multi_seq` instead. Sequences must use
        distinct `seq_id` values (the KV cache keys on `seq_id`)."""
        if not seqs:
            return

        all_tokens: list[int] = []
        all_positions: list[int] = []
        all_seq_ids: list[list[int]] = []
        logits: list[bool] = []

        for sid, toks, start_pos in seqs:
            for i, t in enumerate(toks):
                all_tokens.append(t)
                all_positions.append(start_pos + i)
                all_seq_ids.append([sid])
                logits.append(i == len(toks) - 1)

        if not all_tokens:
            return

        batch = llama_batch_init(n_tokens=len(all_tokens), embd=0, n_seq_max=1)
        try:
            _internals._fill_batch_multi(
                batch, all_tokens, all_positions, all_seq_ids, logits,
                max_seq_per_token=1,
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_pp_multi_seq: llama_decode failed with status code {ret}')

    @staticmethod
    def decode_tg_multi_seq(
        ctx: ptr[llama_context],
        items: list[tuple[int, int, int]],
    ) -> None:
        """### INTERNAL

        Generate one token per sequence in a single batch (parallel text generation).

        Each item is `(seq_id, token, pos)`. All tokens get their `logits` flag set, so
        `llama_get_logits_ith(ctx, i)` returns the next-token logits for item `i` in
        `items`, in order."""
        if not items:
            return

        tokens    = [t for _, t, _ in items]
        positions = [p for _, _, p in items]
        seq_ids   = [[sid] for sid, _, _ in items]
        logits    = [True] * len(items)

        batch = llama_batch_init(n_tokens=len(items), embd=0, n_seq_max=1)
        try:
            _internals._fill_batch_multi(
                batch, tokens, positions, seq_ids, logits,
                max_seq_per_token=1,
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_tg_multi_seq: llama_decode failed with status code {ret}')

    @staticmethod
    def decode_shared_prefix_multi_seq(
        ctx: ptr[llama_context],
        prefix_tokens: list[int],
        prefix_pos: int,
        continuations: list[tuple[int, list[int]]],
    ) -> None:
        """### INTERNAL

        Decode a shared prefix plus per-sequence continuations in a single batch.

        Every prefix token belongs to ALL sequences in `continuations`, so the prefix's KV
        entries are computed exactly once and shared across sequences. Each continuation
        token belongs only to its own sequence; continuation positions start right after the
        prefix (`prefix_pos + len(prefix_tokens)`).

        Logits are enabled on the last token of each non-empty continuation, in continuation
        order (empty continuations are skipped)."""
        if not continuations:
            return

        prefix_seq_ids = [sid for sid, _ in continuations]

        all_tokens: list[int] = []
        all_positions: list[int] = []
        all_seq_ids: list[list[int]] = []
        logits: list[bool] = []

        # shared prefix: belongs to every sequence
        for i, t in enumerate(prefix_tokens):
            all_tokens.append(t)
            all_positions.append(prefix_pos + i)
            all_seq_ids.append(list(prefix_seq_ids))
            logits.append(False)

        # per-sequence continuations
        cont_start_pos = prefix_pos + len(prefix_tokens)
        for sid, toks in continuations:
            for i, t in enumerate(toks):
                all_tokens.append(t)
                all_positions.append(cont_start_pos + i)
                all_seq_ids.append([sid])
                logits.append(i == len(toks) - 1)

        if not all_tokens:
            return

        max_seq = max(1, len(prefix_seq_ids))
        batch = llama_batch_init(n_tokens=len(all_tokens), embd=0, n_seq_max=max_seq)
        try:
            _internals._fill_batch_multi(
                batch, all_tokens, all_positions, all_seq_ids, logits,
                max_seq_per_token=max_seq,
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(
                f'decode_shared_prefix_multi_seq: llama_decode failed with status code {ret}'
            )

    @staticmethod
    def decode_embd_multi_seq(
        ctx: ptr[llama_context],
        items: list[tuple[int, np.ndarray, int]],
        logits_all: bool = False,
    ) -> None:
        """### INTERNAL

        Decode embeddings for multiple sequences in a single batch.

        Each item is `(seq_id, embeddings, start_pos)`, where `embeddings` has shape
        `(n_rows, n_embd_inp)` and dtype float32. Every row is one token assigned to its own
        sequence. Logits are enabled on the last row of each sequence (or on every row when
        `logits_all` is True)."""
        if not items:
            return

        n_embd = llama_model_n_embd_inp(llama_get_model(ctx))

        all_embds: list[np.ndarray] = []
        all_positions: list[int] = []
        all_seq_ids: list[list[int]] = []
        logits: list[bool] = []

        for sid, emb, start_pos in items:
            if emb.ndim != 2 or emb.shape[1] != n_embd:
                raise ValueError(
                    f'decode_embd_multi_seq: expected embeddings of shape (n, {n_embd}), '
                    f'got {emb.shape}'
                )
            n_rows = emb.shape[0]
            all_embds.append(emb)
            for i in range(n_rows):
                all_positions.append(start_pos + i)
                all_seq_ids.append([sid])
                logits.append(logits_all or (i == n_rows - 1))

        total_rows = sum(e.shape[0] for e in all_embds)
        if total_rows == 0:
            return

        n_ctx = llama_n_ctx(ctx)
        if total_rows > n_ctx:
            raise ValueError(
                f'decode_embd_multi_seq: total rows {total_rows} exceeds context length {n_ctx}'
            )

        # concatenate and copy into the batch's `embd` buffer
        combined = np.concatenate(all_embds, axis=0).astype(np.float32, copy=False)
        if not combined.flags['C_CONTIGUOUS']:
            combined = np.ascontiguousarray(combined)

        batch = llama_batch_init(n_tokens=total_rows, embd=n_embd, n_seq_max=1)
        try:
            batch.n_tokens = total_rows
            batch.token    = NULL
            ctypes.memmove(batch.embd, combined.ctypes.data, combined.nbytes)
            _internals._fill_batch_multi(
                batch,
                tokens=[0] * total_rows,  # ignored when use_embd=True
                positions=all_positions,
                seq_ids=all_seq_ids,
                logits=logits,
                max_seq_per_token=1,
                use_embd=True,
            )
            ret = llama_decode(ctx, batch)
        finally:
            llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(
                f'decode_embd_multi_seq: llama_decode failed with status code {ret}'
            )

    @staticmethod
    def process_ext(
        ctx: ptr[llama_context],
        batch: ptr[llama_batch_ext],
        ptype: int = LlamaProcessType.LLAMA_PROCESS_TYPE_DECODE,
    ) -> int:
        """### INTERNAL

        Run `llama_process` on an already-built `llama_batch_ext`.

        Returns the same values as `llama_decode()`:
          - 0        : success
          - 1        : no KV slot for the batch
          - 2        : aborted
          - -1       : invalid input batch
          - < -1     : fatal error (processed ubatches remain in memory)"""
        return llama_process(ctx, ptype, batch)

    @staticmethod
    def decode_ext(
        ctx: ptr[llama_context],
        seqs: list[tuple[int, list[int], int]],
        ptype: int = LlamaProcessType.LLAMA_PROCESS_TYPE_DECODE,
        logits_last_per_seq: bool = True,
    ) -> int:
        """### INTERNAL

        Decode (or encode, via `ptype`) using the extended batch API.

        Each item is `(seq_id, tokens, start_pos)`; each token belongs to exactly one
        sequence. Logits are enabled on the last token of each non-empty sequence when
        `logits_last_per_seq=True`. Set `logits_last_per_seq=False` for pure encode passes.

        Returns the return value of `llama_process` (same semantics as `llama_decode`).
        Raises `RuntimeError` only if building the batch fails; check the return value for
        decode-time errors.

        NOTE: Unlike the other `decode_*` helpers, this function returns the raw
        `llama_process` status instead of raising, so callers can distinguish
        "no KV slot" (1) from "aborted" (2) from fatal errors (< -1)."""
        if not seqs:
            return 0

        batch = llama_batch_ext_init(ctx)
        try:
            # add all tokens
            flat_idx: list[int] = []
            for sid, toks, _start_pos in seqs:
                for t in toks:
                    idx = llama_batch_ext_add_token(batch, sid, t)
                    if idx < 0:
                        raise RuntimeError(
                            f'decode_ext: llama_batch_ext_add_token failed with {idx} '
                            f'(seq_id={sid}, token={t})'
                        )
                    flat_idx.append(idx)

            # set positions (keep arrays alive in case set_pos doesn't copy)
            keepalive: list = []
            pos_cursor = 0
            for _sid, toks, start_pos in seqs:
                for i in range(len(toks)):
                    pos_arr = (llama_pos * 1)(start_pos + i)
                    keepalive.append(pos_arr)
                    if not llama_batch_ext_set_pos(batch, flat_idx[pos_cursor], pos_arr):
                        raise RuntimeError('decode_ext: llama_batch_ext_set_pos failed')
                    pos_cursor += 1

            # enable logits on the last token of each non-empty sequence
            if logits_last_per_seq:
                cursor = 0
                for _sid, toks, _start_pos in seqs:
                    if toks:
                        if not llama_batch_ext_set_output_logits(
                            batch, flat_idx[cursor + len(toks) - 1], True
                        ):
                            raise RuntimeError(
                                'decode_ext: llama_batch_ext_set_output_logits failed'
                            )
                    cursor += len(toks)

            return llama_process(ctx, ptype, batch)
        finally:
            llama_batch_ext_free(batch)

    @staticmethod
    def encode_ext(
        ctx: ptr[llama_context],
        seqs: list[tuple[int, list[int], int]],
    ) -> int:
        """### INTERNAL

        Encode a batch using the extended API (for encoder-decoder models).

        Same `seqs` format as `decode_ext`. Encoder output is stored internally for the
        decoder's cross-attention layers, so no output flags are requested."""
        return _internals.decode_ext(
            ctx, seqs,
            ptype=LlamaProcessType.LLAMA_PROCESS_TYPE_ENCODE,
            logits_last_per_seq=False,
        )

    #
    # logits and embeddings
    #

    @staticmethod
    def get_logits(ctx: ptr[llama_context], n_vocab: int) -> np.ndarray:
        """### INTERNAL

        Get the logits for the last token decoded."""
        c_last_token_logits = llama_get_logits_ith(ctx, -1)
        return np.ctypeslib.as_array(c_last_token_logits, shape=(1, n_vocab))[0]

    @staticmethod
    def get_embeddings_ith(
        ctx: ptr[llama_context],
        i: int,
        n_embd: Optional[int] = None
    ) -> np.ndarray:
        """### INTERNAL

        Get the embeddings for the ith token (-1 == the last one).

        `n_embd` should be the model's *output* embedding size
        (`llama_model_n_embd_out`). NOTE: for classifier/reranker models used with
        `LLAMA_POOLING_TYPE_RANK`, the raw `llama_get_embeddings_seq` returns `n_cls_out`
        floats per sequence, not `n_embd_out`."""
        if n_embd is None:
            n_embd = llama_model_n_embd_out(llama_get_model(ctx))
        c_embd = llama_get_embeddings_ith(ctx, i)
        return np.ctypeslib.as_array(c_embd, shape=(1, n_embd))[0]

    #
    # sampling
    #

    @staticmethod
    def sample_greedy(ctx: ptr[llama_context]) -> int:
        """### INTERNAL

        Sample the most likely token."""
        return llama_sampler_sample(_internals.greedy_sampler, ctx, -1)

    @staticmethod
    def get_sampled_probs(ctx: ptr[llama_context], i: int) -> np.ndarray:
        """### INTERNAL

        Get the backend-sampled probabilities for the `i`th token as a float32 array.

        Returns an empty array if no probabilities were generated."""
        n = llama_get_sampled_probs_count_ith(ctx, i)
        if n <= 0:
            return np.empty(0, dtype=np.float32)
        c_probs = llama_get_sampled_probs_ith(ctx, i)
        if not c_probs:
            return np.empty(0, dtype=np.float32)
        return np.ctypeslib.as_array(c_probs, shape=(n,))

    @staticmethod
    def get_sampled_logits(ctx: ptr[llama_context], i: int) -> np.ndarray:
        """### INTERNAL

        Get the backend-sampled logits for the `i`th token as a float32 array.

        Returns an empty array if no logits were sampled."""
        n = llama_get_sampled_logits_count_ith(ctx, i)
        if n <= 0:
            return np.empty(0, dtype=np.float32)
        c_logits = llama_get_sampled_logits_ith(ctx, i)
        if not c_logits:
            return np.empty(0, dtype=np.float32)
        return np.ctypeslib.as_array(c_logits, shape=(n,))

    @staticmethod
    def get_sampled_candidates(ctx: ptr[llama_context], i: int) -> np.ndarray:
        """### INTERNAL

        Get the backend-sampled candidate token ids for the ith token as an int32 array.

        These map the indices of `get_sampled_probs` / `get_sampled_logits` to vocabulary token
        IDs. Returns an empty array if no candidates were sampled."""
        n = llama_get_sampled_candidates_count_ith(ctx, i)
        if n <= 0:
            return np.empty(0, dtype=np.int32)
        c_candidates = llama_get_sampled_candidates_ith(ctx, i)
        if not c_candidates:
            return np.empty(0, dtype=np.int32)
        return np.ctypeslib.as_array(c_candidates, shape=(n,))

    #
    # tokenization
    #

    @staticmethod
    def tokenize(
        vocab: ptr[llama_vocab],
        text_bytes: bytes,
        n_tokens_max: int,
        add_special: bool,
        parse_special: bool,
    ) -> list[int]:
        """### INTERNAL

        Convert the provided UTF-8 encoded text into tokens

        - vocab:
            A pointer to a llama_vocab to use for tokenization
        - text_bytes:
            The text to be tokenized
        - n_tokens_max:
            Tokenization will fail if the text is more than this many tokens.
            Larger numbers allow more text to be tokenized but will allocate
            more memory (4 bytes per token).
        - add_special:
            Allow to add BOS and EOS tokens if model is configured to do so.
        - parse_special:
            Allow tokenizing special and/or control tokens which otherwise are
            not exposed and treated as plaintext. Does not insert a leading
            space."""
        # unlike detokenization, this buffer is created and destroyed as needed
        # because it could potentially be quite large - each token takes 4 bytes
        tokens_buf = (llama_token * n_tokens_max)()
        n_tokens = llama_tokenize(
            vocab=vocab,
            text=text_bytes,
            text_len=len(text_bytes),
            tokens=tokens_buf,
            n_tokens_max=n_tokens_max,
            add_special=add_special,
            parse_special=parse_special
        )
        if n_tokens == _INT32_MIN:
            raise ValueError(
                'tokenize: tokenization result size exceeds int32_t limit'
            )
        if n_tokens < 0:
            raise ValueError(
                f'tokenize: n_tokens value {-n_tokens} exceeds '
                f'n_tokens_max value {n_tokens_max}'
            )
        ret = list(tokens_buf[:n_tokens])
        del tokens_buf
        return ret

    @staticmethod
    def token_to_piece(vocab: ptr[llama_vocab], token: int, special: bool) -> bytes:
        """### INTERNAL

        Convert token ID to text bytes

        NOTE: uses a per-call buffer so that concurrent callers do not stomp on each other's
        results (the tokenization API is documented as thread-safe)."""
        buf = ctypes.create_string_buffer(MAX_TOKEN_LENGTH)
        n_bytes = llama_token_to_piece(
            vocab=vocab,
            token=token,
            buf=buf,
            length=MAX_TOKEN_LENGTH,
            lstrip=0, # skip up to 'lstrip' leading spaces
            special=special
        )
        if n_bytes < 0:
            # per `llama_token_to_piece`, a negative return means the buffer was too small;
            # `-n_bytes` is the size that would have been required
            raise ValueError(
                f"token_to_piece: the token with ID {token} requires a buffer of size "
                f"{-n_bytes}, but the maximum buffer size is {MAX_TOKEN_LENGTH}"
            )
        # NOTE: do not just do buf.value.decode() because the token could
        #       possibly be a part of a utf-8 bytestring, but not a valid utf-8
        #       string itself. let the caller handle this
        return buf.raw[:n_bytes]

    @staticmethod
    def detokenize(
        vocab: ptr[llama_vocab],
        tokens: list[int],
        special: bool
    ) -> str:
        """### INTERNAL

        Convert the provided tokens into a string

        - special:
            If True, special tokens are rendered in the output

        NOTE: the buffer is allocated per call so concurrent callers are safe."""
        buf = ctypes.create_string_buffer(MAX_TOKEN_LENGTH)
        detok_bytes = b""
        for token in tokens:
            n_bytes = llama_token_to_piece(
                vocab=vocab,
                token=token,
                buf=buf,
                length=MAX_TOKEN_LENGTH,
                lstrip=0, # skip up to 'lstrip' leading spaces
                special=special
            )
            if n_bytes < 0:
                raise ValueError(
                    f"detokenize: the token with ID {token} requires a buffer of size "
                    f"{-n_bytes}, but the maximum buffer size is {MAX_TOKEN_LENGTH}"
                )
            detok_bytes += buf.raw[:n_bytes]
        return str_decode(detok_bytes)

    @staticmethod
    def get_length(
        vocab: ptr[llama_vocab],
        text_bytes: bytes,
        add_special: bool,
        parse_special: bool,
    ) -> int:
        """### INTERNAL

        Return the length of a given text, as measured in tokens"""
        n = llama_tokenize(
            vocab=vocab,
            text=text_bytes,
            text_len=len(text_bytes),
            tokens=NULL,
            n_tokens_max=0,
            add_special=add_special,
            parse_special=parse_special
        )
        if n == _INT32_MIN:
            raise ValueError(
                'get_length: tokenization result size exceeds int32_t limit'
            )
        return -n

    #
    # ctypes array builders
    #

    @staticmethod
    def get_logit_bias_array(logit_biases: dict[int, float]) -> LogitBiasArrayType:
        """### INTERNAL

        Create and return a ctypes array of `llama_logit_bias`"""
        if len(logit_biases) == 0:
            raise ValueError(f'logit_biases parameter cannot be empty')
        LogitBiasArray = llama_logit_bias * len(logit_biases)
        arr = LogitBiasArray()
        i = 0
        for k, v in logit_biases.items():
            arr[i].token = k
            arr[i].bias = v
            i += 1
        return arr

    @staticmethod
    def get_char_p_array(strings: list[str]) -> CharPArrayType:
        """### INTERNAL

        Create and return a ctypes array of `c_char_p` holding UTF-8 encoded copies of the
        given strings.

        Used by `llama_sampler_init_dry` (seq_breakers) and
        `llama_sampler_init_grammar_lazy_patterns` (trigger_patterns)."""
        if len(strings) == 0:
            raise ValueError(f'strings parameter cannot be empty')
        CharPArray = ctypes.c_char_p * len(strings)
        return CharPArray(*[s.encode('utf-8') for s in strings])

    @staticmethod
    def get_float_array(values: list[float]) -> FloatArrayType:
        """### INTERNAL

        Create and return a ctypes array of `c_float`.

        Used by `llama_set_adapters_lora` (scales) and `llama_model_params.tensor_split`."""
        if len(values) == 0:
            raise ValueError(f'values parameter cannot be empty')
        FloatArray = ctypes.c_float * len(values)
        return FloatArray(*values)

    @staticmethod
    def get_adapter_array(adapters: list) -> AdapterArrayType:
        """### INTERNAL

        Create and return a ctypes array of `llama_adapter_lora_p`.

        Used by `llama_set_adapters_lora` (adapters)."""
        if len(adapters) == 0:
            raise ValueError(f'adapters parameter cannot be empty')
        AdapterArray = llama_adapter_lora_p * len(adapters)
        return AdapterArray(*adapters)

    @staticmethod
    def check_dry_seq_breakers(seq_breakers: list[str]) -> None:
        """### INTERNAL

        Raise if more DRY sequence breakers were provided than libllama supports
        (`LLAMA_DRY_MAX_SEQ_BREAKERS`). libllama silently ignores the excess, so failing here
        is friendlier than letting the user believe all of their breakers are active."""
        n = len(seq_breakers)
        if n > LLAMA_DRY_MAX_SEQ_BREAKERS:
            raise ValueError(
                f'dry_sequence_breakers: got {n} breakers, but libllama supports at most '
                f'{LLAMA_DRY_MAX_SEQ_BREAKERS}'
            )

    @staticmethod
    def make_embd(embeddings: np.ndarray) -> llama_embd:
        """### INTERNAL

        Build a `llama_embd` from a numpy array of shape (n_rows, n_embd).

        The array must be float32 and contiguous -- this function never copies, it points
        directly into `embeddings`' buffer. The caller must keep `embeddings` alive for as long
        as the returned `llama_embd` (or anything copied from it) is in use; libllama holds a raw
        pointer into it and does not copy."""
        if embeddings.ndim != 2:
            raise ValueError(
                f'make_embd: expected a 2D array of shape (n_rows, n_embd), got '
                f'{embeddings.shape}'
            )
        if embeddings.dtype != np.float32:
            raise ValueError(
                f'make_embd: expected a float32 array, got {embeddings.dtype}. convert it in '
                f'your own code so you control the lifetime of the converted array.'
            )
        if not embeddings.is_contiguous():
            raise ValueError(
                'make_embd: expected a contiguous array. call np.ascontiguousarray() in your '
                'own code so you control the lifetime of the converted array.'
            )
        embd = llama_embd()
        embd.data   = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        embd.n_rows = embeddings.shape[0]
        embd.n_embd = embeddings.shape[1]
        return embd
