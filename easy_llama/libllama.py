# libllama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay
# -------------------------------------------------------------------------------------------- #
# Manually update these constants to reflect the currently targeted version of llama.cpp:
# -------------------------------------------------------------------------------------------- #
# llama.cpp commit OID
_TARGET_LLAMACPP_COMMIT = "803f8baf4f741d2f0465c46c33f285886b97a071"
# YYYY-MM-DD, the date of the above commit
_TARGET_LLAMACPP_DATE   = "2025-05-31"
# -------------------------------------------------------------------------------------------- #

f"""This file provides a Python interface to LLAMA_API ("libllama"), which is originally defined
in `llama.cpp/include/llama.h`.

This file was last updated to match `llama.cpp/include/llama.h` as of this commit:
- Full SHA: `{_TARGET_LLAMACPP_COMMIT}`
- Date: {_TARGET_LLAMACPP_DATE}

This file's status with respect to the above commit is:
- Synchronized

Helpful references:
- `libllama` API changelog:
    [llama.cpp/issues/9289](https://github.com/ggml-org/llama.cpp/issues/9289)
- `llama.h` at master:
    [llama.cpp/blob/master/include/llama.h](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)"""

from . import __version__

import os
import sys
import ctypes
import faulthandler

import numpy as np

from enum   import IntEnum
from typing import Optional, Callable
from .utils import ptr, log, ez_decode, log_verbose

faulthandler.enable() # prints more helpful info if python crashes

#
# Type hints and other constants
#

NULL = None
NULLPTR = ctypes.c_void_p(NULL)

C_FALSE = ctypes.c_int8(0)
C_TRUE = ctypes.c_int8(1)

MAX_TOKEN_LENGTH = 256 # update this if you find a token that is > 256 bytes long
"""The maximum supported length of a single token's text, in bytes"""

# maximum value for int32, it is used as the value for n_gpu_layers
# when all layers should be offloaded
MAX_OFFLOAD_LAYERS = 0x7FFFFFFF

# keep state for backend
_BACKEND_INIT = False

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
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35
    # GGML_TYPE_IQ4_NL_4_4 = 36
    # GGML_TYPE_IQ4_NL_4_8 = 37
    # GGML_TYPE_IQ4_NL_8_8 = 38
    GGML_TYPE_COUNT   = 39

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
# Constants
#

LLAMA_DEFAULT_SEED = 0xFFFFFFFF

LLAMA_TOKEN_NULL = -1

LLAMA_FILE_MAGIC_GGLA = 0x67676c61 # 'ggla'
LLAMA_FILE_MAGIC_GGSN = 0x6767736e # 'ggsn'
LLAMA_FILE_MAGIC_GGSQ = 0x67677371 # 'ggsq'

LLAMA_SESSION_MAGIC   = LLAMA_FILE_MAGIC_GGSN
LLAMA_SESSION_VERSION = 9

LLAMA_STATE_SEQ_MAGIC   = LLAMA_FILE_MAGIC_GGSQ
LLAMA_STATE_SEQ_VERSION = 2

#
# Begin LLAMA_API
#

class llama_vocab(ctypes.Structure):
    pass

llama_vocab_p = ctypes.POINTER(llama_vocab)

class llama_model(ctypes.Structure):
    pass

llama_model_p = ctypes.POINTER(llama_model)

class llama_context(ctypes.Structure):
    pass

llama_context_p = ctypes.POINTER(llama_context)

class llama_kv_cache(ctypes.Structure):
    pass

llama_kv_cache_p = ctypes.POINTER(llama_kv_cache)

llama_pos    = ctypes.c_int32
llama_token  = ctypes.c_int32
llama_seq_id = ctypes.c_int32

#
# Enums
#

class LlamaVocabType(IntEnum):
    LLAMA_VOCAB_TYPE_NONE = 0 # For models without vocab
    LLAMA_VOCAB_TYPE_SPM  = 1 # LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE  = 2 # GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM  = 3 # BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM  = 4 # T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV = 5 # RWKV tokenizer based on greedy tokenization

class LlamaVocabPreType(IntEnum):
    LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0
    LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3
    LLAMA_VOCAB_PRE_TYPE_FALCON         = 4
    LLAMA_VOCAB_PRE_TYPE_MPT            = 5
    LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6
    LLAMA_VOCAB_PRE_TYPE_GPT2           = 7
    LLAMA_VOCAB_PRE_TYPE_REFACT         = 8
    LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9
    LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10
    LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11
    LLAMA_VOCAB_PRE_TYPE_OLMO           = 12
    LLAMA_VOCAB_PRE_TYPE_DBRX           = 13
    LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14
    LLAMA_VOCAB_PRE_TYPE_PORO           = 15
    LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16
    LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17
    LLAMA_VOCAB_PRE_TYPE_VIKING         = 18
    LLAMA_VOCAB_PRE_TYPE_JAIS           = 19
    LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20
    LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21
    LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22
    LLAMA_VOCAB_PRE_TYPE_BLOOM          = 23
    LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24
    LLAMA_VOCAB_PRE_TYPE_EXAONE         = 25
    LLAMA_VOCAB_PRE_TYPE_CHAMELEON      = 26
    LLAMA_VOCAB_PRE_TYPE_MINERVA        = 27
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM  = 28
    LLAMA_VOCAB_PRE_TYPE_GPT4O          = 29
    LLAMA_VOCAB_PRE_TYPE_SUPERBPE       = 30
    LLAMA_VOCAB_PRE_TYPE_TRILLION       = 31
    LLAMA_VOCAB_PRE_TYPE_BAILINGMOE     = 32
    LLAMA_VOCAB_PRE_TYPE_LLAMA4         = 33
    LLAMA_VOCAB_PRE_TYPE_PIXTRAL        = 34
    LLAMA_VOCAB_PRE_TYPE_SEED_CODER     = 35

class LlamaRopeType(IntEnum):
    LLAMA_ROPE_TYPE_NONE   = -1
    LLAMA_ROPE_TYPE_NORM   = 0
    LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX
    LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE
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
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3 # SPECIAL?
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

    LLAMA_FTYPE_GUESSED = 1024 # not specified in the model file

class LlamaRopeScalingType(IntEnum):
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    LLAMA_ROPE_SCALING_TYPE_NONE        = 0
    LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1
    LLAMA_ROPE_SCALING_TYPE_YARN        = 2
    LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE

class LlamaPoolingType(IntEnum):
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    LLAMA_POOLING_TYPE_NONE = 0
    LLAMA_POOLING_TYPE_MEAN = 1
    LLAMA_POOLING_TYPE_CLS  = 2
    LLAMA_POOLING_TYPE_LAST = 3
    LLAMA_POOLING_TYPE_RANK = 4 # used by reranking models to attach the classification head to the graph

class LlamaAttentionType(IntEnum):
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    LLAMA_ATTENTION_TYPE_CAUSAL      = 0
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1

class LlamaSplitMode(IntEnum):
    LLAMA_SPLIT_MODE_NONE  = 0 # single GPU
    LLAMA_SPLIT_MODE_LAYER = 1 # split layers and KV across GPUs
    LLAMA_SPLIT_MODE_ROW   = 2 # split layers and KV across GPUs, use tensor parallelism if supported

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
        ("sorted",   ctypes.c_bool                   )
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
        ("pos",      ctypes.POINTER(llama_pos)                   ),
        # the sequence to which the respective token belongs
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)              ),
        # the sequence to which the respective token belongs
        ("seq_id",   ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        # if zero, the logits (and/or the embeddings) for the respective token will not be output
        ("logits",   ctypes.POINTER(ctypes.c_int8)               ) # use C_FALSE / C_TRUE
    ]

llama_batch_p = ctypes.POINTER(llama_batch)

class llama_model_kv_override_type(ctypes.Union):
    _fields_ = [
        ("val_i64",  ctypes.c_int64     ),
        ("val_f64",  ctypes.c_double    ),
        ("val_bool", ctypes.c_bool      ),
        ("val_str",  ctypes.c_char * 128)
    ]

llama_model_kv_override_type_p = ctypes.POINTER(llama_model_kv_override_type)

class llama_model_kv_override(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_int                ),
        ("key", ctypes.c_char * 128         ),
        ("val", llama_model_kv_override_type)
    ]
    _anonymous_ = ("val",) # make the `llama_model_kv_override_type` accessible anonymously

llama_model_kv_override_p = ctypes.POINTER(llama_model_kv_override)

ggml_backend_buffer_type_t = ctypes.c_void_p # opaque

class llama_model_tensor_buft_override(ctypes.Structure):
    _fields_ = [
        ("pattern", ctypes.c_char_p           ), # regex pattern to match tensor names
        ("buft",    ggml_backend_buffer_type_t)  # opaque pointer to buffer type
    ]

llama_model_tensor_buft_override_p = ctypes.POINTER(llama_model_tensor_buft_override)

progress_callback_functype = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)

class llama_model_params(ctypes.Structure):
    # NOTE: These fields need to be aligned with struct llama_model_params {...} !!!
    #       If they are not aligned, you might get crashes that are very hard to debug !!!
    _fields_ = [
        # NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ("devices", ctypes.POINTER(ctypes.c_void_p)),

        # NULL-terminated list of buffer types to use for tensors that match a pattern
        ("tensor_buft_overrides", ctypes.POINTER(llama_model_tensor_buft_override)),

        ("n_gpu_layers", ctypes.c_int32), # number of layers to store in VRAM
        ("split_mode",   ctypes.c_int  ), # how to split the model across multiple GPUs

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
        ("vocab_only",    ctypes.c_bool), # only load the vocabulary, no weights
        ("use_mmap",      ctypes.c_bool), # use mmap if possible
        ("use_mlock",     ctypes.c_bool), # force system to keep model in RAM
        ("check_tensors", ctypes.c_bool)  # validate model tensor data
    ]

llama_model_params_p = ctypes.POINTER(llama_model_params)

eval_callback_functype = ctypes.CFUNCTYPE(None, ctypes.c_bool, ctypes.c_void_p)

abort_callback_functype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

class llama_context_params(ctypes.Structure):
    # NOTE: These fields need to be aligned with struct llama_context_params {...} !!!
    #       If they are not aligned, you might get crashes that are very hard to debug !!!
    _fields_ = [
        ("n_ctx",           ctypes.c_uint32), # text context, 0 = from model
        ("n_batch",         ctypes.c_uint32), # logical maximum batch size that can be submitted to llama_decode
        ("n_ubatch",        ctypes.c_uint32), # physical maximum batch size (micro-batch size)
        ("n_seq_max",       ctypes.c_uint32), # max number of sequences (i.e. distinct states for recurrent models)
        ("n_threads",       ctypes.c_int32 ), # number of threads to use for generation
        ("n_threads_batch", ctypes.c_int32 ), # number of threads to use for batch processing

        ("rope_scaling_type", ctypes.c_int), # RoPE scaling type, from `enum llama_rope_scaling_type`
        ("pooling_type",      ctypes.c_int), # whether to pool (sum) embedding results by sequence id
        ("attention_type",    ctypes.c_int), # attention type to use for embeddings

        # ref: https://github.com/ggml-org/llama.cpp/pull/2054
        ("rope_freq_base",   ctypes.c_float ), # RoPE base frequency, 0 = from model
        ("rope_freq_scale",  ctypes.c_float ), # RoPE frequency scaling factor, 0 = from model
        ("yarn_ext_factor",  ctypes.c_float ), # YaRN extrapolation mix factor, negative = from model
        ("yarn_attn_factor", ctypes.c_float ), # YaRN magnitude scaling factor
        ("yarn_beta_fast",   ctypes.c_float ), # YaRN low correction dim
        ("yarn_beta_slow",   ctypes.c_float ), # YaRN high correction dim
        ("yarn_orig_ctx",    ctypes.c_uint32), # YaRN original context size
        ("defrag_thold",     ctypes.c_float ), # defragment the KV cache if holes/size > thold, < 0 disabled (default)

        ("cb_eval",           eval_callback_functype), # callback for eval
        ("cb_eval_user_data", ctypes.c_void_p       ), # user data for eval callback

        ("type_k", ctypes.c_int), # data type for K cache [EXPERIMENTAL]
        ("type_v", ctypes.c_int), # data type for V cache [EXPERIMENTAL]

        # Abort callback
        # if it returns true, execution of llama_decode() will be aborted
        # currently works only with CPU execution
        ("abort_callback",      abort_callback_functype), # callback for abort
        ("abort_callback_data", ctypes.c_void_p        ), # user data for abort callback

        # Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        ("embeddings",  ctypes.c_bool), # if true, extract embeddings (together with logits)
        ("offload_kqv", ctypes.c_bool), # whether to offload the KQV ops (including the KV cache) to GPU
        ("flash_attn",  ctypes.c_bool), # whether to use flash attention [EXPERIMENTAL]
        ("no_perf",     ctypes.c_bool), # whether to measure performance timings
        ("op_offload",  ctypes.c_bool), # whether to offload host tensor operations to device
        ("swa_full",    ctypes.c_bool)  # use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
    ]

llama_context_params_p = ctypes.POINTER(llama_context_params)

class llama_model_quantize_params(ctypes.Structure):
    _fields_ = [
        ("nthread",                ctypes.c_int32 ), # number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        ("ftype",                  ctypes.c_int   ), # quantize to this llama_ftype
        ("output_tensor_type",     ctypes.c_int   ), # output tensor type
        ("token_embedding_type",   ctypes.c_int   ), # token embeddings tensor type
        ("allow_requantize",       ctypes.c_bool  ), # allow quantizing non-f32/f16 tensors
        ("quantize_output_tensor", ctypes.c_bool  ), # quantize output.weight
        ("only_copy",              ctypes.c_bool  ), # only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        ("pure",                   ctypes.c_bool  ), # quantize all tensors to the default type
        ("keep_split",             ctypes.c_bool  ), # quantize to the same number of shards
        ("imatrix",                ctypes.c_void_p), # pointer to importance matrix data
        ("kv_overrides",           ctypes.c_void_p), # pointer to vector containing overrides
        ("tensor_types",           ctypes.c_void_p)  # pointer to vector containing tensor types
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
    
    log_verbose(
        f'easy_llama v{__version__} '
        f'targeting llama.cpp@{_TARGET_LLAMACPP_COMMIT[:7]} ({_TARGET_LLAMACPP_DATE})'
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
            f"set LIBLLAMA to the path of your libllama shared library file (`.so`, `.dll`, or "
            f"`.dylib`). For example, on Linux: `export LIBLLAMA=/path/to/your/libllama.so`."
        )

    if os.path.isdir(lib_path):
        log(f'failed to load libllama: $LIBLLAMA is a directory, not a file: {lib_path}', 3)
        raise IsADirectoryError(
            f"The llama.cpp shared library could not be loaded because the LIBLLAMA "
            f"environment variable points to a directory, but it should point to a file "
            f"instead. You must set LIBLLAMA to the path of your libllama shared library file "
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

    # initialize the built-in detokenization buffer
    _internals.detok_buffer = ctypes.create_string_buffer(MAX_TOKEN_LENGTH)

    # don't let this function be called again
    _BACKEND_INIT = True

#
# Helpers for getting default parameters
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
    libllama.llama_backend_free.argtypes = []
    libllama.llama_backend_free.restype = None
    libllama.llama_backend_free()
    _BACKEND_INIT = False

def llama_numa_init(numa: int) -> None:
    """Initialize NUMA optimizations globally"""
    libllama.llama_numa_init.argtypes = [ctypes.c_int]
    libllama.llama_numa_init.restype = None
    libllama.llama_numa_init(numa)

def llama_attach_threadpool(ctx: ptr[llama_context], ggml_threadpool: ptr, threadpool_batch: ptr) -> None:
    """Attach a threadpool to a llama_context"""
    libllama.llama_attach_threadpool.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_attach_threadpool.restype = None
    libllama.llama_attach_threadpool(ctx, ggml_threadpool, threadpool_batch)

def llama_detach_threadpool(ctx: ptr[llama_context]) -> None:
    """Detach a threadpool from a llama_context"""
    libllama.llama_detach_threadpool.argtypes = [llama_context_p]
    libllama.llama_detach_threadpool.restype = None
    libllama.llama_detach_threadpool(ctx)

def llama_model_load_from_file(path_model: str, params: llama_model_params) -> ptr[llama_model]:
    """Load a llama model from a file. Returns a pointer."""
    libllama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    libllama.llama_model_load_from_file.restype = llama_model_p
    return libllama.llama_model_load_from_file(path_model.encode('utf-8'), params)

def llama_model_load_from_splits(paths: list[str], n_paths: int, params: llama_model_params) -> ptr[llama_model]:
    """Load a llama model from multiple splits (support custom naming scheme). Returns a pointer."""
    libllama.llama_model_load_from_splits.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t, llama_model_params]
    libllama.llama_model_load_from_splits.restype = llama_model_p
    c_paths = (ctypes.c_char_p * n_paths)(*[p.encode("utf-8") for p in paths])
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

def llama_init_from_model(model: ptr[llama_model], params: llama_context_params) -> ptr[llama_context]:
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
# Llama backend helper functions
#

def llama_time_us() -> int:
    """Get the current time in microseconds"""
    libllama.llama_time_us.argtypes = []
    libllama.llama_time_us.restype = ctypes.c_int
    return libllama.llama_time_us()

def llama_max_devices() -> int:
    """Get the maximum number of devices"""
    libllama.llama_max_devices.argtypes = []
    libllama.llama_max_devices.restype = ctypes.c_int
    return libllama.llama_max_devices()

def llama_max_parallel_sequences() -> int:
    """Get the maximum number of parallel sequences supported by the backend."""
    libllama.llama_max_parallel_sequences.argtypes = []
    libllama.llama_max_parallel_sequences.restype = ctypes.c_size_t
    return libllama.llama_max_parallel_sequences()

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
# Getters for llama_context
#

def llama_n_ctx(ctx: ptr[llama_context]) -> int:
    """Get the context size"""
    libllama.llama_n_ctx.argtypes = [llama_context_p]
    libllama.llama_n_ctx.restype = ctypes.c_int
    return libllama.llama_n_ctx(ctx)

def llama_n_batch(ctx: ptr[llama_context]) -> int:
    """Get the logical maximum batch size"""
    libllama.llama_n_batch.argtypes = [llama_context_p]
    libllama.llama_n_batch.restype = ctypes.c_int
    return libllama.llama_n_batch(ctx)

def llama_n_ubatch(ctx: ptr[llama_context]) -> int:
    """Get the physical maximum batch size"""
    libllama.llama_n_ubatch.argtypes = [llama_context_p]
    libllama.llama_n_ubatch.restype = ctypes.c_int
    return libllama.llama_n_ubatch(ctx)

def llama_n_seq_max(ctx: ptr[llama_context]) -> int:
    """Get the maximum number of sequences"""
    libllama.llama_n_seq_max.argtypes = [llama_context_p]
    libllama.llama_n_seq_max.restype = ctypes.c_int
    return libllama.llama_n_seq_max(ctx)

def llama_get_model(ctx: ptr[llama_context]) -> ptr[llama_model]:
    """Get the model associated with a context"""
    libllama.llama_get_model.argtypes = [llama_context_p]
    libllama.llama_get_model.restype = llama_model_p
    return libllama.llama_get_model(ctx)

def llama_get_kv_self(ctx: ptr[llama_context]) -> ptr[llama_kv_cache]:
    libllama.llama_get_kv_self.argtypes = [llama_context_p]
    libllama.llama_get_kv_self.restype = llama_kv_cache_p
    return libllama.llama_get_kv_self(ctx)

def llama_pooling_type(ctx: ptr[llama_context]) -> int:
    """Get the pooling type used by a context"""
    libllama.llama_pooling_type.argtypes = [llama_context_p]
    libllama.llama_pooling_type.restype = ctypes.c_int
    return libllama.llama_pooling_type(ctx)

#
# Getters for llama_model
#

def llama_model_get_vocab(model: ptr[llama_model]) -> ptr[llama_vocab]:
    """Get a pointer to the llama_vocab struct"""
    libllama.llama_model_get_vocab.argtypes = [llama_model_p]
    libllama.llama_model_get_vocab.restype = llama_vocab_p
    return libllama.llama_model_get_vocab(model)

def llama_model_rope_type(model: ptr[llama_model]) -> int:
    """Get the RoPE type used by a model"""
    libllama.llama_model_rope_type.argtypes = [llama_model_p]
    libllama.llama_model_rope_type.restype = ctypes.c_int
    return libllama.llama_model_rope_type(model)

def llama_model_n_ctx_train(model: ptr[llama_model]) -> int:
    """Get the context size used during training"""
    libllama.llama_model_n_ctx_train.argtypes = [llama_model_p]
    libllama.llama_model_n_ctx_train.restype = ctypes.c_int
    return libllama.llama_model_n_ctx_train(model)

def llama_model_n_embd(model: ptr[llama_model]) -> int:
    """Get the embedding size"""
    libllama.llama_model_n_embd.argtypes = [llama_model_p]
    libllama.llama_model_n_embd.restype = ctypes.c_int
    return libllama.llama_model_n_embd(model)

def llama_model_n_layer(model: ptr[llama_model]) -> int:
    """Get the number of layers"""
    libllama.llama_model_n_layer.argtypes = [llama_model_p]
    libllama.llama_model_n_layer.restype = ctypes.c_int
    return libllama.llama_model_n_layer(model)

def llama_model_n_head(model: ptr[llama_model]) -> int:
    """Get the number of attention heads"""
    libllama.llama_model_n_head.argtypes = [llama_model_p]
    libllama.llama_model_n_head.restype = ctypes.c_int
    return libllama.llama_model_n_head(model)

def llama_model_n_head_kv(model: ptr[llama_model]) -> int:
    """Get the number of KV heads"""
    libllama.llama_model_n_head_kv.argtypes = [llama_model_p]
    libllama.llama_model_n_head_kv.restype = ctypes.c_int
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

#
# Getters for llama_vocab
#

def llama_vocab_type(vocab: ptr[llama_vocab]) -> int:
    """Get the LlamaVocabType of this llama_vocab"""
    libllama.llama_vocab_type.argtypes = [llama_vocab_p]
    libllama.llama_vocab_type.restype = ctypes.c_int
    return libllama.llama_vocab_type(vocab)

def llama_vocab_n_tokens(vocab: ptr[llama_vocab]) -> int:
    """Get the number of tokens in this llama_vocab"""
    libllama.llama_vocab_n_tokens.argtypes = [llama_vocab_p]
    libllama.llama_vocab_n_tokens.restype = ctypes.c_int
    return libllama.llama_vocab_n_tokens(vocab)

#
# GGUF metadata functions
#

def llama_model_meta_val_str(model: ptr[llama_model], key: str, buf: ctypes.c_char_p, buf_size: int) -> int:
    """Get a metadata value as a string"""
    libllama.llama_model_meta_val_str.argtypes = [llama_model_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    libllama.llama_model_meta_val_str.restype = ctypes.c_int
    return libllama.llama_model_meta_val_str(model, key.encode('utf-8'), buf, buf_size)

def llama_model_meta_count(model: ptr[llama_model]) -> int:
    """Get the number of metadata key-value pairs"""
    libllama.llama_model_meta_count.argtypes = [llama_model_p]
    libllama.llama_model_meta_count.restype = ctypes.c_int
    return libllama.llama_model_meta_count(model)

def llama_model_meta_key_by_index(model: ptr[llama_model], i: int, buf: ctypes.c_char_p, buf_size: int) -> int:
    """Get a metadata key by index"""
    libllama.llama_model_meta_key_by_index.argtypes = [llama_model_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    libllama.llama_model_meta_key_by_index.restype = ctypes.c_int
    return libllama.llama_model_meta_key_by_index(model, i, buf, buf_size)

def llama_model_meta_val_str_by_index(model: ptr[llama_model], i: int, buf: ctypes.c_char_p, buf_size: int) -> int:
    """Get a metadata value by index"""
    libllama.llama_model_meta_val_str_by_index.argtypes = [llama_model_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    libllama.llama_model_meta_val_str_by_index.restype = ctypes.c_int
    return libllama.llama_model_meta_val_str_by_index(model, i, buf, buf_size)

def llama_model_desc(model: ptr[llama_model], buf: ctypes.c_char_p, buf_size: int) -> int:
    """Get a string describing the model type"""
    libllama.llama_model_desc.argtypes = [llama_model_p, ctypes.c_char_p, ctypes.c_int]
    libllama.llama_model_desc.restype = ctypes.c_int
    return libllama.llama_model_desc(model, buf, buf_size)

def llama_model_size(model: ptr[llama_model]) -> int:
    """Get the total size of all tensors in the model in bytes"""
    libllama.llama_model_size.argtypes = [llama_model_p]
    libllama.llama_model_size.restype = ctypes.c_size_t
    return libllama.llama_model_size(model)

def llama_model_chat_template(model: ptr[llama_model], name: Optional[str] = None) -> Optional[str]:
    """Get the default chat template. Returns None if not available.
    If `name` is None, return the default chat template."""
    libllama.llama_model_chat_template.argtypes = [llama_model_p, ctypes.c_char_p]
    libllama.llama_model_chat_template.restype = ctypes.c_char_p
    c_name = name.encode("utf-8") if name is not None else None
    c_template = libllama.llama_model_chat_template(model, c_name)
    if c_template:
        c_template_p = ctypes.c_char_p(c_template)
        return c_template_p.value.decode('utf-8')
    else:
        return None

def llama_model_n_params(model: ptr[llama_model]) -> int:
    """Get the total number of parameters in the model"""
    libllama.llama_model_n_params.argtypes = [llama_model_p]
    libllama.llama_model_n_params.restype = ctypes.c_size_t
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
    """Get the start token for the decoder"""
    libllama.llama_model_decoder_start_token.argtypes = [llama_model_p]
    libllama.llama_model_decoder_start_token.restype = ctypes.c_int
    return libllama.llama_model_decoder_start_token(model)

def llama_model_is_recurrent(model: ptr[llama_model]) -> bool:
    """Check if the model is recurrent"""
    libllama.llama_model_is_recurrent.argtypes = [llama_model_p]
    libllama.llama_model_is_recurrent.restype = ctypes.c_bool
    return libllama.llama_model_is_recurrent(model)

#
# Quantization
#

def llama_model_quantize(fname_inp: str, fname_out: str, params: llama_model_quantize_params) -> int:
    """Quantize a model. Returns 0 on success"""
    libllama.llama_model_quantize.argtypes = [ctypes.c_char_p, ctypes.c_char_p, llama_model_quantize_params_p]
    libllama.llama_model_quantize.restype = ctypes.c_int
    return libllama.llama_model_quantize(fname_inp.encode('utf-8'), fname_out.encode('utf-8'), ctypes.byref(params))

#
# Adapters
#

def llama_adapter_lora_init(model: ptr[llama_model], path_lora: str) -> ptr[llama_adapter_lora]:
    """Initialize a LoRA adapter"""
    libllama.llama_adapter_lora_init.argtypes = [llama_model_p, ctypes.c_char_p]
    libllama.llama_adapter_lora_init.restype = llama_adapter_lora_p
    return libllama.llama_adapter_lora_init(model, path_lora.encode('utf-8'))

def llama_adapter_lora_set(ctx: ptr[llama_context], adapter: llama_adapter_lora, scale: float) -> int:
    """Set a LoRA adapter for a context"""
    libllama.llama_set_adapter_lora.argtypes = [llama_context_p, llama_adapter_lora_p, ctypes.c_float]
    libllama.llama_set_adapter_lora.restype = ctypes.c_int
    return libllama.llama_set_adapter_lora(ctx, adapter, scale)

def llama_rm_adapter_lora(ctx: ptr[llama_context], adapter: ptr[llama_adapter_lora]) -> int:
    """Remove a LoRA adapter from a context"""
    libllama.llama_rm_adapter_lora.argtypes = [llama_context_p, llama_adapter_lora_p]
    libllama.llama_rm_adapter_lora.restype = ctypes.c_int
    return libllama.llama_rm_adapter_lora(ctx, adapter)

def llama_clear_adapter_lora(ctx: ptr[llama_context]) -> None:
    """Clear all LoRA adapters from a context"""
    libllama.llama_clear_adapter_lora.argtypes = [llama_context_p]
    libllama.llama_clear_adapter_lora.restype = None
    libllama.llama_clear_adapter_lora(ctx)

def llama_adapter_lora_free(adapter: ptr[llama_adapter_lora]) -> None:
    """Free a LoRA adapter"""
    libllama.llama_adapter_lora_free.argtypes = [llama_adapter_lora_p]
    libllama.llama_adapter_lora_free.restype = None
    libllama.llama_adapter_lora_free(adapter)

#
# Control vector
#

def llama_apply_adapter_cvec(ctx: ptr[llama_context], data: ctypes.c_void_p, len: int, n_embd: int, il_start: int, il_end: int) -> int:
    """Apply a control vector to a context"""
    libllama.llama_apply_adapter_cvec.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_apply_adapter_cvec.restype = ctypes.c_int
    return libllama.llama_apply_adapter_cvec(ctx, data, len, n_embd, il_start, il_end)

#
# KV cache
#

def llama_kv_self_clear(ctx: ptr[llama_context]) -> None:
    """Clear the KV cache"""
    libllama.llama_kv_self_clear.argtypes = [llama_context_p]
    libllama.llama_kv_self_clear.restype = None
    libllama.llama_kv_self_clear(ctx)

def llama_kv_self_seq_rm(ctx: ptr[llama_context], seq_id: int, p0: int, p1: int) -> bool:
    """Remove tokens from a sequence in the KV cache"""
    libllama.llama_kv_self_seq_rm.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_kv_self_seq_rm.restype = ctypes.c_bool
    return libllama.llama_kv_self_seq_rm(ctx, seq_id, p0, p1)

def llama_kv_self_seq_cp(ctx: ptr[llama_context], seq_id_src: int, seq_id_dst: int, p0: int, p1: int) -> None:
    """Copy tokens from one sequence to another in the KV cache"""
    libllama.llama_kv_self_seq_cp.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_kv_self_seq_cp.restype = None
    libllama.llama_kv_self_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1)

def llama_kv_self_seq_keep(ctx: ptr[llama_context], seq_id: int) -> None:
    """Keep only the tokens of a sequence in the KV cache"""
    libllama.llama_kv_self_seq_keep.argtypes = [llama_context_p, ctypes.c_int]
    libllama.llama_kv_self_seq_keep.restype = None
    libllama.llama_kv_self_seq_keep(ctx, seq_id)

def llama_kv_self_seq_add(ctx: ptr[llama_context], seq_id: int, p0: int, p1: int, delta: int) -> None:
    """Add a relative position to tokens in a sequence in the KV cache"""
    libllama.llama_kv_self_seq_add.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_kv_self_seq_add.restype = None
    libllama.llama_kv_self_seq_add(ctx, seq_id, p0, p1, delta)

def llama_kv_self_seq_div(ctx: ptr[llama_context], seq_id: int, p0: int, p1: int, d: int) -> None:
    """Divide the positions of tokens in a sequence in the KV cache by a factor"""
    libllama.llama_kv_self_seq_div.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_kv_self_seq_div.restype = None
    libllama.llama_kv_self_seq_div(ctx, seq_id, p0, p1, d)

def llama_kv_self_seq_pos_min(ctx: ptr[llama_context], seq_id: int) -> int:
    """Returns the smallest position present in the KV cache for the specified sequence. This is
    typically non-zero only for SWA caches. Return -1 if the sequence is empty."""
    libllama.llama_kv_self_seq_pos_min.argtypes = [llama_context_p, llama_seq_id]
    libllama.llama_kv_self_seq_pos_min.restype = llama_pos
    return libllama.llama_kv_self_seq_pos_min(ctx, seq_id)

def llama_kv_self_seq_pos_max(ctx: ptr[llama_context], seq_id: int) -> int:
    """Get the maximum position of a sequence in the KV cache. Return -1 if the sequence is
    empty."""
    libllama.llama_kv_self_seq_pos_max.argtypes = [llama_context_p, llama_seq_id]
    libllama.llama_kv_self_seq_pos_max.restype = llama_pos
    return libllama.llama_kv_self_seq_pos_max(ctx, seq_id)

def llama_kv_self_can_shift(ctx: ptr[llama_context]) -> bool:
    """Check if the context supports KV cache shifting"""
    libllama.llama_kv_self_can_shift.argtypes = [llama_context_p]
    libllama.llama_kv_self_can_shift.restype = ctypes.c_bool
    return libllama.llama_kv_self_can_shift(ctx)

#
# State / session management
#

def llama_state_get_size(ctx: ptr[llama_context]) -> int:
    """Get the size of the state in bytes"""
    libllama.llama_state_get_size.argtypes = [llama_context_p]
    libllama.llama_state_get_size.restype = ctypes.c_size_t
    return libllama.llama_state_get_size(ctx)

def llama_state_get_data(ctx: ptr[llama_context], dst: ctypes.c_void_p, size: int) -> int:
    """Copy the state to a destination address"""
    libllama.llama_state_get_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_size_t]
    libllama.llama_state_get_data.restype = ctypes.c_size_t
    return libllama.llama_state_get_data(ctx, dst, size)

def llama_state_set_data(ctx: ptr[llama_context], src: ctypes.c_void_p, size: int) -> int:
    """Set the state from a source address"""
    libllama.llama_state_set_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_size_t]
    libllama.llama_state_set_data.restype = ctypes.c_size_t
    return libllama.llama_state_set_data(ctx, src, size)

def llama_state_load_file(ctx: ptr[llama_context], path_session: str, tokens_out: ptr[llama_token], n_token_capacity: int, n_token_count_out: ptr[ctypes.c_size_t]) -> bool:
    """Load the context state and session tokens from a file.

    :param path_session: The path to the session file
    :param tokens_out: A buffer to receive the loaded session tokens
    :param n_token_capacity: The maximum capacity of the `tokens_out` buffer
    :param n_token_count_out: A pointer to store the actual number of tokens loaded
    :return: True on success, False on failure"""
    libllama.llama_state_load_file.argtypes = [llama_context_p, ctypes.c_char_p, ctypes.POINTER(llama_token), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    libllama.llama_state_load_file.restype = ctypes.c_bool
    return libllama.llama_state_load_file(ctx, path_session.encode('utf-8'), tokens_out, ctypes.c_size_t(n_token_capacity), n_token_count_out)

def llama_state_save_file(ctx: ptr[llama_context], path_session: str, tokens: list[int]) -> bool:
    """Save the context state and current tokens to a file.

    :param path_session: The path to save the session file
    :param tokens: A list of tokens representing the current prompt/session
    :param n_token_count: The number of tokens in the `tokens` list
    :return: True on success, False on failure"""
    libllama.llama_state_save_file.argtypes = [llama_context_p, ctypes.c_char_p, ctypes.POINTER(llama_token), ctypes.c_size_t]
    libllama.llama_state_save_file.restype = ctypes.c_bool
    n_token_count = len(tokens)
    return libllama.llama_state_save_file(ctx, path_session.encode('utf-8'), (llama_token * n_token_count)(*tokens), ctypes.c_size_t(n_token_count))

def llama_state_seq_get_size(ctx: ptr[llama_context], llama_seq_id: int) -> int:
    """Get the exact size needed to copy the KV cache of a single sequence"""
    libllama.llama_state_seq_get_size.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_state_seq_get_size.restype = ctypes.c_size_t
    return libllama.llama_state_seq_get_size(ctx, llama_seq_id)

def llama_state_seq_get_data(ctx: ptr[llama_context], dst: ctypes.c_void_p, size: int, seq_id: int) -> int:
    """Copy the KV cache of a single sequence into the specified buffer"""
    libllama.llama_state_seq_get_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32]
    libllama.llama_state_seq_get_data.restype = ctypes.c_size_t
    return libllama.llama_state_seq_get_data(ctx, dst, size, seq_id)

def llama_state_seq_set_data(ctx: ptr[llama_context], src: ctypes.c_void_p, size: int, dest_seq_id: int) -> int:
    """Copy the sequence data (originally copied with `llama_state_seq_get_data`)
    into the specified sequence
    
    Returns:
    - Positive: Ok
    - Zero: Failed to load"""
    libllama.llama_state_seq_set_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32]
    libllama.llama_state_seq_set_data.restype = ctypes.c_size_t
    return libllama.llama_state_seq_set_data(ctx, src, size, dest_seq_id)

def llama_state_seq_save_file(ctx: ptr[llama_context], filepath: str, seq_id: int, tokens: ptr[ctypes.c_int32], n_token_count: int) -> int:
    libllama.llama_state_seq_save_file.argtypes = [llama_context_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
    libllama.llama_state_seq_save_file.restype = ctypes.c_size_t
    return libllama.llama_state_seq_save_file(ctx, filepath.encode('utf-8'), seq_id, tokens, n_token_count)

def llama_state_seq_load_file(ctx: ptr[llama_context], filepath: str, dest_seq_id: int, tokens_out: ptr[ctypes.c_int32], n_token_capacity: int, n_token_count_out: ptr[ctypes.c_size_t]) -> int:
    libllama.llama_state_seq_load_file.argtypes = [llama_context_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    libllama.llama_state_seq_load_file.restype = ctypes.c_size_t
    return libllama.llama_state_seq_load_file(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)

#
# Batch
#

def llama_batch_get_one(tokens: ptr[llama_token], n_tokens: int) -> llama_batch:
    """AVOID USING

    This function will be deprecated and removed at some point. Refer to:
    https://github.com/ggml-org/llama.cpp/issues/6475#issuecomment-2040350410

    Return batch for single sequence of tokens"""
    log(
        f'you are using libllama.llama_batch_get_one which will be deprecated '
        f'and removed at some point. you should use libllama.llama_batch_init '
        f'instead', 2
    )
    libllama.llama_batch_get_one.argtypes = [ctypes.POINTER(llama_token), ctypes.c_int32]
    libllama.llama_batch_get_one.restype = llama_batch
    return libllama.llama_batch_get_one(tokens, n_tokens)

def llama_batch_init(n_tokens: int, embd: int, n_seq_max: int) -> llama_batch:
    """Allocate a batch of tokens"""
    libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    libllama.llama_batch_init.restype = llama_batch
    return libllama.llama_batch_init(n_tokens, embd, n_seq_max)

def llama_batch_free(batch: llama_batch) -> None:
    """Frees a batch of tokens"""
    libllama.llama_batch_free.argtypes = [llama_batch]
    libllama.llama_batch_free.restype = None
    libllama.llama_batch_free(batch)

#
# Encode / decode
#

def llama_encode(ctx: ptr[llama_context], batch: llama_batch) -> int:
    """Process a batch of tokens.
    In contrast to llama_decode() - this call does not use KV cache.
    For encoder-decoder contexts, processes the batch using the encoder.
    Can store the encoder output internally for later use by the decoder's cross-attention
    layers.
    Returns:
    - 0:
        success
    - < 0:
        error. the KV cache state is restored to the state before this call"""
    libllama.llama_encode.argtypes = [llama_context_p, llama_batch]
    libllama.llama_encode.restype = ctypes.c_int32
    return libllama.llama_encode(ctx, batch)

def llama_decode(ctx: ptr[llama_context], batch: llama_batch) -> int:
    """Process a batch of tokens.
    Requires KV cache.
    For encoder-decoder contexts, processes the batch using the decoder.
    Positive return values does not mean a fatal error, but rather a warning.
    Upon non-zero return values, the KV cache state is restored to the state before this call

    Returns:
    - 0:
        success
    - 1:
        could not find a KV slot for the batch (try reducing the size of
        the batch or increase the context)
    - 2:
        aborted
    - -1:
        invalid input batch
    - < -1:
        error"""
    # TODO: wrap the function call in try/except for KeyboardInterrupt? something like that?
    libllama.llama_decode.argtypes = [llama_context_p, llama_batch]
    libllama.llama_decode.restype = ctypes.c_int32
    return libllama.llama_decode(ctx, batch)

def llama_set_n_threads(ctx: ptr[llama_context], n_threads: int, n_threads_batch: int) -> None:
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
    """Set whether to use embeddings mode or not"""
    libllama.llama_set_embeddings.argtypes = [llama_context_p, ctypes.c_bool]
    libllama.llama_set_embeddings.restype = None
    libllama.llama_set_embeddings(ctx, embeddings)

def llama_set_warmup(ctx: ptr[llama_context], warmup: bool) -> None:
    libllama.llama_set_warmup.argtypes = [llama_context_p, ctypes.c_bool]
    libllama.llama_set_warmup.restype = None
    libllama.llama_set_warmup(ctx, warmup)

def llama_set_causal_attn(ctx: ptr[llama_context], causal_attn: bool) -> None:
    """Set whether to use causal attention or not"""
    libllama.llama_set_causal_attn.argtypes = [llama_context_p, ctypes.c_bool]
    libllama.llama_set_causal_attn.restype = None
    libllama.llama_set_causal_attn(ctx, causal_attn)

def llama_set_abort_callback(ctx: ptr[llama_context], abort_callback: ptr, abort_callback_data: ptr) -> None:
    """Set an abort callback"""
    libllama.llama_set_abort_callback.argtypes = [llama_context_p, abort_callback_functype, ctypes.c_void_p]
    libllama.llama_set_abort_callback.restype = None
    libllama.llama_set_abort_callback(ctx, abort_callback, abort_callback_data)

def llama_synchronize(ctx: ptr[llama_context]) -> None:
    """Wait until all computations are finished

    Not necessary to call explicitly in most cases"""
    libllama.llama_synchronize.argtypes = [llama_context_p]
    libllama.llama_synchronize.restype = None
    libllama.llama_synchronize(ctx)

def llama_get_logits(ctx: ptr[llama_context]) -> ptr[ctypes.c_float]:
    """Get the token logits obtained from the last call to llama_decode()
    
    Rows: number of tokens for which llama_batch.logits[i] != 0
    Cols: n_vocab"""
    libllama.llama_get_logits.argtypes = [llama_context_p]
    libllama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_logits(ctx)

def llama_get_logits_ith(ctx: ptr[llama_context], i: int) -> ptr[ctypes.c_float]:
    """Get the logits for the ith token"""
    libllama.llama_get_logits_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_logits_ith(ctx, i)

def llama_get_embeddings(ctx: ptr[llama_context]) -> ptr[ctypes.c_float]:
    """Get all output token embeddings"""
    libllama.llama_get_embeddings.argtypes = [llama_context_p]
    libllama.llama_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_embeddings(ctx)

def llama_get_embeddings_ith(ctx: ptr[llama_context], i: int) -> ptr[ctypes.c_float]:
    """Get the embeddings for the ith token"""
    libllama.llama_get_embeddings_ith.argtypes = [llama_context_p, ctypes.c_int32]
    libllama.llama_get_embeddings_ith.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_embeddings_ith(ctx, i)

def llama_get_embeddings_seq(ctx: ptr[llama_context], seq_id: int) -> ptr[ctypes.c_float]:
    """Get the embeddings for a sequence id"""
    libllama.llama_get_embeddings_seq.argtypes = [llama_context_p, llama_seq_id]
    libllama.llama_get_embeddings_seq.restype = ctypes.POINTER(ctypes.c_float)
    return libllama.llama_get_embeddings_seq(ctx, seq_id)

#
# Vocab
#

def llama_vocab_get_text(vocab: llama_vocab, token: int) -> bytes:
    """Get the text representation of a token (as bytes)"""
    libllama.llama_vocab_get_text.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_get_text.restype = ctypes.c_char_p
    return libllama.llama_vocab_get_text(vocab, token)

def llama_vocab_get_score(vocab: llama_vocab, token: int) -> float:
    """Get the score of a token"""
    libllama.llama_vocab_get_score.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_get_score.restype = ctypes.c_float
    return libllama.llama_vocab_get_score(vocab, token)

def llama_vocab_get_attr(vocab: llama_vocab, token: int) -> int:
    """Get the attributes of a token. See `LlamaTokenAttr`."""
    libllama.llama_vocab_get_attr.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_get_attr.restype = ctypes.c_int # LlamaTokenAttr
    return libllama.llama_vocab_get_attr(vocab, token)

def llama_vocab_is_eog(vocab: llama_vocab, token: int) -> bool:
    """Check if the token is supposed to end generation (end-of-generation, eg. EOS,
    EOT, etc.)"""
    libllama.llama_vocab_is_eog.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_is_eog.restype = ctypes.c_bool
    return libllama.llama_vocab_is_eog(vocab, token)

def llama_vocab_is_control(vocab: llama_vocab, token: int) -> bool:
    """Identify if token ID is a control token or a render-able token"""
    libllama.llama_vocab_is_control.argtypes = [llama_vocab_p, llama_token]
    libllama.llama_vocab_is_control.restype = ctypes.c_bool
    return libllama.llama_vocab_is_control(vocab, token)

#
# Vocab (special tokens)
#

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

def llama_vocab_get_add_bos(vocab: ptr[llama_vocab]) -> bool:
    """Whether BOS token should be added to tokenizations"""
    libllama.llama_vocab_get_add_bos.argtypes = [llama_vocab_p]
    libllama.llama_vocab_get_add_bos.restype = ctypes.c_bool
    return libllama.llama_vocab_get_add_bos(vocab)

def llama_vocab_get_add_eos(vocab: ptr[llama_vocab]) -> bool:
    """Whether EOS token should be added to tokenizations"""
    libllama.llama_vocab_get_add_eos.argtypes = [llama_vocab_p]
    libllama.llama_vocab_get_add_eos.restype = ctypes.c_bool
    return libllama.llama_vocab_get_add_eos(vocab)

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
# Tokenization
#

def llama_tokenize(vocab: ptr[llama_vocab], text: bytes, text_len: int, tokens: ptr[llama_token], n_tokens_max: int, add_special: bool, parse_special: bool) -> int:
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
    returned."""
    libllama.llama_tokenize.argtypes = [llama_vocab_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(llama_token), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]
    libllama.llama_tokenize.restype = ctypes.c_int32
    return libllama.llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special)

def llama_token_to_piece(vocab: ptr[llama_vocab], token: int, buf: ctypes.c_char_p, length: int, lstrip: int, special: bool) -> int:
    """Convert a single token to a piece of text"""
    libllama.llama_token_to_piece.argtypes = [llama_vocab_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
    libllama.llama_token_to_piece.restype = ctypes.c_int32
    return libllama.llama_token_to_piece(vocab, token, buf, length, lstrip, special)

def llama_detokenize(vocab: ptr[llama_vocab], tokens: ptr[llama_token], n_tokens: int, text: ctypes.c_char_p, text_len_max: int, remove_special: bool, unparse_special: bool) -> int:
    """Convert the provided tokens into text.
    - Returns the number of chars/bytes on success, no more than text_len_max.
    - Returns a negative number on failure - the number of chars/bytes that would have been
    returned."""
    libllama.llama_detokenize.argtypes = [llama_vocab_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]
    libllama.llama_detokenize.restype = ctypes.c_int32
    return libllama.llama_detokenize(vocab, tokens, n_tokens, text, text_len_max, remove_special, unparse_special)

#
# Chat templating
#

def llama_chat_apply_template(tmpl: ptr[ctypes.c_char], chat: ptr[llama_chat_message], n_msg: int, add_ass: bool, buf: ptr[ctypes.c_char], length: int):
    libllama.llama_chat_apply_template.argtypes = [ctypes.c_char_p, llama_chat_message_p, ctypes.c_size_t, ctypes.c_bool, ctypes.c_char_p, ctypes.c_int32]
    libllama.llama_chat_apply_template.restype = ctypes.c_int32
    return libllama.llama_chat_apply_template(tmpl, chat, n_msg, add_ass, buf, length)

def llama_chat_builtin_templates(output: ptr[ctypes.c_char_p], len: int):
    libllama.llama_chat_builtin_templates.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
    libllama.llama_chat_builtin_templates.restype = ctypes.c_int32
    return libllama.llama_chat_builtin_templates(output, len)

#
# Sampling
#

class llama_sampler_i(ctypes.Structure):
    _fields_ = [
        ("name",   ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)                             ), # can be NULL
        ("accept", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32)                        ), # can be NULL
        ("apply",  ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(llama_token_data_array))), # required
        ("reset",  ctypes.CFUNCTYPE(None, ctypes.c_void_p)                                        ), # can be NULL
        ("clone",  ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)                             ), # can be NULL if ctx is NULL
        ("free",   ctypes.CFUNCTYPE(None, ctypes.c_void_p)                                        )  # can be NULL if ctx is NULL
    ]

llama_sampler_i_p = ctypes.POINTER(llama_sampler_i)

class llama_sampler(ctypes.Structure):
    _fields_ = [
        ("iface", ctypes.POINTER(llama_sampler_i)),
        ("ctx",   ctypes.c_void_p                )
    ]

llama_sampler_p = ctypes.POINTER(llama_sampler)

def llama_sampler_init(iface: ptr[llama_sampler_i], ctx: ptr) -> ptr[llama_sampler]:
    libllama.llama_sampler_init.argtypes = [llama_sampler_i_p, ctypes.c_void_p] # llama_sampler_context_t
    libllama.llama_sampler_init.restype = llama_sampler_p
    return libllama.llama_sampler_init(iface, ctx)

def llama_sampler_name(smpl: ptr[llama_sampler]) -> str:
    """Get the name of a sampler"""
    libllama.llama_sampler_name.argtypes = [llama_sampler_p]
    libllama.llama_sampler_name.restype = ctypes.c_char_p
    return libllama.llama_sampler_name(smpl).decode('utf-8')

def llama_sampler_accept(smpl: ptr[llama_sampler], token: int) -> None:
    """Accept a token sampled by a sampler"""
    libllama.llama_sampler_accept.argtypes = [llama_sampler_p, ctypes.c_int]
    libllama.llama_sampler_accept.restype = None
    libllama.llama_sampler_accept(smpl, token)

def llama_sampler_apply(smpl: ptr[llama_sampler], cur_p: llama_token_data_array) -> None:
    """Apply a sampler to a token data array"""
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

def llama_sampler_free(smpl: ptr[llama_sampler]) -> None:
    """Free a sampler
    
    NOTE: Do not free if the sampler has been added to a llama_sampler_chain
    (via llama_sampler_chain_add)"""
    libllama.llama_sampler_free.argtypes = [llama_sampler_p]
    libllama.llama_sampler_free.restype = None
    libllama.llama_sampler_free(smpl)

#
# Sampler chain
#

def llama_sampler_chain_init(params: llama_sampler_chain_params) -> ptr[llama_sampler]:
    """Initialize a sampler chain"""
    libllama.llama_sampler_chain_init.argtypes = [llama_sampler_chain_params_p]
    libllama.llama_sampler_chain_init.restype = llama_sampler_p
    return libllama.llama_sampler_chain_init(params)

def llama_sampler_chain_add(chain: ptr[llama_sampler], smpl: ptr[llama_sampler]) -> None:
    """Add a sampler to a sampler chain
    
    Takes ownership of the sampler object and will free it when llama_sampler_free is called"""
    libllama.llama_sampler_chain_add.argtypes = [llama_sampler_p, llama_sampler_p]
    libllama.llama_sampler_chain_add.restype = None
    libllama.llama_sampler_chain_add(chain, smpl)

def llama_sampler_chain_get(chain: ptr[llama_sampler], i: int) -> ptr[llama_sampler]:
    """Get a sampler from a sampler chain"""
    libllama.llama_sampler_chain_get.argtypes = [llama_sampler_p, ctypes.c_int]
    libllama.llama_sampler_chain_get.restype = llama_sampler_p
    return libllama.llama_sampler_chain_get(chain, i)

def llama_sampler_chain_n(chain: ptr[llama_sampler]) -> int:
    """Get the number of samplers in a sampler chain"""
    libllama.llama_sampler_chain_n.argtypes = [llama_sampler_p]
    libllama.llama_sampler_chain_n.restype = ctypes.c_int
    return libllama.llama_sampler_chain_n(chain)

def llama_sampler_chain_remove(chain: ptr[llama_sampler], i: int) -> ptr[llama_sampler]:
    """Remove a sampler from a sampler chain
    
    After removing a sampler, the chain will no longer own it, and it will not be freed when
    the chain is freed.
    """
    libllama.llama_sampler_chain_remove.argtypes = [llama_sampler_p, ctypes.c_int]
    libllama.llama_sampler_chain_remove.restype = llama_sampler_p
    return libllama.llama_sampler_chain_remove(chain, i)

#
# Samplers
#

def llama_sampler_init_greedy() -> ptr[llama_sampler]:
    """Initialize a greedy sampler"""
    libllama.llama_sampler_init_greedy.argtypes = []
    libllama.llama_sampler_init_greedy.restype = llama_sampler_p
    return libllama.llama_sampler_init_greedy()

def llama_sampler_init_dist(seed: int) -> ptr[llama_sampler]:
    """Initialize a distribution sampler"""
    libllama.llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
    libllama.llama_sampler_init_dist.restype = llama_sampler_p
    return libllama.llama_sampler_init_dist(seed)

def llama_sampler_init_top_k(k: int) -> ptr[llama_sampler]:
    """Initialize a top-K sampler. Setting k <= 0 makes this a no-op."""
    libllama.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    libllama.llama_sampler_init_top_k.restype = llama_sampler_p
    return libllama.llama_sampler_init_top_k(k)

def llama_sampler_init_top_p(p: float, min_keep: int) -> ptr[llama_sampler]:
    """Initialize a top-p sampler"""
    libllama.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    libllama.llama_sampler_init_top_p.restype = llama_sampler_p
    return libllama.llama_sampler_init_top_p(p, min_keep)

def llama_sampler_init_min_p(p: float, min_keep: int) -> ptr[llama_sampler]:
    """Initialize a min-p sampler"""
    libllama.llama_sampler_init_min_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    libllama.llama_sampler_init_min_p.restype = llama_sampler_p
    return libllama.llama_sampler_init_min_p(p, min_keep)

def llama_sampler_init_typical(p: float, min_keep: int) -> ptr[llama_sampler]:
    """Initialize a locally typical sampler"""
    libllama.llama_sampler_init_typical.argtypes = [ctypes.c_float, ctypes.c_size_t]
    libllama.llama_sampler_init_typical.restype = llama_sampler_p
    return libllama.llama_sampler_init_typical(p, min_keep)

def llama_sampler_init_temp(t: float) -> ptr[llama_sampler]:
    """Initialize a temperature sampler
    
    When `t` <= 0.0, the maximum logit is kept at it's original value, the rest are set to
    -inf"""
    libllama.llama_sampler_init_temp.argtypes = [ctypes.c_float]
    libllama.llama_sampler_init_temp.restype = llama_sampler_p
    return libllama.llama_sampler_init_temp(t)

def llama_sampler_init_temp_ext(t: float, delta: float, exponent: float) -> ptr[llama_sampler]:
    """Initialize an dynamic temperature / entropy sampler"""
    libllama.llama_sampler_init_temp_ext.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    libllama.llama_sampler_init_temp_ext.restype = llama_sampler_p
    return libllama.llama_sampler_init_temp_ext(t, delta, exponent)

def llama_sampler_init_xtc(p: float, t: float, min_keep: int, seed: int) -> ptr[llama_sampler]:
    """Initialize an XTC sampler"""
    libllama.llama_sampler_init_xtc.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_size_t, ctypes.c_uint32]
    libllama.llama_sampler_init_xtc.restype = llama_sampler_p
    return libllama.llama_sampler_init_xtc(p, t, min_keep, seed)

def llama_sampler_init_top_n_sigma(n: float) -> ptr[llama_sampler]:
    """Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need"
    https://arxiv.org/pdf/2411.07641"""
    libllama.llama_sampler_init_top_n_sigma.argtypes = [ctypes.c_float]
    libllama.llama_sampler_init_top_n_sigma.restype = llama_sampler_p
    return libllama.llama_sampler_init_top_n_sigma(n)

def llama_sampler_init_mirostat(n_vocab: int, seed: int, tau: float, eta: float, m: int) -> ptr[llama_sampler]:
    """Initialize a Mirostat sampler"""
    libllama.llama_sampler_init_mirostat.argtypes = [ctypes.c_int32, ctypes.c_uint32, ctypes.c_float, ctypes.c_float, ctypes.c_int32]
    libllama.llama_sampler_init_mirostat.restype = llama_sampler_p
    return libllama.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)

def llama_sampler_init_mirostat_v2(seed: int, tau: float, eta: float) -> ptr[llama_sampler]:
    """Initialize a Mirostat v2 sampler"""
    libllama.llama_sampler_init_mirostat_v2.argtypes = [ctypes.c_uint32, ctypes.c_float, ctypes.c_float]
    libllama.llama_sampler_init_mirostat_v2.restype = llama_sampler_p
    return libllama.llama_sampler_init_mirostat_v2(seed, tau, eta)

def llama_sampler_init_grammar(vocab: ptr[llama_vocab], grammar_str: str, grammar_root: str) -> ptr[llama_sampler]:
    """Intializes a GBNF grammar"""
    libllama.llama_sampler_init_grammar.argtypes = [llama_vocab_p, ctypes.c_char_p, ctypes.c_char_p]
    libllama.llama_sampler_init_grammar.restype = llama_sampler_p
    return libllama.llama_sampler_init_grammar(vocab, grammar_str.encode('utf-8'), grammar_root.encode('utf-8'))

def llama_sampler_init_grammar_lazy_patterns(vocab: ptr[llama_vocab], grammar_str: str, grammar_root: str, trigger_patterns: list[str], num_trigger_patterns: int, trigger_tokens: ptr[llama_token], num_trigger_tokens: int) -> ptr[llama_sampler]:
    """Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639"""
    libllama.llama_sampler_init_grammar_lazy_patterns.argtypes = [llama_vocab_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t, ctypes.POINTER(llama_token), ctypes.c_size_t]
    libllama.llama_sampler_init_grammar_lazy_patterns.restype = llama_sampler_p
    c_trigger_patterns = (ctypes.c_char_p * num_trigger_patterns)(*[pattern.encode('utf-8') for pattern in trigger_patterns])
    return libllama.llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.encode('utf-8'), grammar_root.encode('utf-8'), c_trigger_patterns, num_trigger_patterns, trigger_tokens, num_trigger_tokens)

def llama_sampler_init_penalties(penalty_last_n: int, penalty_repeat: float, penalty_freq: float, penalty_present: float) -> ptr[llama_sampler]:
    """Initialize a penalties sampler"""
    libllama.llama_sampler_init_penalties.argtypes = [ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    libllama.llama_sampler_init_penalties.restype = llama_sampler_p
    return libllama.llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)

def llama_sampler_init_dry(vocab: ptr[llama_vocab], n_ctx_train: int, dry_multiplier: float, dry_base: float, dry_allowed_length: int, dry_penalty_last_n: int, seq_breakers: ptr[ctypes.c_char_p], num_breakers: int) -> ptr[llama_sampler]:
    """Initialize a DRY sampler"""
    libllama.llama_sampler_init_dry.argtypes = [llama_vocab_p, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
    libllama.llama_sampler_init_dry.restype = llama_sampler_p
    return libllama.llama_sampler_init_dry(vocab, n_ctx_train, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers)

def llama_sampler_init_logit_bias(n_vocab: int, n_logit_bias: int, logit_bias: ptr[llama_logit_bias]) -> ptr[llama_sampler]:
    """Initialize a logit bias sampler"""
    libllama.llama_sampler_init_logit_bias.argtypes = [ctypes.c_int32, ctypes.c_int32, llama_logit_bias_p]
    libllama.llama_sampler_init_logit_bias.restype = llama_sampler_p
    return libllama.llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias)

def llama_sampler_init_infill(vocab: ptr[llama_vocab]) -> ptr[llama_sampler]:
    """Initialize an infill sampler
    
    This sampler is meant to be used for fill-in-the-middle infilling. It's supposed to be used
    after top_k + top_p sampling"""
    libllama.llama_sampler_init_infill.argtypes = [llama_vocab_p]
    libllama.llama_sampler_init_infill.restype = llama_sampler_p
    return libllama.llama_sampler_init_infill(vocab)

def llama_sampler_get_seed(smpl: ptr[llama_sampler]) -> int:
    """Get the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise"""
    libllama.llama_sampler_get_seed.argtypes = [llama_sampler_p]
    libllama.llama_sampler_get_seed.restype = ctypes.c_uint32
    return libllama.llama_sampler_get_seed(smpl)

def llama_sampler_sample(smpl: ptr[llama_sampler], ctx: ptr[llama_context], idx: int) -> int:
    """Sample and accept a token from the idx-th output of the last evaluation"""
    libllama.llama_sampler_sample.argtypes = [llama_sampler_p, llama_context_p, ctypes.c_int32]
    libllama.llama_sampler_sample.restype = llama_token
    return libllama.llama_sampler_sample(smpl, ctx, idx)

#
# Model split
#

def llama_split_path(split_path: ctypes.c_char_p, maxlen: int, path_prefix: str, split_no: int, split_count: int) -> int:
    """Build a split GGUF final path for a chunk"""
    libllama.llama_split_path.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    libllama.llama_split_path.restype = ctypes.c_int
    return libllama.llama_split_path(split_path, maxlen, path_prefix.encode('utf-8'), split_no, split_count)

def llama_split_prefix(split_prefix: ctypes.c_char_p, maxlen: int, split_path: str, split_no: int, split_count: int) -> int:
    """Extract the path prefix from a split path"""
    libllama.llama_split_prefix.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    libllama.llama_split_prefix.restype = ctypes.c_int
    return libllama.llama_split_prefix(split_prefix, maxlen, split_path.encode('utf-8'), split_no, split_count)

#
# Print system info
#

def llama_print_system_info() -> None:
    """Get system information"""
    libllama.llama_print_system_info.argtypes = []
    libllama.llama_print_system_info.restype = ctypes.c_char_p
    text = libllama.llama_print_system_info()
    text = text.decode()
    print(text, file=sys.stderr, flush=True)

#
# Log callback
#

def llama_log_set(log_callback: ctypes.c_void_p, user_data: ctypes.c_void_p) -> None:
    """Set a callback for logging events"""
    libllama.llama_log_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_log_set.restype = None
    libllama.llama_log_set(log_callback, user_data)

#
# Performance utils
#

class llama_perf_context_data(ctypes.Structure):
    _fields_ = [
        ("t_start_ms",  ctypes.c_double),
        ("t_load_ms",   ctypes.c_double),
        ("t_p_eval_ms", ctypes.c_double),
        ("t_eval_ms",   ctypes.c_double),
        ("n_p_eval",    ctypes.c_int32 ),
        ("n_eval",      ctypes.c_int32 )
    ]

llama_perf_context_data_p = ctypes.POINTER(llama_perf_context_data)

class llama_perf_sampler_data(ctypes.Structure):
    _fields_ = [
        ("t_sample_ms", ctypes.c_double),
        ("n_sample",    ctypes.c_int32 )
    ]

llama_perf_sampler_data_p = ctypes.POINTER(llama_perf_sampler_data)

# NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.

def llama_perf_context(ctx: ptr[llama_context]) -> llama_perf_context_data:
    """AVOID USING

    Get performance data for a context"""
    libllama.llama_perf_context.argtypes = [llama_context_p]
    libllama.llama_perf_context.restype = llama_perf_context_data
    return libllama.llama_perf_context(ctx)

def llama_perf_context_print(ctx: ptr[llama_context]) -> None:
    """AVOID USING

    Print performance data for a context"""
    libllama.llama_perf_context_print.argtypes = [llama_context_p]
    libllama.llama_perf_context_print.restype = ctypes.c_char_p
    results_str = libllama.llama_perf_context_print(ctx)
    print(results_str, file=sys.stderr, flush=True)

def llama_perf_context_reset(ctx: ptr[llama_context]) -> None:
    """AVOID USING
    
    Reset performance data for a context"""
    libllama.llama_perf_context_reset.argtypes = [llama_context_p]
    libllama.llama_perf_context_reset.restype = None
    libllama.llama_perf_context_reset(ctx)

# NOTE: the following work only with samplers constructed via llama_sampler_chain_init

def llama_perf_sampler(smpl: ptr[llama_sampler]) -> llama_perf_sampler_data:
    """Get performance data for a sampler"""
    libllama.llama_perf_sampler.argtypes = [llama_sampler_p]
    libllama.llama_perf_sampler.restype = llama_perf_sampler_data
    return libllama.llama_perf_sampler(smpl)

def llama_perf_sampler_print(smpl: ptr[llama_sampler]) -> None:
    """Print performance data for a sampler"""
    libllama.llama_perf_sampler_print.argtypes = [llama_sampler_p]
    libllama.llama_perf_sampler_print.restype = ctypes.c_char_p
    results_str = libllama.llama_perf_sampler_print(smpl)
    print(results_str, file=sys.stderr, flush=True)

def llama_perf_sampler_reset(smpl: ptr[llama_sampler]) -> None:
    """Reset performance data for a sampler"""
    libllama.llama_perf_sampler_reset.argtypes = [llama_sampler_p]
    libllama.llama_perf_sampler_reset.restype = None
    libllama.llama_perf_sampler_reset(smpl)

# NOTE: training functions are not included

#
# End of LLAMA_API
#

class _internals:
    """### INTERNAL
    
    Helper functions used by the `llama` and `sampling` modules."""

    # these will be initialized when libllama is initialized
    greedy_sampler: Optional[ptr[llama_sampler]]          = None
    detok_buffer:   Optional[ctypes.Array[ctypes.c_char]] = None

    class LogitBiasArrayType:
        """Type hint for a `ctypes.Array[llama_logit_bias]` of arbitrary length"""

    @staticmethod
    def decode_pp(
        ctx: ptr[llama_context],
        pos: int,
        tokens: list[int],
        n_tokens: int
    ) -> None:
        """### INTERNAL

        Decode with batch size > 1 (prompt processing)."""
        batch = llama_batch_init(n_tokens=n_tokens, embd=0, n_seq_max=1)
        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            batch.token[i] = tokens[i]
            batch.pos[i] = pos + i
            batch.seq_id[i][0] = 0
            batch.n_seq_id[i] = 1
            batch.logits[i] = C_FALSE
        batch.logits[n_tokens - 1] = C_TRUE
        ret = llama_decode(ctx, batch)
        llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_pp: llama_decode failed with status code {ret}')

    @staticmethod
    def decode_tg(
        ctx: ptr[llama_context],
        pos: int,
        token: int
    ) -> None:
        """### INTERNAL

        Decode with batch size == 1 (text generation)."""
        batch = llama_batch_init(n_tokens=1, embd=0, n_seq_max=1)
        batch.n_tokens = 1
        batch.token[0] = token
        batch.pos[0] = pos
        batch.seq_id[0][0] = 0
        batch.n_seq_id[0] = 1
        batch.logits[0] = C_TRUE
        ret = llama_decode(ctx, batch)
        llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f'decode_tg: llama_decode failed with status code {ret}')

    @staticmethod
    def decode_pp_with_logits(
        ctx: ptr[llama_context],
        pos: int,
        tokens: list[int],
        n_tokens: int,
        n_vocab: int
    ) -> np.ndarray:
        """### INTERNAL

        Decode with batch size > 1 (prompt processing).
        
        Return logits for all tokens in the batch. The returned logits have shape 
        `(n_tokens, n_vocab)`.
        
        The logits at index `i` are the predictions for the token at index `i + 1` in the batch.
        
        The last column in the array (`logits[n_tokens]`) contains the logits for the inferred
        next token."""
        batch = llama_batch_init(n_tokens=n_tokens, embd=0, n_seq_max=1)
        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            batch.token[i] = tokens[i]
            batch.pos[i] = pos + i
            batch.seq_id[i][0] = 0
            batch.n_seq_id[i] = 1
            batch.logits[i] = C_TRUE
        ret = llama_decode(ctx, batch)
        llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(
                f'decode_pp_with_logits: llama_decode failed with status code {ret}'
            )
        c_logits = llama_get_logits(ctx)
        logits: np.ndarray = np.ctypeslib.as_array(c_logits, shape=(n_tokens, n_vocab))
        return logits

    @staticmethod
    def decode_tg_with_logits(
        ctx: ptr[llama_context],
        pos: int,
        token: int,
        n_vocab: int
    ) -> np.ndarray:
        """### INTERNAL

        Decode with batch size == 1 (text generation).
        
        Return the logits for the inferred next token."""
        batch = llama_batch_init(n_tokens=1, embd=0, n_seq_max=1)
        batch.n_tokens = 1
        batch.token[0] = token
        batch.pos[0] = pos
        batch.seq_id[0][0] = 0
        batch.n_seq_id[0] = 1
        batch.logits[0] = C_TRUE
        ret = llama_decode(ctx, batch)
        llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(
                f'decode_tg_with_logits: llama_decode failed with status code {ret}'
            )
        c_logits = llama_get_logits(ctx)
        logits: np.ndarray = np.ctypeslib.as_array(c_logits, shape=(1, n_vocab))[0]
        return logits
    
    @staticmethod
    def decode_embd(
        ctx: ptr[llama_context],
        embeddings: np.ndarray,
        pos: int, # starting position for the first embedding (usually 0)
        seq_id: int = 0
    ) -> None:
        """### INTERNAL

        Decode a batch of embeddings. The shape of the embeddings must be (n_tokens, n_embd)."""
        n_prompt_tokens = embeddings.shape[0]
        if n_prompt_tokens == 0:
            return None

        n_ctx = llama_n_ctx(ctx)
        if n_prompt_tokens > n_ctx:
            raise ValueError(
                f'decode_embd: length of embeddings exceeds context length ({n_prompt_tokens} '
                f'> {n_ctx})'
            )

        n_embd = llama_model_n_embd(llama_get_model(ctx))

        if embeddings.shape[1] != n_embd:
            raise ValueError(
                f"decode_embd: embedding dimension mismatch: expected {n_embd}, got "
                f"{embeddings.shape[1]}"
            )

        batch = llama_batch_init(n_tokens=n_prompt_tokens, embd=n_embd, n_seq_max=1)
        batch.n_tokens = n_prompt_tokens
        batch.token = NULL

        ctypes.memmove(batch.embd, embeddings.ctypes.data, embeddings.nbytes)

        for i in range(n_prompt_tokens):
            batch.pos[i] = pos + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = seq_id
            batch.logits[i] = C_FALSE # by default, don't compute logits ...

        # ... except for the last embedding
        batch.logits[n_prompt_tokens - 1] = C_TRUE

        ret = llama_decode(ctx, batch)
        llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(
                f'decode_embd: llama_decode failed with status code {ret}'
            )
    
    # TODO: add internal functions that handle decoding of multiple sequences
    
    @staticmethod
    def get_logits(ctx: ptr[llama_context], n_vocab: int) -> np.ndarray:
        """### INTERNAL
        
        Get the logits for the last token decoded."""
        c_last_token_logits = llama_get_logits_ith(ctx, -1)
        return np.ctypeslib.as_array(c_last_token_logits, shape=(1, n_vocab))[0]

    @staticmethod
    def sample_greedy(ctx: ptr[llama_context]) -> int:
        """### INTERNAL

        Sample the most likely token."""
        return llama_sampler_sample(_internals.greedy_sampler, ctx, -1)

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

        Convert token ID to text bytes"""
        n_bytes = llama_token_to_piece(
            vocab=vocab,
            token=token,
            buf=_internals.detok_buffer,
            length=MAX_TOKEN_LENGTH,
            lstrip=0, # skip up to 'lstrip' leading spaces
            special=special
        )
        if n_bytes > MAX_TOKEN_LENGTH:
            raise ValueError(
                f"token_to_piece: the token with ID {token} requires a "
                f"buffer of size {n_bytes}, but the maximum buffer size is "
                f"{MAX_TOKEN_LENGTH}"
            )
        # NOTE: do not just do buf.value.decode() because the token could
        #       possibly be a part of a utf-8 bytestring, but not a valid utf-8
        #       string itself. let the caller handle this
        return _internals.detok_buffer.raw[:n_bytes]

    @staticmethod
    def detokenize(
        vocab: ptr[llama_vocab],
        tokens: list[int],
        special: bool
    ) -> str:
        """### INTERNAL

        Convert the provided tokens into a string

        - special:
            If True, special tokens are rendered in the output"""
        # this function is just like token_to_piece but in a loop
        detok_bytes = b""
        for token in tokens:
            n_bytes = llama_token_to_piece(
                vocab=vocab,
                token=token,
                buf=_internals.detok_buffer,
                length=MAX_TOKEN_LENGTH,
                lstrip=0, # skip up to 'lstrip' leading spaces
                special=special
            )
            if n_bytes > MAX_TOKEN_LENGTH:
                raise ValueError(
                    f"detokenize: the token with ID {token} requires a buffer of size "
                    f"{n_bytes}, but the maximum buffer size is {MAX_TOKEN_LENGTH}"
                )
            detok_bytes += _internals.detok_buffer.raw[:n_bytes]
        return ez_decode(detok_bytes)

    @staticmethod
    def get_length(
        vocab: ptr[llama_vocab],
        text_bytes: bytes,
        add_special: bool,
        parse_special: bool,
    ) -> int:
        """### INTERNAL

        Return the length of a given text, as measured in tokens"""
        return -llama_tokenize(
            vocab=vocab,
            text=text_bytes,
            text_len=len(text_bytes),
            tokens=NULL,
            n_tokens_max=0,
            add_special=add_special,
            parse_special=parse_special
        )
    
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

#
# Deprecation
#

class LlamaDeprecatedException(Exception):
    """Exception raised when calling functions marked with DEPRECATED in libllama"""

def DEPRECATED(new_func: Optional[Callable] = None):
    """Decorator for functions that are marked with DEPRECATED in libllama"""
    
    def decorator(func: Callable):
        def deprecator(*args, **kwargs):
            if new_func is None:
                raise LlamaDeprecatedException(
                    f"The function {func.__name__!r} is marked as deprecated. You cannot "
                    f"use it."
                )
            else:
                raise LlamaDeprecatedException(
                    f"The function {func.__name__!r} is marked as deprecated. You cannot "
                    f"use it. Use {new_func.__name__!r} instead."
                )

        return deprecator

    return decorator

@DEPRECATED(new_func=llama_model_load_from_file)
def llama_load_model_from_file(*args):
    pass

@DEPRECATED(new_func=llama_model_free)
def llama_free_model(*args):
    pass

@DEPRECATED(new_func=llama_init_from_model)
def llama_new_context_with_model(*args):
    pass

@DEPRECATED(new_func=llama_model_n_ctx_train)
def llama_n_ctx_train(*args):
    pass

@DEPRECATED(new_func=llama_model_n_embd)
def llama_n_embd(*args):
    pass

@DEPRECATED(new_func=llama_model_n_layer)
def llama_n_layer(*args):
    pass

@DEPRECATED(new_func=llama_model_n_head)
def llama_n_head(*args):
    pass

@DEPRECATED(new_func=llama_vocab_n_tokens)
def llama_n_vocab(*args):
    pass

@DEPRECATED(new_func=llama_model_rope_freq_scale_train)
def llama_rope_freq_scale_train(*args):
    pass

@DEPRECATED(new_func=llama_adapter_lora_init)
def llama_lora_adapter_init(*args):
    pass

@DEPRECATED(new_func=llama_adapter_lora_set)
def llama_lora_adapter_set(*args):
    pass

@DEPRECATED(new_func=llama_rm_adapter_lora)
def llama_lora_adapter_remove(*args):
    pass

@DEPRECATED(new_func=None)
def llama_lora_adapter_clear(*args):
    pass

@DEPRECATED(new_func=llama_adapter_lora_free)
def llama_lora_adapter_free(*args):
    pass

@DEPRECATED(new_func=llama_apply_adapter_cvec)
def llama_control_vector_apply(*args):
    pass

@DEPRECATED(new_func=llama_state_get_size)
def llama_get_state_size(*args):
    pass

@DEPRECATED(new_func=llama_state_get_data)
def llama_copy_state_data(*args):
    pass

@DEPRECATED(new_func=llama_state_set_data)
def llama_set_state_data(*args):
    pass

@DEPRECATED(new_func=llama_state_load_file)
def llama_load_session_file(*args):
    pass

@DEPRECATED(new_func=llama_state_save_file)
def llama_save_session_file(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_pre)
def llama_token_prefix(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_mid)
def llama_token_middle(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_suf)
def llama_token_suffix(*args):
    pass

@DEPRECATED(new_func=llama_vocab_get_text)
def llama_token_get_text(*args):
    pass

@DEPRECATED(new_func=llama_vocab_get_score)
def llama_token_get_score(*args):
    pass

@DEPRECATED(new_func=llama_vocab_get_attr)
def llama_token_get_attr(*args):
    pass

@DEPRECATED(new_func=llama_vocab_is_eog)
def llama_token_is_eog(*args):
    pass

@DEPRECATED(new_func=llama_vocab_is_control)
def llama_token_is_control(*args):
    pass

@DEPRECATED(new_func=llama_vocab_bos)
def llama_token_bos(*args):
    pass

@DEPRECATED(new_func=llama_vocab_eos)
def llama_token_eos(*args):
    pass

@DEPRECATED(new_func=llama_vocab_eot)
def llama_token_eot(*args):
    pass

@DEPRECATED(new_func=llama_vocab_bos)
def llama_token_cls(*args):
    pass

@DEPRECATED(new_func=llama_vocab_sep)
def llama_token_sep(*args):
    pass

@DEPRECATED(new_func=llama_vocab_nl)
def llama_token_nl(*args):
    pass

@DEPRECATED(new_func=llama_vocab_pad)
def llama_token_pad(*args):
    pass

@DEPRECATED(new_func=llama_vocab_get_add_bos)
def llama_add_bos_token(*args):
    pass

@DEPRECATED(new_func=llama_vocab_get_add_eos)
def llama_add_eos_token(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_pre)
def llama_token_fim_pre(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_suf)
def llama_token_fim_suf(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_mid)
def llama_token_fim_mid(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_pad)
def llama_token_fim_pad(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_rep)
def llama_token_fim_rep(*args):
    pass

@DEPRECATED(new_func=llama_vocab_fim_sep)
def llama_token_fim_sep(*args):
    pass

@DEPRECATED(new_func=llama_sampler_init_dist)
def llama_sampler_init_softmax(*args):
    pass

@DEPRECATED(new_func=llama_sampler_init_grammar_lazy_patterns)
def llama_sampler_init_grammar_lazy(*args):
    pass

@DEPRECATED(new_func=None)
def llama_kv_self_n_tokens(*args):
    pass

@DEPRECATED(new_func=None)
def llama_kv_self_used_cells(*args):
    pass

@DEPRECATED(new_func=None)
def llama_kv_self_defrag(*args):
    pass

@DEPRECATED(new_func=None)
def llama_kv_self_update(*args):
    pass
