# libllama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""
This file provides a Python interface to LLAMA_API ("libllama"), which is
originally defined in `llama.cpp/include/llama.h`.

This file was last updated to match commit `f66f5829276650cd83a087ab2cfed1a760183ea1`.

---

Helpful references:

- `libllama` changelog:
    [llama.cpp/issues/9289](https://github.com/ggerganov/llama.cpp/issues/9289)
- `llama.h` at master:
    [llama.cpp/blob/master/include/llama.h](https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h)

"""

import sys
import ctypes
import faulthandler

from enum   import IntEnum
from typing import Optional, Iterable
from utils  import ptr, print_warning

faulthandler.enable() # prints more helpful info if python crashes

class LlamaDeprecatedException(Exception):
    """
    Exception raised when calling functions marked with DEPRECATED in libllama
    """

def DEPRECATED(new_func_name: Optional[str] = None):
    """
    Decorator for functions that are marked with DEPRECATED in libllama
    """

    def decorator(func):

        def deprecator(*args, **kwargs):
            if new_func_name is None:
                raise LlamaDeprecatedException(
                    f"the function {func.__name__} is marked as deprecated. you "
                    f"cannot use it."
                )
            else:
                raise LlamaDeprecatedException(
                    f"the function {func.__name__} is marked as deprecated. you "
                    f"cannot use it. use {new_func_name} instead."
                )
        
        return deprecator
    
    return decorator

#
# Import shared library
#

#libllama = ctypes.CDLL('/Users/dylan/Documents/AI/easy-llama/easy_llama/libllama.dylib')
libllama = ctypes.CDLL('/Users/dylan/Documents/AI/llama.cpp/build/src/libllama.dylib')

#
# Type hints and other constants
#

NULL = None
NULLPTR = ctypes.c_void_p(NULL)

# maximum value for int32, it is used as the value for n_gpu_layers
# when all layers should be offloaded
MAX_OFFLOAD_LAYERS = 0x7FFFFFFF

# keep state for backend
_BACKEND_INIT = False

#
# Stuff from llama.cpp/ggml/include/ggml.h
#

GGML_EXIT_SUCCESS = 0
GGML_EXIT_ABORTED = 1

GGML_ROPE_TYPE_NEOX = 2
GGML_ROPE_TYPE_MROPE = 8
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
    # GGML_TYPE_Q4_2  = 4 -- support has been removed
    # GGML_TYPE_Q4_3  = 5 -- support has been removed
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
    GGML_TYPE_TQ1_0   = 3,
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
# Begin LLAMA_API
#

class llama_model(ctypes.Structure):
    """Dummy `ctypes.Structure`"""

llama_model_p = ctypes.POINTER(llama_model)
"""Pointer to a llama_model struct"""

class llama_context(ctypes.Structure):
    """Dummy `ctypes.Structure`"""

llama_context_p = ctypes.POINTER(llama_context)
"""Pointer to a llama_context struct"""

class llama_sampler(ctypes.Structure):
    """Dummy `ctypes.Structure`"""

llama_sampler_p = ctypes.POINTER(llama_context)
"""Pointer to a llama_sampler struct"""

size_t = ctypes.c_ulong

llama_pos    = ctypes.c_int32
llama_token  = ctypes.c_int32
llama_seq_id = ctypes.c_int32

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
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3  # SPECIAL?
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9

class LlamaFType(IntEnum): # model file types
    LLAMA_FTYPE_ALL_F32              = 0
    LLAMA_FTYPE_MOSTLY_F16           = 1  # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_0          = 2  # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_1          = 3  # except 1d tensors
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
    LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35 # except 1d tensors
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
        ("id", ctypes.c_int32),  # token id
        ("logit", ctypes.c_float),  # log-odds of the token
        ("p", ctypes.c_float),  # probability of the token
    ]

llama_token_data_p = ctypes.POINTER(llama_token_data)

class llama_token_data_array(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(llama_token_data)),  # NOTE: this pointer can be modified by the samplers
        ("size", ctypes.c_size_t),
        ("selected", ctypes.c_int64),  # this is the index in the data array (i.e. not the token id)
        ("sorted", ctypes.c_bool),
    ]

llama_token_data_array_p = ctypes.POINTER(llama_token_data_array)

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),

        ("token", ctypes.POINTER(llama_token)),  # the token ids of the input (used when embd is NULL)
        ("embd", ctypes.POINTER(ctypes.c_float)),  # token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
        ("pos", ctypes.POINTER(llama_pos)),  # the positions of the respective token in the sequence
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),  # the sequence to which the respective token belongs
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),  # the sequence to which the respective token belongs
        ("logits", ctypes.POINTER(ctypes.c_int8)),  # if zero, the logits (and/or the embeddings) for the respective token will not be output
    ]

llama_batch_p = ctypes.POINTER(llama_batch)

class llama_model_kv_override_type(ctypes.Union):
    _fields_ = [
        ("val_i64", ctypes.c_int64),
        ("val_f64", ctypes.c_double),
        ("val_bool", ctypes.c_bool),
        ("val_str", ctypes.c_char * 128),
    ]

llama_model_kv_override_type_p = ctypes.POINTER(llama_model_kv_override_type)

class llama_model_kv_override(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_int),
        ("key", ctypes.c_char * 128),
        ("val", llama_model_kv_override_type),
    ]

llama_model_kv_override_p = ctypes.POINTER(llama_model_kv_override)

dummy_progress_callback = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p
)

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),  # NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ("n_gpu_layers", ctypes.c_int32),  # number of layers to store in VRAM
        ("split_mode", ctypes.c_int),  # how to split the model across multiple GPUs
        ("main_gpu", ctypes.c_int32),  # the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),  # proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        ("rpc_servers", ctypes.c_char_p),  # comma separated list of RPC servers to use for offloading
        ("progress_callback", dummy_progress_callback),  # Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        ("progress_callback_user_data", ctypes.c_void_p),  # context pointer passed to the progress callback
        ("kv_overrides", ctypes.POINTER(llama_model_kv_override)),  # override key-value pairs of the model meta data
        ("vocab_only", ctypes.c_bool),  # only load the vocabulary, no weights
        ("use_mmap", ctypes.c_bool),  # use mmap if possible
        ("use_mlock", ctypes.c_bool),  # force system to keep model in RAM
        ("check_tensors", ctypes.c_bool),  # validate model tensor data
    ]

llama_model_params_p = ctypes.POINTER(llama_model_params)

dummy_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
)

dummy_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),  # text context, 0 = from model
        ("n_batch", ctypes.c_uint32),  # logical maximum batch size that can be submitted to llama_decode
        ("n_ubatch", ctypes.c_uint32),  # physical maximum batch size
        ("n_seq_max", ctypes.c_uint32),  # max number of sequences (i.e. distinct states for recurrent models)
        ("n_threads", ctypes.c_int32),  # number of threads to use for generation
        ("n_threads_batch", ctypes.c_int32),  # number of threads to use for batch processing
        ("rope_scaling_type", ctypes.c_int),  # RoPE scaling type, from `enum llama_rope_scaling_type`
        ("pooling_type", ctypes.c_int),  # whether to pool (sum) embedding results by sequence id
        ("attention_type", ctypes.c_int),  # attention type to use for embeddings
        ("rope_freq_base", ctypes.c_float),  # RoPE base frequency, 0 = from model
        ("rope_freq_scale", ctypes.c_float),  # RoPE frequency scaling factor, 0 = from model
        ("yarn_ext_factor", ctypes.c_float),  # YaRN extrapolation mix factor, negative = from model
        ("yarn_attn_factor", ctypes.c_float),  # YaRN magnitude scaling factor
        ("yarn_beta_fast", ctypes.c_float),  # YaRN low correction dim
        ("yarn_beta_slow", ctypes.c_float),  # YaRN high correction dim
        ("yarn_orig_ctx", ctypes.c_uint32),  # YaRN original context size
        ("defrag_thold", ctypes.c_float),  # defragment the KV cache if holes/size > thold, < 0 disabled (default)
        ("cb_eval", dummy_eval_callback),  # callback for eval
        ("cb_eval_user_data", ctypes.c_void_p),  # user data for eval callback
        ("type_k", ctypes.c_int),  # data type for K cache [EXPERIMENTAL]
        ("type_v", ctypes.c_int),  # data type for V cache [EXPERIMENTAL]
        ("logits_all", ctypes.c_bool),  # the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
        ("embeddings", ctypes.c_bool),  # if true, extract embeddings (together with logits)
        ("offload_kqv", ctypes.c_bool),  # whether to offload the KQV ops (including the KV cache) to GPU
        ("flash_attn", ctypes.c_bool),  # whether to use flash attention [EXPERIMENTAL]
        ("no_perf", ctypes.c_bool),  # whether to measure performance timings
        ("abort_callback", dummy_abort_callback),  # callback for abort
        ("abort_callback_data", ctypes.c_void_p),  # user data for abort callback
    ]

llama_context_params_p = ctypes.POINTER(llama_context_params)

class llama_model_quantize_params(ctypes.Structure):
    _fields_ = [
        ("nthread", ctypes.c_int32),  # number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        ("ftype", ctypes.c_int),  # quantize to this llama_ftype
        ("output_tensor_type", ctypes.c_int),  # output tensor type
        ("token_embedding_type", ctypes.c_int),  # token embeddings tensor type
        ("allow_requantize", ctypes.c_bool),  # allow quantizing non-f32/f16 tensors
        ("quantize_output_tensor", ctypes.c_bool),  # quantize output.weight
        ("only_copy", ctypes.c_bool),  # only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        ("pure", ctypes.c_bool),  # quantize all tensors to the default type
        ("keep_split", ctypes.c_bool),  # quantize to the same number of shards
        ("imatrix", ctypes.c_void_p),  # pointer to importance matrix data
        ("kv_overrides", ctypes.c_void_p),  # pointer to vector containing overrides
    ]

llama_model_quantize_params_p = ctypes.POINTER(llama_model_quantize_params)

class llama_logit_bias(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_int32),
        ("bias", ctypes.c_float),
    ]

llama_logit_bias_p = ctypes.POINTER(llama_logit_bias)

class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),  # whether to measure performance timings
    ]

llama_sampler_chain_params_p = ctypes.POINTER(llama_sampler_chain_params)

class llama_chat_message(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p),
    ]

llama_chat_message_p = ctypes.POINTER(llama_chat_message)

# // TODO: rename to llama_adapter_lora
class llama_lora_adapter(ctypes.Structure):
    pass

llama_lora_adapter_p = ctypes.POINTER(llama_lora_adapter)

#
# Helpers for getting default parameters
#

libllama.llama_model_default_params.argtypes = []
libllama.llama_model_default_params.restype = llama_model_params

def llama_model_default_params() -> llama_model_params:
    """Get the default parameters for a llama model"""
    return libllama.llama_model_default_params()

libllama.llama_context_default_params.argtypes = []
libllama.llama_context_default_params.restype = llama_context_params

def llama_context_default_params() -> llama_context_params:
    """Get the default parameters for a llama context"""
    return libllama.llama_context_default_params()

libllama.llama_sampler_chain_default_params.argtypes = []
libllama.llama_sampler_chain_default_params.restype = llama_sampler_chain_params

def llama_sampler_chain_default_params() -> llama_sampler_chain_params:
    """Get the default parameters for a sampler chain"""
    return libllama.llama_sampler_chain_default_params()

libllama.llama_model_quantize_default_params.argtypes = []
libllama.llama_model_quantize_default_params.restype = llama_model_quantize_params

def llama_model_quantize_default_params() -> llama_model_quantize_params:
    """Get the default parameters for model quantization"""
    return libllama.llama_model_quantize_default_params()

#
# Setup and teardown
#

libllama.llama_backend_init.argtypes = []
libllama.llama_backend_init.restype = None

def llama_backend_init() -> None:
    """Initialize the llama + ggml backend"""
    global _BACKEND_INIT
    libllama.llama_backend_init()
    _BACKEND_INIT = True

libllama.llama_numa_init.argtypes = [ctypes.c_int]
libllama.llama_numa_init.restype = None

def llama_numa_init(numa: int) -> None:
    """Initialize NUMA optimizations globally"""
    libllama.llama_numa_init(numa)

libllama.llama_attach_threadpool.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_void_p]
libllama.llama_attach_threadpool.restype = None

def llama_attach_threadpool(ctx: llama_context, ggml_threadpool: ptr, threadpool_batch: ptr) -> None:
    """Attach a threadpool to a llama_context"""
    libllama.llama_attach_threadpool(ctx, ggml_threadpool, threadpool_batch)

libllama.llama_detach_threadpool.argtypes = [llama_context_p]
libllama.llama_detach_threadpool.restype = None

def llama_detach_threadpool(ctx: llama_context) -> None:
    """Detach a threadpool from a llama_context"""
    libllama.llama_detach_threadpool(ctx)

libllama.llama_backend_free.argtypes = []
libllama.llama_backend_free.restype = None

def llama_backend_free() -> None:
    """
    Free the llama + ggml backend
    
    Call once at the end of the program - currently only used for MPI
    """
    global _BACKEND_INIT
    libllama.llama_backend_free()
    _BACKEND_INIT = False

libllama.llama_load_model_from_file.argtypes = [ctypes.c_char_p, llama_model_params_p]
libllama.llama_load_model_from_file.restype = llama_model_p

def llama_load_model_from_file(path_model: str, params: llama_model_params) -> ptr[llama_model]:
    """Load a llama model from a file - returns a pointer"""
    return libllama.llama_load_model_from_file(path_model.encode('utf-8'), ctypes.byref(params))

libllama.llama_free_model.argtypes = [llama_model_p]
libllama.llama_free_model.restype = None

# // TODO: rename to llama_model_free
def llama_free_model(model: llama_model) -> None:
    """Free a model"""
    libllama.llama_free_model(model)

libllama.llama_new_context_with_model.argtypes = [llama_model_p, llama_context_params_p]
libllama.llama_new_context_with_model.restype = llama_context_p

def llama_new_context_with_model(model: llama_model, params: llama_context_params) -> ptr[llama_context]:
    """Create a new llama context with a loaded model"""
    return libllama.llama_new_context_with_model(model, ctypes.byref(params))

libllama.llama_free.argtypes = [llama_context_p]
libllama.llama_free.restype = None

def llama_free(ctx: llama_context) -> None:
    """Frees all allocated memory"""
    libllama.llama_free(ctx)

#
# Llama backend helper functions
#

libllama.llama_time_us.argtypes = []
libllama.llama_time_us.restype = ctypes.c_int

def llama_time_us() -> int:
    """Get the current time in microseconds"""
    return libllama.llama_time_us()

libllama.llama_max_devices.argtypes = []
libllama.llama_max_devices.restype = ctypes.c_int

def llama_max_devices() -> int:
    """Get the maximum number of devices"""
    return libllama.llama_max_devices()

libllama.llama_supports_mmap.argtypes = []
libllama.llama_supports_mmap.restype = ctypes.c_bool

def llama_supports_mmap() -> bool:
    """Check if mmap is supported"""
    return libllama.llama_supports_mmap()

libllama.llama_supports_mlock.argtypes = []
libllama.llama_supports_mlock.restype = ctypes.c_bool

def llama_supports_mlock() -> bool:
    """Check if mlock is supported"""
    return libllama.llama_supports_mlock()

libllama.llama_supports_gpu_offload.argtypes = []
libllama.llama_supports_gpu_offload.restype = ctypes.c_bool

def llama_supports_gpu_offload() -> bool:
    """Check if GPU offload is supported"""
    return libllama.llama_supports_gpu_offload()

libllama.llama_supports_rpc.argtypes = []
libllama.llama_supports_rpc.restype = ctypes.c_bool

def llama_supports_rpc() -> bool:
    """Check if RPC is supported"""
    return libllama.llama_supports_rpc()

libllama.llama_n_ctx.argtypes = [llama_context_p]
libllama.llama_n_ctx.restype = ctypes.c_int

def llama_n_ctx(ctx: llama_context) -> int:
    """Get the context size"""
    return libllama.llama_n_ctx(ctx)

libllama.llama_n_batch.argtypes = [llama_context_p]
libllama.llama_n_batch.restype = ctypes.c_int

def llama_n_batch(ctx: llama_context) -> int:
    """Get the logical maximum batch size"""
    return libllama.llama_n_batch(ctx)

libllama.llama_n_ubatch.argtypes = [llama_context_p]
libllama.llama_n_ubatch.restype = ctypes.c_int

def llama_n_ubatch(ctx: llama_context) -> int:
    """Get the physical maximum batch size"""
    return libllama.llama_n_ubatch(ctx)

libllama.llama_n_seq_max.argtypes = [llama_context_p]
libllama.llama_n_seq_max.restype = ctypes.c_int

def llama_n_seq_max(ctx: llama_context) -> int:
    """Get the maximum number of sequences"""
    return libllama.llama_n_seq_max(ctx)

#
# Getters for model attributes
#

libllama.llama_n_vocab.argtypes = [llama_model_p]
libllama.llama_n_vocab.restype = ctypes.c_int

def llama_n_vocab(model: llama_model) -> int:
    """Get the number of tokens in the vocabulary"""
    return libllama.llama_n_vocab(model)

libllama.llama_n_ctx_train.argtypes = [llama_model_p]
libllama.llama_n_ctx_train.restype = ctypes.c_int

def llama_n_ctx_train(model: llama_model) -> int:
    """Get the context size used during training"""
    return libllama.llama_n_ctx_train(model)

libllama.llama_n_embd.argtypes = [llama_model_p]
libllama.llama_n_embd.restype = ctypes.c_int

def llama_n_embd(model: llama_model) -> int:
    """Get the embedding size"""
    return libllama.llama_n_embd(model)

libllama.llama_n_layer.argtypes = [llama_model_p]
libllama.llama_n_layer.restype = ctypes.c_int

def llama_n_layer(model: llama_model) -> int:
    """Get the number of layers"""
    return libllama.llama_n_layer(model)

libllama.llama_n_head.argtypes = [llama_model_p]
libllama.llama_n_head.restype = ctypes.c_int

def llama_n_head(model: llama_model) -> int:
    """Get the number of attention heads"""
    return libllama.llama_n_head(model)

# More getters for llama_context ...

libllama.llama_get_model.argtypes = [llama_context_p]
libllama.llama_get_model.restype = llama_model_p

def llama_get_model(ctx: llama_context) -> ptr[llama_model]:
    """Get the model associated with a context"""
    return libllama.llama_get_model(ctx)

libllama.llama_pooling_type.argtypes = [llama_context_p]
libllama.llama_pooling_type.restype = ctypes.c_int

def llama_pooling_type(ctx: llama_context) -> int:
    """Get the pooling type used by a context"""
    return libllama.llama_pooling_type(ctx)

# More getters for llama_model ...

libllama.llama_vocab_type.argtypes = [llama_model_p]
libllama.llama_vocab_type.restype = ctypes.c_int

def llama_vocab_type(model: llama_model) -> int:
    """Get the vocabulary type used by a model"""
    return libllama.llama_vocab_type(model)

libllama.llama_rope_type.argtypes = [llama_model_p]
libllama.llama_rope_type.restype = ctypes.c_int

def llama_rope_type(model: llama_model) -> int:
    """Get the RoPE type used by a model"""
    return libllama.llama_rope_type(model)

libllama.llama_rope_freq_scale_train.argtypes = [llama_model_p]
libllama.llama_rope_freq_scale_train.restype = ctypes.c_float

def llama_rope_freq_scale_train(model: llama_model) -> float:
    """Get the RoPE frequency scaling factor used during training"""
    return libllama.llama_rope_freq_scale_train(model)

libllama.llama_model_meta_val_str.argtypes = [llama_model_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
libllama.llama_model_meta_val_str.restype = ctypes.c_int

def llama_model_meta_val_str(model: llama_model, key: str, buf: ctypes.c_char_p, buf_size: int) -> int:
    """Get a metadata value as a string"""
    return libllama.llama_model_meta_val_str(model, key.encode('utf-8'), buf, buf_size)

libllama.llama_model_meta_count.argtypes = [llama_model_p]
libllama.llama_model_meta_count.restype = ctypes.c_int

def llama_model_meta_count(model: llama_model) -> int:
    """Get the number of metadata key-value pairs"""
    return libllama.llama_model_meta_count(model)

libllama.llama_model_meta_key_by_index.argtypes = [llama_model_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
libllama.llama_model_meta_key_by_index.restype = ctypes.c_int

def llama_model_meta_key_by_index(model: llama_model, i: int, buf: ctypes.c_char_p, buf_size: int) -> int:
    """Get a metadata key by index"""
    return libllama.llama_model_meta_key_by_index(model, i, buf, buf_size)

libllama.llama_model_meta_val_str_by_index.argtypes = [llama_model_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
libllama.llama_model_meta_val_str_by_index.restype = ctypes.c_int

def llama_model_meta_val_str_by_index(model: llama_model, i: int, buf: ctypes.c_char_p, buf_size: int) -> int:
    """Get a metadata value by index"""
    return libllama.llama_model_meta_val_str_by_index(model, i, buf, buf_size)

libllama.llama_model_desc.argtypes = [llama_model_p, ctypes.c_char_p, ctypes.c_int]
libllama.llama_model_desc.restype = ctypes.c_int

def llama_model_desc(model: llama_model, buf: ctypes.c_char_p, buf_size: int) -> int:
    """Get a string describing the model type"""
    return libllama.llama_model_desc(model, buf, buf_size)

libllama.llama_model_size.argtypes = [llama_model_p]
libllama.llama_model_size.restype = size_t

def llama_model_size(model: llama_model) -> int:
    """Get the total size of all tensors in the model in bytes"""
    return libllama.llama_model_size(model)

libllama.llama_model_n_params.argtypes = [llama_model_p]
libllama.llama_model_n_params.restype = size_t

def llama_model_n_params(model: llama_model) -> int:
    """Get the total number of parameters in the model"""
    return libllama.llama_model_n_params(model)

libllama.llama_model_has_encoder.argtypes = [llama_model_p]
libllama.llama_model_has_encoder.restype = ctypes.c_bool

def llama_model_has_encoder(model: llama_model) -> bool:
    """Check if the model has an encoder"""
    return libllama.llama_model_has_encoder(model)

libllama.llama_model_has_decoder.argtypes = [llama_model_p]
libllama.llama_model_has_decoder.restype = ctypes.c_bool

def llama_model_has_decoder(model: llama_model) -> bool:
    """Check if the model has a decoder"""
    return libllama.llama_model_has_decoder(model)

libllama.llama_model_decoder_start_token.argtypes = [llama_model_p]
libllama.llama_model_decoder_start_token.restype = ctypes.c_int

def llama_model_decoder_start_token(model: llama_model) -> int:
    """Get the start token for the decoder"""
    return libllama.llama_model_decoder_start_token(model)

libllama.llama_model_is_recurrent.argtypes = [llama_model_p]
libllama.llama_model_is_recurrent.restype = ctypes.c_bool

def llama_model_is_recurrent(model: llama_model) -> bool:
    """Check if the model is recurrent"""
    return libllama.llama_model_is_recurrent(model)

#
# Quantization
#

libllama.llama_model_quantize.argtypes = [ctypes.c_char_p, ctypes.c_char_p, llama_model_quantize_params_p]
libllama.llama_model_quantize.restype = ctypes.c_int

def llama_model_quantize(fname_inp: str, fname_out: str, params: llama_model_quantize_params) -> int:
    """Quantize a model. Returns 0 on success"""
    return libllama.llama_model_quantize(fname_inp.encode('utf-8'), fname_out.encode('utf-8'), ctypes.byref(params))

#
# LoRA
#

libllama.llama_lora_adapter_init.argtypes = [llama_model_p, ctypes.c_char_p]
libllama.llama_lora_adapter_init.restype = llama_lora_adapter_p

# // TODO: rename to llama_adapter_lora_init
def llama_lora_adapter_init(model: llama_model, path_lora: str) -> ptr[llama_lora_adapter]:
    """Initialize a LoRA adapter"""
    return libllama.llama_lora_adapter_init(model, path_lora.encode('utf-8'))

libllama.llama_lora_adapter_set.argtypes = [llama_context_p, llama_lora_adapter_p, ctypes.c_float]
libllama.llama_lora_adapter_set.restype = ctypes.c_int

# // TODO: rename to llama_set_adapter_lora
def llama_lora_adapter_set(ctx: llama_context, adapter: llama_lora_adapter, scale: float) -> int:
    """Set a LoRA adapter for a context"""
    return libllama.llama_lora_adapter_set(ctx, adapter, scale)

libllama.llama_lora_adapter_remove.argtypes = [llama_context_p, llama_lora_adapter_p]
libllama.llama_lora_adapter_remove.restype = ctypes.c_int

# // TODO: rename to llama_rm_adapter_lora
def llama_lora_adapter_remove(ctx: llama_context, adapter: llama_lora_adapter) -> int:
    """Remove a LoRA adapter from a context"""
    return libllama.llama_lora_adapter_remove(ctx, adapter)

libllama.llama_lora_adapter_clear.argtypes = [llama_context_p]
libllama.llama_lora_adapter_clear.restype = None

# // TODO: rename to llama_clear_adapter_lora
def llama_lora_adapter_clear(ctx: llama_context) -> None:
    """Clear all LoRA adapters from a context"""
    libllama.llama_lora_adapter_clear(ctx)

libllama.llama_lora_adapter_free.argtypes = [llama_lora_adapter_p]
libllama.llama_lora_adapter_free.restype = None

# // TODO: rename to llama_adapter_lora_free
def llama_lora_adapter_free(adapter: llama_lora_adapter) -> None:
    """Free a LoRA adapter"""
    libllama.llama_lora_adapter_free(adapter)

#
# Control vector
#

libllama.llama_control_vector_apply.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libllama.llama_control_vector_apply.restype = ctypes.c_int

# // TODO: rename to llama_adapter_cvec_apply
def llama_control_vector_apply(ctx: llama_context, data: ctypes.c_void_p, len: int, n_embd: int, il_start: int, il_end: int) -> int:
    """Apply a control vector to a context"""
    return libllama.llama_control_vector_apply(ctx, data, len, n_embd, il_start, il_end)

#
# KV cache
#

# // TODO: remove llama_kv_cache_view_* API

class llama_kv_cache_view_cell(ctypes.Structure):
    _fields_ = [
        ("pos", ctypes.c_int32),  # The position for this cell. Takes KV cache shifts into account.
    ]

llama_kv_cache_view_cell_p = ctypes.POINTER(llama_kv_cache_view_cell)

class llama_kv_cache_view(ctypes.Structure):
    _fields_ = [
        ("n_cells", ctypes.c_int32),  # Number of KV cache cells. This will be the same as the context size.
        ("n_seq_max", ctypes.c_int32),  # Maximum number of sequences that can exist in a cell. It's not an error
        ("token_count", ctypes.c_int32),  # Number of tokens in the cache. For example, if there are two populated
        ("used_cells", ctypes.c_int32),  # Number of populated cache cells.
        ("max_contiguous", ctypes.c_int32),  # Maximum contiguous empty slots in the cache.
        ("max_contiguous_idx", ctypes.c_int32),  # Index to the start of the max_contiguous slot range. Can be negative
        ("cells", ctypes.POINTER(llama_kv_cache_view_cell)),  # Information for an individual cell.
        ("cells_sequences", ctypes.POINTER(ctypes.c_int32)),  # The sequences for each cell. There will be n_seq_max items per cell.
    ]

llama_kv_cache_view_p = ctypes.POINTER(llama_kv_cache_view)

libllama.llama_kv_cache_view_init.argtypes = [llama_context_p, ctypes.c_int]
libllama.llama_kv_cache_view_init.restype = llama_kv_cache_view_p

def llama_kv_cache_view_init(ctx: llama_context, n_seq_max: int) -> ptr[llama_kv_cache_view]:
    """
    DEBUG ONLY

    Create an empty KV cache view (use only for debugging purposes)
    """
    return libllama.llama_kv_cache_view_init(ctx, n_seq_max)

libllama.llama_kv_cache_view_free.argtypes = [llama_kv_cache_view_p]
libllama.llama_kv_cache_view_free.restype = None

def llama_kv_cache_view_free(view: llama_kv_cache_view) -> None:
    """
    DEBUG ONLY

    Free a KV cache view
    """
    libllama.llama_kv_cache_view_free(view)

libllama.llama_kv_cache_view_update.argtypes = [llama_context_p, llama_kv_cache_view_p]
libllama.llama_kv_cache_view_update.restype = None

# // TODO: change signature to llama_kv_cache_view_update(struct llama_kv_cache_view * view, const struct llama_context * ctx)
def llama_kv_cache_view_update(ctx: llama_context, view: llama_kv_cache_view) -> None:
    """
    DEBUG ONLY
    
    Update a KV cache view with the current state of the KV cache
    """
    libllama.llama_kv_cache_view_update(ctx, view)

libllama.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
libllama.llama_get_kv_cache_token_count.restype = ctypes.c_int

def llama_get_kv_cache_token_count(ctx: llama_context) -> int:
    """
    DEBUG ONLY & SLOW
    
    Get the number of tokens in the KV cache
    """
    return libllama.llama_get_kv_cache_token_count(ctx)

libllama.llama_get_kv_cache_used_cells.argtypes = [llama_context_p]
libllama.llama_get_kv_cache_used_cells.restype = ctypes.c_int

def llama_get_kv_cache_used_cells(ctx: llama_context) -> int:
    """Get the number of used KV cells"""
    return libllama.llama_get_kv_cache_used_cells(ctx)

libllama.llama_kv_cache_clear.argtypes = [llama_context_p]
libllama.llama_kv_cache_clear.restype = None

def llama_kv_cache_clear(ctx: llama_context) -> None:
    """Clear the KV cache"""
    libllama.llama_kv_cache_clear(ctx)

libllama.llama_kv_cache_seq_rm.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libllama.llama_kv_cache_seq_rm.restype = ctypes.c_bool

def llama_kv_cache_seq_rm(ctx: llama_context, seq_id: int, p0: int, p1: int) -> bool:
    """Remove tokens from a sequence in the KV cache"""
    return libllama.llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)

libllama.llama_kv_cache_seq_cp.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libllama.llama_kv_cache_seq_cp.restype = None

def llama_kv_cache_seq_cp(ctx: llama_context, seq_id_src: int, seq_id_dst: int, p0: int, p1: int) -> None:
    """Copy tokens from one sequence to another in the KV cache"""
    libllama.llama_kv_cache_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1)

libllama.llama_kv_cache_seq_keep.argtypes = [llama_context_p, ctypes.c_int]
libllama.llama_kv_cache_seq_keep.restype = None

def llama_kv_cache_seq_keep(ctx: llama_context, seq_id: int) -> None:
    """Keep only the tokens of a sequence in the KV cache"""
    libllama.llama_kv_cache_seq_keep(ctx, seq_id)

libllama.llama_kv_cache_seq_add.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libllama.llama_kv_cache_seq_add.restype = None

def llama_kv_cache_seq_add(ctx: llama_context, seq_id: int, p0: int, p1: int, delta: int) -> None:
    """Add a relative position to tokens in a sequence in the KV cache"""
    libllama.llama_kv_cache_seq_add(ctx, seq_id, p0, p1, delta)

libllama.llama_kv_cache_seq_div.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libllama.llama_kv_cache_seq_div.restype = None

def llama_kv_cache_seq_div(ctx: llama_context, seq_id: int, p0: int, p1: int, d: int) -> None:
    """Divide the positions of tokens in a sequence in the KV cache by a factor"""
    libllama.llama_kv_cache_seq_div(ctx, seq_id, p0, p1, d)

libllama.llama_kv_cache_seq_pos_max.argtypes = [llama_context_p, ctypes.c_int]
libllama.llama_kv_cache_seq_pos_max.restype = ctypes.c_int

def llama_kv_cache_seq_pos_max(ctx: llama_context, seq_id: int) -> int:
    """Get the maximum position of a sequence in the KV cache"""
    return libllama.llama_kv_cache_seq_pos_max(ctx, seq_id)

libllama.llama_kv_cache_defrag.argtypes = [llama_context_p]
libllama.llama_kv_cache_defrag.restype = None

def llama_kv_cache_defrag(ctx: llama_context) -> None:
    """Defragment the KV cache"""
    libllama.llama_kv_cache_defrag(ctx)

libllama.llama_kv_cache_update.argtypes = [llama_context_p]
libllama.llama_kv_cache_update.restype = None

def llama_kv_cache_update(ctx: llama_context) -> None:
    """Apply KV cache updates"""
    libllama.llama_kv_cache_update(ctx)

libllama.llama_kv_cache_can_shift.argtypes = [llama_context_p]
libllama.llama_kv_cache_can_shift.restype = ctypes.c_bool

def llama_kv_cache_can_shift(ctx: llama_context) -> bool:
    """Check if the context supports KV cache shifting"""
    return libllama.llama_kv_cache_can_shift(ctx)

#
# State management
#

libllama.llama_state_get_size.argtypes = [llama_context_p]
libllama.llama_state_get_size.restype = ctypes.c_int

def llama_state_get_size(ctx: llama_context) -> int:
    """Get the size of the state in bytes"""
    return libllama.llama_state_get_size(ctx)

@DEPRECATED(new_func_name="llama_state_get_size")
def llama_get_state_size(*args):
    pass

libllama.llama_state_get_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_int]
libllama.llama_state_get_data.restype = ctypes.c_int

def llama_state_get_data(ctx: llama_context, dst: ctypes.c_void_p, size: int) -> int:
    """Copy the state to a destination address"""
    return libllama.llama_state_get_data(ctx, dst, size)

@DEPRECATED(new_func_name="llama_state_get_data")
def llama_copy_state_data(*args):
    pass

libllama.llama_state_set_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_int]
libllama.llama_state_set_data.restype = ctypes.c_int

def llama_state_set_data(ctx: llama_context, src: ctypes.c_void_p, size: int) -> int:
    """Set the state from a source address"""
    return libllama.llama_state_set_data(ctx, src, size)

@DEPRECATED(new_func_name="llama_state_set_data")
def llama_set_state_data(*args):
    pass

libllama.llama_state_load_file.argtypes = [llama_context_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
libllama.llama_state_load_file.restype = ctypes.c_bool

def llama_state_load_file(ctx: llama_context, path_session: str, tokens_out: ptr[ctypes.c_int], n_token_capacity: int, n_token_count_out: ptr[ctypes.c_int]) -> bool:
    """Load a state from a file"""
    return libllama.llama_state_load_file(ctx, path_session.encode('utf-8'), tokens_out, n_token_capacity, n_token_count_out)

@DEPRECATED(new_func_name="llama_state_load_file")
def llama_load_session_file(*args):
    pass

libllama.llama_state_save_file.argtypes = [llama_context_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
libllama.llama_state_save_file.restype = ctypes.c_bool

def llama_state_save_file(ctx: llama_context, path_session: str, tokens: ptr[ctypes.c_int], n_token_count: int) -> bool:
    """Save a state to a file"""
    return libllama.llama_state_save_file(ctx, path_session.encode('utf-8'), tokens, n_token_count)

@DEPRECATED(new_func_name="llama_state_save_file")
def llama_save_session_file(*args):
    pass

libllama.llama_state_seq_get_size.argtypes = [llama_context_p, ctypes.c_int32]
libllama.llama_state_seq_get_size.restype = ctypes.c_ulong

def llama_state_seq_get_size(ctx: llama_context, llama_seq_id: int) -> int:
    """Get the exact size needed to copy the KV cache of a single sequence"""
    return libllama.llama_state_seq_get_size(ctx, llama_seq_id)

libllama.llama_state_seq_get_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int32]
libllama.llama_state_seq_get_data.restype = ctypes.c_ulong

def llama_state_seq_get_data(ctx: llama_context, dst: ctypes.c_void_p, size: int, seq_id: int) -> int:
    """Copy the KV cache of a single sequence into the specified buffer"""
    return libllama.llama_state_seq_get_data(ctx, dst, size, seq_id)

libllama.llama_state_seq_set_data.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int32]
libllama.llama_state_seq_set_data.restype = ctypes.c_ulong

def llama_state_seq_set_data(ctx: llama_context, src: ctypes.c_void_p, size: int, dest_seq_id: int) -> int:
    """
    Copy the sequence data (originally copied with `llama_state_seq_get_data`)
    into the specified sequence
    
    Returns:
    - Positive: Ok
    - Zero: Failed to load
    """
    return libllama.llama_state_seq_set_data(ctx, src, size, dest_seq_id)

libllama.llama_state_seq_save_file.argtypes = [llama_context_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int), ctypes.c_ulong]
libllama.llama_state_seq_save_file.restype = ctypes.c_ulong

def llama_state_seq_save_file(ctx: llama_context, filepath: str, seq_id: int, tokens: ptr[ctypes.c_int32], n_token_count: int) -> int:
    return libllama.llama_state_seq_save_file(ctx, filepath.encode('utf-8'), seq_id, tokens, n_token_count)

libllama.llama_state_seq_load_file.argtypes = [llama_context_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int), ctypes.c_int32, ctypes.POINTER(ctypes.c_int)]
libllama.llama_state_seq_load_file.restype = ctypes.c_ulong

def llama_state_seq_load_file(ctx: llama_context, filepath: str, dest_seq_id: int, tokens_out: ptr[ctypes.c_int32], n_token_capacity: int, n_token_count_out: ptr[ctypes.c_int32]) -> int:
    return libllama.llama_state_seq_load_file(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)

#
# Batch
#

libllama.llama_batch_get_one.argtypes = [ctypes.POINTER(llama_token), ctypes.c_int32]
libllama.llama_batch_get_one.restype = llama_batch

def llama_batch_get_one(tokens: ptr[llama_token], n_tokens: int) -> llama_batch:
    """
    AVOID USING

    This function will be deprecated and removed at some point. Refer to:
    https://github.com/ggerganov/llama.cpp/issues/6475#issuecomment-2040350410

    Return batch for single sequence of tokens
    """
    print_warning(
        f'you are using libllama.llama_batch_get_one which will be deprecated '
        f'and removed at some point. you should use libllama.llama_batch_init '
        f'instead'
    )
    return libllama.llama_batch_get_one(tokens, n_tokens)

libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_batch_init.restype = llama_batch

def llama_batch_init(n_tokens: int, embd: int, n_seq_max: int) -> llama_batch:
    """Allocate a batch of tokens"""
    return libllama.llama_batch_init(n_tokens, embd, n_seq_max)

libllama.llama_batch_free.argtypes = [llama_batch]
libllama.llama_batch_free.restype = None

def llama_batch_free(batch: llama_batch) -> None:
    """Frees a batch of tokens"""
    libllama.llama_batch_free(batch)

#
# Encode / decode
#

libllama.llama_encode.argtypes = [llama_context_p, llama_batch_p]
libllama.llama_encode.restype = ctypes.c_int

def llama_encode(ctx: llama_context, batch: llama_batch) -> int:
    """Process a batch of tokens with the encoder part of the encoder-decoder model"""
    return libllama.llama_encode(ctx, batch)

libllama.llama_decode.argtypes = [llama_context_p, llama_batch]
libllama.llama_decode.restype = ctypes.c_int32

def llama_decode(ctx: llama_context, batch: llama_batch) -> int:
    """
    Process a batch of tokens with the decoder part of the encoder-decoder model

    Returns:
    - 0:
        success
    - 1:
        could not find a KV slot for the batch (try reducing the size of
        the batch or increase the context)
    - < 0:
        error. the KV cache state is restored to the state before this
        call
    """
    return libllama.llama_decode(ctx, batch)

libllama.llama_set_n_threads.argtypes = [llama_context_p, ctypes.c_int, ctypes.c_int]
libllama.llama_set_n_threads.restype = None

def llama_set_n_threads(ctx: llama_context, n_threads: int, n_threads_batch: int) -> None:
    """Set the number of threads used for decoding"""
    libllama.llama_set_n_threads(ctx, n_threads, n_threads_batch)

libllama.llama_n_threads.argtypes = [llama_context_p]
libllama.llama_n_threads.restype = ctypes.c_int

def llama_n_threads(ctx: llama_context) -> int:
    """Get the number of threads used for generation of a single token"""
    return libllama.llama_n_threads(ctx)

libllama.llama_n_threads_batch.argtypes = [llama_context_p]
libllama.llama_n_threads_batch.restype = ctypes.c_int

def llama_n_threads_batch(ctx: llama_context) -> int:
    """Get the number of threads used for prompt and batch processing"""
    return libllama.llama_n_threads_batch(ctx)

libllama.llama_set_embeddings.argtypes = [llama_context_p, ctypes.c_bool]
libllama.llama_set_embeddings.restype = None

def llama_set_embeddings(ctx: llama_context, embeddings: bool) -> None:
    """Set whether to use embeddings mode or not"""
    libllama.llama_set_embeddings(ctx, embeddings)

libllama.llama_set_causal_attn.argtypes = [llama_context_p, ctypes.c_bool]
libllama.llama_set_causal_attn.restype = None

def llama_set_causal_attn(ctx: llama_context, causal_attn: bool) -> None:
    """Set whether to use causal attention or not"""
    libllama.llama_set_causal_attn(ctx, causal_attn)

libllama.llama_set_abort_callback.argtypes = [llama_context_p, ctypes.c_void_p, ctypes.c_void_p]
libllama.llama_set_abort_callback.restype = None

def llama_set_abort_callback(ctx: llama_context, abort_callback: ctypes.c_void_p, abort_callback_data: ctypes.c_void_p) -> None:
    """Set an abort callback"""
    libllama.llama_set_abort_callback(ctx, abort_callback, abort_callback_data)

libllama.llama_synchronize.argtypes = [llama_context_p]
libllama.llama_synchronize.restype = None

def llama_synchronize(ctx: llama_context) -> None:
    """
    Wait until all computations are finished

    Not necessary to call explicitly in most cases
    """
    libllama.llama_synchronize(ctx)

libllama.llama_get_logits.argtypes = [llama_context_p]
libllama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

def llama_get_logits(ctx: llama_context) -> ptr[ctypes.c_float]:
    """
    Get the token logits obtained from the last call to llama_decode()
    
    Rows: number of tokens for which llama_batch.logits[i] != 0
    Cols: n_vocab
    """
    return libllama.llama_get_logits(ctx)

libllama.llama_get_logits_ith.argtypes = [llama_context_p, ctypes.c_int]
libllama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

def llama_get_logits_ith(ctx: llama_context, i: int) -> ptr[ctypes.c_float]:
    """Get the logits for the ith token"""
    return libllama.llama_get_logits_ith(ctx, i)

libllama.llama_get_embeddings.argtypes = [llama_context_p]
libllama.llama_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)

def llama_get_embeddings(ctx: llama_context) -> ptr[ctypes.c_float]:
    """Get all output token embeddings"""
    return libllama.llama_get_embeddings(ctx)

libllama.llama_get_embeddings_ith.argtypes = [llama_context_p, ctypes.c_int]
libllama.llama_get_embeddings_ith.restype = ctypes.POINTER(ctypes.c_float)

def llama_get_embeddings_ith(ctx: llama_context, i: int) -> ptr[ctypes.c_float]:
    """Get the embeddings for the ith token"""
    return libllama.llama_get_embeddings_ith(ctx, i)

libllama.llama_get_embeddings_seq.argtypes = [llama_context_p, ctypes.c_int]
libllama.llama_get_embeddings_seq.restype = ctypes.POINTER(ctypes.c_float)

def llama_get_embeddings_seq(ctx: llama_context, seq_id: int) -> ptr[ctypes.c_float]:
    """Get the embeddings for a sequence id"""
    return libllama.llama_get_embeddings_seq(ctx, seq_id)

#
# Vocab
#

libllama.llama_token_get_text.argtypes = [llama_model_p, ctypes.c_int]
libllama.llama_token_get_text.restype = ctypes.c_char_p

def llama_token_get_text(model: llama_model, token: int) -> str:
    """Get the text representation of a token"""
    return libllama.llama_token_get_text(model, token).decode('utf-8')

libllama.llama_token_get_score.argtypes = [llama_model_p, ctypes.c_int]
libllama.llama_token_get_score.restype = ctypes.c_float

def llama_token_get_score(model: llama_model, token: int) -> float:
    """Get the score of a token"""
    return libllama.llama_token_get_score(model, token)

libllama.llama_token_get_attr.argtypes = [llama_model_p, ctypes.c_int]
libllama.llama_token_get_attr.restype = ctypes.c_int

def llama_token_get_attr(model: llama_model, token: int) -> int:
    """Get the attributes of a token"""
    return libllama.llama_token_get_attr(model, token)

libllama.llama_token_is_eog.argtypes = [llama_model_p, ctypes.c_int]
libllama.llama_token_is_eog.restype = ctypes.c_bool

def llama_token_is_eog(model: llama_model, token: int) -> bool:
    """Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)"""
    return libllama.llama_token_is_eog(model, token)

libllama.llama_token_is_control.argtypes = [llama_model_p, ctypes.c_int]
libllama.llama_token_is_control.restype = ctypes.c_bool

def llama_token_is_control(model: llama_model, token: int) -> bool:
    """Identify if Token Id is a control token or a render-able token"""
    return libllama.llama_token_is_control(model, token)

#
# Special tokens
#

libllama.llama_token_bos.argtypes = [llama_model_p]
libllama.llama_token_bos.restype = ctypes.c_int

def llama_token_bos(model: llama_model) -> int:
    """Get the BOS token ID. Returns -1 if not found."""
    return libllama.llama_token_bos(model)

libllama.llama_token_eos.argtypes = [llama_model_p]
libllama.llama_token_eos.restype = ctypes.c_int

def llama_token_eos(model: llama_model) -> int:
    """Get the EOS token ID. Returns -1 if not found."""
    return libllama.llama_token_eos(model)

libllama.llama_token_eot.argtypes = [llama_model_p]
libllama.llama_token_eot.restype = ctypes.c_int

def llama_token_eot(model: llama_model) -> int:
    """Get the end-of-turn token ID. Returns -1 if not found."""
    return libllama.llama_token_eot(model)

libllama.llama_token_cls.argtypes = [llama_model_p]
libllama.llama_token_cls.restype = ctypes.c_int

def llama_token_cls(model: llama_model) -> int:
    """Get the classification token ID. Returns -1 if not found."""
    return libllama.llama_token_cls(model)

libllama.llama_token_sep.argtypes = [llama_model_p]
libllama.llama_token_sep.restype = ctypes.c_int

def llama_token_sep(model: llama_model) -> int:
    """Get the sentence separator token ID. Returns -1 if not found."""
    return libllama.llama_token_sep(model)

libllama.llama_token_nl.argtypes = [llama_model_p]
libllama.llama_token_nl.restype = ctypes.c_int

def llama_token_nl(model: llama_model) -> int:
    """Get the newline token ID. Returns -1 if not found."""
    return libllama.llama_token_nl(model)

libllama.llama_token_pad.argtypes = [llama_model_p]
libllama.llama_token_pad.restype = ctypes.c_int

def llama_token_pad(model: llama_model) -> int:
    """Get the padding token ID. Returns -1 if not found."""
    return libllama.llama_token_pad(model)

libllama.llama_add_bos_token.argtypes = [llama_model_p]
libllama.llama_add_bos_token.restype = ctypes.c_bool

def llama_add_bos_token(model: llama_model) -> bool:
    """Whether BOS token should be added to tokenizations"""
    return libllama.llama_add_bos_token(model)

libllama.llama_add_eos_token.argtypes = [llama_model_p]
libllama.llama_add_eos_token.restype = ctypes.c_bool

def llama_add_eos_token(model: llama_model) -> bool:
    """Whether EOS token should be added to tokenizations"""
    return libllama.llama_add_eos_token(model)

@DEPRECATED(new_func_name="llama_token_fim_pre")
def llama_token_prefix(*args):
    pass

@DEPRECATED(new_func_name="llama_token_fim_mid")
def llama_token_middle(*args):
    pass

@DEPRECATED(new_func_name="llama_token_fim_suf")
def llama_token_suffix(*args):
    pass

libllama.llama_token_fim_pre.argtypes = [llama_model_p]
libllama.llama_token_fim_pre.restype = ctypes.c_int32

def llama_token_fim_pre(model: llama_model) -> int:
    """Get the infill prefix token ID. Returns -1 if not found."""
    return libllama.llama_token_fim_pre(model)

libllama.llama_token_fim_suf.argtypes = [llama_model_p]
libllama.llama_token_fim_suf.restype = ctypes.c_int32

def llama_token_fim_suf(model: llama_model) -> int:
    """Get the infill suffix token ID. Returns -1 if not found."""
    return libllama.llama_token_fim_suf(model)

libllama.llama_token_fim_mid.argtypes = [llama_model_p]
libllama.llama_token_fim_mid.restype = ctypes.c_int32

def llama_token_fim_mid(model: llama_model) -> int:
    """Get the infill middle token ID. Returns -1 if not found."""
    return libllama.llama_token_fim_mid(model)

libllama.llama_token_fim_pad.argtypes = [llama_model_p]
libllama.llama_token_fim_pad.restype = ctypes.c_int32

def llama_token_fim_pad(model: llama_model) -> int:
    """Get the infill pad token ID. Returns -1 if not found."""
    return libllama.llama_token_fim_pad(model)

libllama.llama_token_fim_rep.argtypes = [llama_model_p]
libllama.llama_token_fim_rep.restype = ctypes.c_int32

def llama_token_fim_rep(model: llama_model) -> int:
    """Get the infill repo token ID. Returns -1 if not found."""
    return libllama.llama_token_fim_rep(model)

libllama.llama_token_fim_sep.argtypes = [llama_model_p]
libllama.llama_token_fim_sep.restype = ctypes.c_int32

def llama_token_fim_sep(model: llama_model) -> int:
    """Get the infill separator token ID. Returns -1 if not found."""
    return libllama.llama_token_fim_sep(model)

#
# Tokenization
#

libllama.llama_tokenize.argtypes = [llama_model_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
libllama.llama_tokenize.restype = ctypes.c_int

def llama_tokenize(model: llama_model, text: bytes, text_len: int, tokens: ptr[ctypes.c_int32], n_tokens_max: int, add_special: bool, parse_special: bool) -> int:
    """
    Convert the provided text into tokens

    - model:
        The llama_model to use.
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
    """
    return libllama.llama_tokenize(model, text, text_len, tokens, n_tokens_max, add_special, parse_special)

libllama.llama_token_to_piece.argtypes = [llama_model_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
libllama.llama_token_to_piece.restype = ctypes.c_int

def llama_token_to_piece(model: llama_model, token: int, buf: ctypes.c_char_p, length: int, lstrip: int, special: bool) -> int:
    """Convert a single token to a piece of text"""
    return libllama.llama_token_to_piece(model, token, buf, length, lstrip, special)

libllama.llama_detokenize.argtypes = [llama_model_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
libllama.llama_detokenize.restype = ctypes.c_int

def llama_detokenize(model: llama_model, tokens: ptr[ctypes.c_int32], n_tokens: int, text: ctypes.c_char_p, text_len_max: int, remove_special: bool, unparse_special: bool) -> int:
    """
    Convert the provided tokens into text

    - model:
        The llama_model to use.
    - tokens:
        The tokens to convert.
    - n_tokens:
        TODO
    - text:
        The char pointer must be large enough to hold the resulting text.
    - text_len_max:
        TODO
    - remove_special:
        Allow to remove BOS and EOS tokens if model is configured to do so.
    - unparse_special:
        If true, special tokens are rendered in the output.
    
    Returns the number of chars/bytes on success, no more than text_len_max.
    Returns a negative number on failure - the number of chars/bytes that would
    have been returned.
    """
    return libllama.llama_detokenize(model, tokens, n_tokens, text, text_len_max, remove_special, unparse_special)

#
# Chat templating
#

libllama.llama_chat_apply_template.argtypes = [llama_model_p, ctypes.c_char_p, llama_chat_message_p, size_t, ctypes.c_bool, ctypes.c_char_p, ctypes.c_int32]
libllama.llama_chat_apply_template.restype = ctypes.c_int32

def llama_chat_apply_template(model: ptr[llama_model], tmpl: ptr[ctypes.c_char], chat: ptr[llama_chat_message], n_msg: int, add_ass: bool, buf: ptr[ctypes.c_char], length: int):
    return libllama.llama_chat_apply_template(model, tmpl, chat, n_msg, add_ass, buf, length)

libllama.llama_chat_builtin_templates.argtypes = [ctypes.POINTER(ctypes.c_char_p), size_t]
libllama.llama_chat_builtin_templates.restype = ctypes.c_int32

def llama_chat_builtin_templates(output: ptr[ctypes.c_char_p], len: int):
    return libllama.llama_chat_builtin_templates(output, len)

#
# Sampling
#

class llama_sampler_i(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)),  # can be NULL
        ("accept", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32)),  # can be NULL
        ("apply", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(llama_token_data_array))),  # required
        ("reset", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),  # can be NULL
        ("clone", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)),  # can be NULL if ctx is NULL
        ("free", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),  # can be NULL if ctx is NULL
    ]

llama_sampler_i_p = ctypes.POINTER(llama_sampler_i)

class llama_sampler(ctypes.Structure):
    _fields_ = [
        ("iface", ctypes.POINTER(llama_sampler_i)),
        ("ctx", ctypes.c_void_p),
    ]

llama_sampler_p = ctypes.POINTER(llama_sampler)

libllama.llama_sampler_name.argtypes = [llama_sampler_p]
libllama.llama_sampler_name.restype = ctypes.c_char_p

def llama_sampler_name(smpl: llama_sampler) -> str:
    """Get the name of a sampler"""
    return libllama.llama_sampler_name(smpl).decode('utf-8')

libllama.llama_sampler_accept.argtypes = [llama_sampler_p, ctypes.c_int]
libllama.llama_sampler_accept.restype = None

def llama_sampler_accept(smpl: llama_sampler, token: int) -> None:
    """Accept a token sampled by a sampler"""
    libllama.llama_sampler_accept(smpl, token)

libllama.llama_sampler_apply.argtypes = [llama_sampler_p, llama_token_data_array_p]
libllama.llama_sampler_apply.restype = None

def llama_sampler_apply(smpl: llama_sampler, cur_p: llama_token_data_array) -> None:
    """Apply a sampler to a token data array"""
    libllama.llama_sampler_apply(smpl, cur_p)

libllama.llama_sampler_reset.argtypes = [llama_sampler_p]
libllama.llama_sampler_reset.restype = None

def llama_sampler_reset(smpl: llama_sampler) -> None:
    """Reset a sampler"""
    libllama.llama_sampler_reset(smpl)

libllama.llama_sampler_clone.argtypes = [llama_sampler_p]
libllama.llama_sampler_clone.restype = llama_sampler_p

def llama_sampler_clone(smpl: llama_sampler) -> llama_sampler:
    """Clone a sampler"""
    return libllama.llama_sampler_clone(smpl)

libllama.llama_sampler_free.argtypes = [llama_sampler_p]
libllama.llama_sampler_free.restype = None

def llama_sampler_free(smpl: llama_sampler) -> None:
    """
    Free a sampler
    
    NOTE: Do not free if the sampler has been added to a llama_sampler_chain
    (via llama_sampler_chain_add)
    """
    libllama.llama_sampler_free(smpl)

#
# Sampler chain
#

libllama.llama_sampler_chain_init.argtypes = [llama_sampler_chain_params_p]
libllama.llama_sampler_chain_init.restype = llama_sampler_p

def llama_sampler_chain_init(params: llama_sampler_chain_params) -> llama_sampler:
    """Initialize a sampler chain"""
    return libllama.llama_sampler_chain_init(params)

libllama.llama_sampler_chain_add.argtypes = [llama_sampler_p, llama_sampler_p]
libllama.llama_sampler_chain_add.restype = None

def llama_sampler_chain_add(chain: llama_sampler, smpl: llama_sampler) -> None:
    """
    Add a sampler to a sampler chain
    
    Takes ownership of the sampler object and will free it when llama_sampler_free is called
    """
    libllama.llama_sampler_chain_add(chain, smpl)

libllama.llama_sampler_chain_get.argtypes = [llama_sampler_p, ctypes.c_int]
libllama.llama_sampler_chain_get.restype = llama_sampler_p

def llama_sampler_chain_get(chain: llama_sampler, i: int) -> llama_sampler:
    """Get a sampler from a sampler chain"""
    return libllama.llama_sampler_chain_get(chain, i)

libllama.llama_sampler_chain_n.argtypes = [llama_sampler_p]
libllama.llama_sampler_chain_n.restype = ctypes.c_int

def llama_sampler_chain_n(chain: llama_sampler) -> int:
    """Get the number of samplers in a sampler chain"""
    return libllama.llama_sampler_chain_n(chain)

libllama.llama_sampler_chain_remove.argtypes = [llama_sampler_p, ctypes.c_int]
libllama.llama_sampler_chain_remove.restype = llama_sampler_p

def llama_sampler_chain_remove(chain: llama_sampler, i: int) -> llama_sampler:
    """
    Remove a sampler from a sampler chain
    
    after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    """
    return libllama.llama_sampler_chain_remove(chain, i)

#
# Samplers
#

libllama.llama_sampler_init_greedy.argtypes = []
libllama.llama_sampler_init_greedy.restype = llama_sampler_p

def llama_sampler_init_greedy() -> llama_sampler:
    """Initialize a greedy sampler"""
    return libllama.llama_sampler_init_greedy()

libllama.llama_sampler_init_dist.argtypes = [ctypes.c_int]
libllama.llama_sampler_init_dist.restype = llama_sampler_p

def llama_sampler_init_dist(seed: int) -> llama_sampler:
    """Initialize a distribution sampler"""
    return libllama.llama_sampler_init_dist(seed)

libllama.llama_sampler_init_top_k.argtypes = [ctypes.c_int]
libllama.llama_sampler_init_top_k.restype = llama_sampler_p

def llama_sampler_init_top_k(k: int) -> llama_sampler:
    """Initialize a top-K sampler"""
    return libllama.llama_sampler_init_top_k(k)

libllama.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_int]
libllama.llama_sampler_init_top_p.restype = llama_sampler_p

def llama_sampler_init_top_p(p: float, min_keep: int) -> llama_sampler:
    """Initialize a top-p sampler"""
    return libllama.llama_sampler_init_top_p(p, min_keep)

libllama.llama_sampler_init_min_p.argtypes = [ctypes.c_float, ctypes.c_int]
libllama.llama_sampler_init_min_p.restype = llama_sampler_p

def llama_sampler_init_min_p(p: float, min_keep: int) -> llama_sampler:
    """Initialize a min-p sampler"""
    return libllama.llama_sampler_init_min_p(p, min_keep)

libllama.llama_sampler_init_typical.argtypes = [ctypes.c_float, ctypes.c_int]
libllama.llama_sampler_init_typical.restype = llama_sampler_p

def llama_sampler_init_typical(p: float, min_keep: int) -> llama_sampler:
    """Initialize a locally typical sampler"""
    return libllama.llama_sampler_init_typical(p, min_keep)

libllama.llama_sampler_init_temp.argtypes = [ctypes.c_float]
libllama.llama_sampler_init_temp.restype = llama_sampler_p

def llama_sampler_init_temp(t: float) -> llama_sampler:
    """
    Initialize a temperature sampler
    
    When `t` <= 0.0, the maximum logit is kept at it's original value, the rest are set to -inf
    """
    return libllama.llama_sampler_init_temp(t)

libllama.llama_sampler_init_temp_ext.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
libllama.llama_sampler_init_temp_ext.restype = llama_sampler_p

def llama_sampler_init_temp_ext(t: float, delta: float, exponent: float) -> llama_sampler:
    """Initialize an dynamic temperature / entropy sampler"""
    return libllama.llama_sampler_init_temp_ext(t, delta, exponent)

libllama.llama_sampler_init_xtc.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int]
libllama.llama_sampler_init_xtc.restype = llama_sampler_p

def llama_sampler_init_xtc(p: float, t: float, min_keep: int, seed: int) -> llama_sampler:
    """Initialize an XTC sampler"""
    return libllama.llama_sampler_init_xtc(p, t, min_keep, seed)

libllama.llama_sampler_init_mirostat.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
libllama.llama_sampler_init_mirostat.restype = llama_sampler_p

def llama_sampler_init_mirostat(seed: int, tau: float, eta: float) -> llama_sampler:
    """Initialize a Mirostat sampler"""
    return libllama.llama_sampler_init_mirostat(seed, tau, eta)

libllama.llama_sampler_init_mirostat_v2.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
libllama.llama_sampler_init_mirostat_v2.restype = llama_sampler_p

def llama_sampler_init_mirostat_v2(seed: int, tau: float, eta: float) -> llama_sampler:
    """Initialize a Mirostat v2 sampler"""
    return libllama.llama_sampler_init_mirostat_v2(seed, tau, eta)

libllama.llama_sampler_init_grammar.argtypes = [llama_model_p, ctypes.c_char_p, ctypes.c_char_p]
libllama.llama_sampler_init_grammar.restype = llama_sampler_p

def llama_sampler_init_grammar(model: llama_model, grammar_str: str, grammar_root: str) -> llama_sampler:
    """Initialize a grammar sampler"""
    return libllama.llama_sampler_init_grammar(model, grammar_str.encode('utf-8'), grammar_root.encode('utf-8'))

libllama.llama_sampler_init_penalties.argtypes = [ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float]
libllama.llama_sampler_init_penalties.restype = llama_sampler_p

def llama_sampler_init_penalties(penalty_last_n: int, penalty_repeat: float, penalty_freq: float, penalty_present: float) -> llama_sampler:
    """Initialize a penalties sampler"""
    return libllama.llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)

libllama.llama_sampler_init_dry.argtypes = [llama_model_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
libllama.llama_sampler_init_dry.restype = llama_sampler_p

def llama_sampler_init_dry(model: llama_model, dry_multiplier: float, dry_base: float, dry_allowed_length: int, dry_penalty_last_n: int, seq_breakers: ptr[ctypes.c_char_p], num_breakers: int) -> llama_sampler:
    """Initialize a DRY sampler"""
    return libllama.llama_sampler_init_dry(model, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers)

libllama.llama_sampler_init_logit_bias.argtypes = [ctypes.c_int, ctypes.c_int, llama_logit_bias_p]
libllama.llama_sampler_init_logit_bias.restype = llama_sampler_p

def llama_sampler_init_logit_bias(n_vocab: int, n_logit_bias: int, logit_bias: ptr[llama_logit_bias]) -> llama_sampler:
    """Initialize a logit bias sampler"""
    return libllama.llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias)

libllama.llama_sampler_init_infill.argtypes = [llama_model_p]
libllama.llama_sampler_init_infill.restype = llama_sampler_p

def llama_sampler_init_infill(model: llama_model) -> llama_sampler:
    """
    Initialize an infill sampler
    
    This sampler is meant to be used for fill-in-the-middle infilling. It's supposed to be used after top_k + top_p sampling
    """
    return libllama.llama_sampler_init_infill(model)

libllama.llama_sampler_get_seed.argtypes = [llama_sampler_p]
libllama.llama_sampler_get_seed.restype = ctypes.c_int

def llama_sampler_get_seed(smpl: llama_sampler) -> int:
    """
    Get the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    """
    return libllama.llama_sampler_get_seed(smpl)

libllama.llama_sampler_sample.argtypes = [llama_sampler_p, llama_context_p, ctypes.c_int]
libllama.llama_sampler_sample.restype = ctypes.c_int

def llama_sampler_sample(smpl: llama_sampler, ctx: llama_context, idx: int) -> int:
    """
    Sample and accept a token from the idx-th output of the last evaluation
    """
    return libllama.llama_sampler_sample(smpl, ctx, idx)

#
# Model split
#

libllama.llama_split_path.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
libllama.llama_split_path.restype = ctypes.c_int

def llama_split_path(split_path: ctypes.c_char_p, maxlen: int, path_prefix: str, split_no: int, split_count: int) -> int:
    """Build a split GGUF final path for a chunk"""
    return libllama.llama_split_path(split_path, maxlen, path_prefix.encode('utf-8'), split_no, split_count)

libllama.llama_split_prefix.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
libllama.llama_split_prefix.restype = ctypes.c_int

def llama_split_prefix(split_prefix: ctypes.c_char_p, maxlen: int, split_path: str, split_no: int, split_count: int) -> int:
    """Extract the path prefix from a split path"""
    return libllama.llama_split_prefix(split_prefix, maxlen, split_path.encode('utf-8'), split_no, split_count)

#
# Print system info
#

libllama.llama_print_system_info.argtypes = []
libllama.llama_print_system_info.restype = ctypes.c_char_p

def llama_print_system_info() -> None:
    """Get system information"""
    text = libllama.llama_print_system_info()
    text = text.decode()
    print(text, file=sys.stderr, flush=True)

#
# Log callback
#

libllama.llama_log_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libllama.llama_log_set.restype = None

def llama_log_set(log_callback: ctypes.c_void_p, user_data: ctypes.c_void_p) -> None:
    """Set a callback for logging events"""
    libllama.llama_log_set(log_callback, user_data)

#
# Performance utils
#

class llama_perf_context_data(ctypes.Structure):
    _fields_ = [
        ("t_start_ms", ctypes.c_double),
        ("t_load_ms", ctypes.c_double),
        ("t_p_eval_ms", ctypes.c_double),
        ("t_eval_ms", ctypes.c_double),
        ("n_p_eval", ctypes.c_int32),
        ("n_eval", ctypes.c_int32),
    ]

llama_perf_context_data_p = ctypes.POINTER(llama_perf_context_data)

class llama_perf_sampler_data(ctypes.Structure):
    _fields_ = [
        ("t_sample_ms", ctypes.c_double),
        ("n_sample", ctypes.c_int32),
    ]

llama_perf_sampler_data_p = ctypes.POINTER(llama_perf_sampler_data)

# NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.

libllama.llama_perf_context.argtypes = [llama_context_p]
libllama.llama_perf_context.restype = llama_perf_context_data_p

def llama_perf_context(ctx: llama_context) -> llama_perf_context_data:
    """
    AVOID USING

    Get performance data for a context
    """
    return libllama.llama_perf_context(ctx)

libllama.llama_perf_context_print.argtypes = [llama_context_p]
libllama.llama_perf_context_print.restype = ctypes.c_char_p

def llama_perf_context_print(ctx: llama_context) -> None:
    """
    AVOID USING

    Print performance data for a context
    """
    results_str = libllama.llama_perf_context_print(ctx)
    print(results_str, file=sys.stderr, flush=True)

libllama.llama_perf_context_reset.argtypes = [llama_context_p]
libllama.llama_perf_context_reset.restype = None

def llama_perf_context_reset(ctx: llama_context) -> None:
    """
    AVOID USING
    
    Reset performance data for a context
    """
    libllama.llama_perf_context_reset(ctx)

# NOTE: the following work only with samplers constructed via llama_sampler_chain_init

libllama.llama_perf_sampler.argtypes = [llama_sampler_p]
libllama.llama_perf_sampler.restype = llama_perf_sampler_data_p

def llama_perf_sampler(smpl: llama_sampler) -> llama_perf_sampler_data:
    """Get performance data for a sampler"""
    return libllama.llama_perf_sampler(smpl)

libllama.llama_perf_sampler_print.argtypes = [llama_sampler_p]
libllama.llama_perf_sampler_print.restype = ctypes.c_char_p

def llama_perf_sampler_print(smpl: llama_sampler) -> None:
    """Print performance data for a sampler"""
    results_str = libllama.llama_perf_sampler_print(smpl)
    print(results_str, file=sys.stderr, flush=True)

libllama.llama_perf_sampler_reset.argtypes = [llama_sampler_p]
libllama.llama_perf_sampler_reset.restype = None

def llama_perf_sampler_reset(smpl: llama_sampler) -> None:
    """Reset performance data for a sampler"""
    libllama.llama_perf_sampler_reset(smpl)

#
# End of LLAMA_API
#

class _internals:
    #
    # This used to be a separate module but ctypes was throwing fits about
    # type mismatches. Easier to put it here.
    #
    MAX_TOKEN_LENGTH = 256
    """The maximum supported length of a single token's text, in bytes"""

    class LogitBiasArray:
        """Type hint for `ctypes.Array[llama_logit_bias]` of arbitrary length"""

    def decode_pp(
        ctx: ptr[llama_context],
        pos: int,
        tokens: list[int],
        n_tokens: int
    ) -> None:
        """
        ### INTERNAL

        Decode with batch size > 1 (prompt processing)
        """
        batch = llama_batch_init(n_tokens=n_tokens, embd=0, n_seq_max=1)
        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            batch.token[i] = tokens[i]
            batch.pos[i] = pos + i
            batch.seq_id[i][0] = 0
            batch.n_seq_id[i] = 1
            batch.logits[i] = False
        batch.logits[n_tokens - 1] = True
        ret = llama_decode(ctx, batch)
        llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(
                f'decode_pp: llama_decode failed with status code {ret}'
            )

    def decode_tg(
        ctx: ptr[llama_context],
        pos: int,
        token: int
    ) -> None:
        """
        ### INTERNAL

        Decode with batch size == 1 (text generation)
        """
        batch = llama_batch_init(n_tokens=1, embd=0, n_seq_max=1)
        batch.n_tokens = 1
        batch.token[0] = token
        batch.pos[0] = pos
        batch.seq_id[0][0] = 0
        batch.n_seq_id[0] = 1
        batch.logits[0] = True
        ret = llama_decode(ctx, batch)
        llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(
                f'decode_tg: llama_decode failed with status code {ret}'
            )

    greedy_sampler = llama_sampler_init_greedy()

    def sample_greedy(ctx: ptr[llama_context]) -> int:
        """
        ### INTERNAL

        Sample the most likely token
        """
        return llama_sampler_sample(_internals.greedy_sampler, ctx, -1)

    def tokenize(
        model: ptr[llama_model],
        text_bytes: bytes,
        n_tokens_max: int,
        add_special: bool,
        parse_special: bool,
    ) -> list[int]:
        """
        ### INTERNAL

        Convert the provided UTF-8 encoded text into tokens

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
            space.
        """
        # unlike detokenization, this buffer is created and destroyed as needed
        # because it could potentially be quite large - each token takes 4 bytes
        tokens_buf = (ctypes.c_int32 * n_tokens_max)()
        n_tokens = llama_tokenize(
            model=model,
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

    # this buffer is re-used every time llama_token_to_piece() is called
    # it is only 256 bytes, so OK to keep in memory
    detok_buffer = ctypes.create_string_buffer(MAX_TOKEN_LENGTH)

    def token_to_piece(
        model: ptr[llama_model], token: int, special: bool
    ) -> bytes:
        """
        ### INTERNAL

        Convert token ID to text bytes
        """
        n_bytes = llama_token_to_piece(
            model=model,
            token=token,
            buf=_internals.detok_buffer,
            length=_internals.MAX_TOKEN_LENGTH,
            lstrip=0, # skip up to 'lstrip' leading spaces
            special=special
        )
        if n_bytes > _internals.MAX_TOKEN_LENGTH:
            raise ValueError(
                f"token_to_piece: the token with ID {token} requires a "
                f"buffer of size {n_bytes}, but the maximum buffer size is "
                f"{_internals.MAX_TOKEN_LENGTH}"
            )
        # NOTE: do not just do buf.value.decode() because the token could
        #       possibly be a part of a utf-8 bytestring, but not a valid utf-8
        #       string itself. let the caller handle this
        return _internals.detok_buffer.raw[:n_bytes]

    def detokenize(
        model: ptr[llama_model],
        tokens: Iterable[int],
        special: bool
    ) -> bytes:
        """
        ### INTERNAL

        Convert the provided tokens into UTF-8 encoded text

        - special:
            If True, special tokens are rendered in the output
        """
        # this function is just like token_to_piece but in a loop
        detok_bytes = b""
        for token in tokens:
            n_bytes = llama_token_to_piece(
                model=model,
                token=token,
                buf=_internals.detok_buffer,
                length=_internals.MAX_TOKEN_LENGTH,
                lstrip=0, # skip up to 'lstrip' leading spaces
                special=special
            )
            if n_bytes > _internals.MAX_TOKEN_LENGTH:
                raise ValueError(
                    f"detokenize: the token with ID {token} requires a buffer "
                    f"of size {n_bytes}, but the maximum buffer size is "
                    f"{_internals.MAX_TOKEN_LENGTH}"
                )
            detok_bytes += _internals.detok_buffer.raw[:n_bytes]
        return detok_bytes

    def get_length(
        model: ptr[llama_model],
        text_bytes: bytes,
        add_special: bool,
        parse_special: bool,
    ) -> int:
        """
        ### INTERNAL

        Return the length of a given text, as measured in tokens
        """
        return -llama_tokenize(
            model=model,
            text=text_bytes,
            text_len=len(text_bytes),
            tokens=NULL,
            n_tokens_max=0,
            add_special=add_special,
            parse_special=parse_special
        )
    
    def perf_ctx_print_and_reset(ctx: ptr[llama_context]) -> None:
        llama_perf_context_print(ctx)
        llama_perf_context_reset(ctx)

    def perf_smpl_print_and_reset(smpl: ptr[llama_sampler]) -> None:
        llama_perf_sampler_print(smpl)
        llama_perf_sampler_reset(smpl)
    
    def get_logit_bias_array(logit_biases: dict[int, float]) -> LogitBiasArray:
        if len(logit_biases) == 0:
            raise ValueError(f'logit_biases parameter cannot be empty')
        LogitBiasArrayType = llama_logit_bias * len(logit_biases)
        arr = LogitBiasArrayType()
        i = 0
        for k, v in logit_biases.items():
            arr[i].token = k
            arr[i].bias = v
            i += 1
        return arr
    
def main():

    # Handy-dandy basic test of libllama

    import os

    #test_model_path = "/Users/dylan/Documents/AI/models/Llama-3.2-1B-Instruct-q8_0-q8_0.gguf"
    test_model_path = "/Users/dylan/Documents/AI/models/Meta-Llama-3.1-8B-Instruct-q8_0-q6_K.gguf"

    if not os.path.exists(test_model_path):
        raise FileNotFoundError(f'the model {test_model_path!r} was not found')
    
    print("-" * 80)

    llama_backend_init()

    model_params = llama_model_default_params()
    model_params.n_gpu_layers = MAX_OFFLOAD_LAYERS
    model_params.use_mmap = True

    model = llama_load_model_from_file(test_model_path, model_params)

    ctx_params = llama_context_default_params()
    ctx_params.n_ctx = 8192
    ctx_params.n_batch = 2048
    ctx_params.n_threads = 4
    ctx_params.n_threads_batch = 8
    ctx_params.offload_kqv = True
    ctx_params.flash_attn = True

    ctx = llama_new_context_with_model(model, ctx_params)

    llama_set_n_threads(ctx, ctx_params.n_threads, ctx_params.n_threads_batch)

    logit_biases = {67722: -100.00, 55152: -100.00}
    logit_bias_arr = _internals.get_logit_bias_array(logit_biases)

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())

    llama_sampler_chain_add(
        smpl, llama_sampler_init_logit_bias(
            n_vocab=llama_n_vocab(model),
            n_logit_bias=len(logit_biases),
            logit_bias=logit_bias_arr
        )
    )
    llama_sampler_chain_add(
        smpl, llama_sampler_init_greedy()
    )

    def sample_logit_bias(ctx: ptr[llama_context]) -> int:
        id = llama_sampler_sample(smpl, ctx, -1)
        llama_sampler_accept(smpl, id)
        return id

    tokens = [128000, 128006, 9125, 128007, 271, 2675, 527, 264, 11190, 15592, 18328, 13, 128009, 128006, 882, 128007, 271, 3923, 374, 55152, 11495, 369, 30, 128009, 128006, 78191, 128007, 271]
    ctx_tokens = tokens
    pos = 0
    _internals.decode_pp(ctx, pos, tokens, len(tokens))
    pos += len(tokens)
    id = sample_logit_bias(ctx)
    ctx_tokens.append(id)
    tok_txt = _internals.token_to_piece(model, id, True).decode()
    print(tok_txt, end='', flush=True)
    while pos < llama_n_ctx(ctx):
        _internals.decode_tg(ctx, pos, id)
        pos += 1
        id = sample_logit_bias(ctx)
        ctx_tokens.append(id)
        tok_txt = _internals.token_to_piece(model, id, True).decode()
        print(tok_txt, end='', flush=True)

if __name__ == '__main__':
    main()