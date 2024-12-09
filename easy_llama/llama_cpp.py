# llama_cpp.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import ctypes

from enum import IntEnum

class Deprecated(Exception):
    """
    Exception raised when calling functions marked with DEPRECATED in the
    llama C interface
    """

#
# Import shared library
#

libllama = ctypes.CDLL('/Users/dylan/Documents/AI/easy-llama/easy_llama/libllama.dylib')

#
# Type hints
#

class ptr:
    """
    Generic type hint representing any `ctypes` pointer
    """

class Types:
    """
    Type hints for `ctypes` based on libllama
    """

    bool = ctypes.c_bool

    size_t = ctypes.c_ulong

    llama_model   = ctypes.c_void_p
    llama_context = ctypes.c_void_p

    llama_pos    = ctypes.c_int32
    llama_token  = ctypes.c_int32
    llama_seq_id = ctypes.c_int32

    llama_token_data       = ctypes.c_void_p
    llama_token_data_array = ctypes.c_void_p

    llama_batch = ctypes.c_void_p

    llama_model_kv_override = ctypes.c_void_p

    llama_model_params = ctypes.c_void_p

    llama_context_params = ctypes.c_void_p

    llama_model_quantize_params = ctypes.c_void_p

    llama_logit_bias = ctypes.c_void_p

    llama_sampler_chain_params = ctypes.c_void_p

    llama_chat_message = ctypes.c_void_p

    llama_lora_adapter = ctypes.c_void_p

    llama_kv_cache_view_cell = ctypes.c_void_p

    llama_kv_cache_view = ctypes.c_void_p

    llama_sampler_i = ctypes.c_void_p

    llama_sampler = ctypes.c_void_p

    llama_perf_context_data = ctypes.c_void_p

    llama_perf_sampler_data = ctypes.c_void_p

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

# this constant is added by ddh0
# it is the maximum value for int32, it is used as the value for n_gpu_layers
# when all layers should be offloaded
MAX_OFFLOAD_LAYERS = 0x7FFFFFFF

#
# Stuff from llama.cpp/ggml/include/ggml.h
#

GGML_EXIT_SUCCESS = 0
GGML_EXIT_ABORTED = 1

GGML_ROPE_TYPE_NEOX = 2

GGUF_MAGIC = 0x46554747 # "GGUF"

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
    LLAMA_ROPE_TYPE_NONE = -1
    LLAMA_ROPE_TYPE_NORM = 0
    LLAMA_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX

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

# NOTE: enum llama_model_kv_override_type { ... } is missing for now

#
# Helpers for getting default parameters
#

def llama_model_default_params() -> Types.llama_model_params:
    """Get the default parameters for a llama model"""
    libllama.llama_model_default_params.argtypes = []
    libllama.llama_model_default_params.restype = ctypes.c_void_p
    return libllama.llama_model_default_params()

def llama_context_default_params() -> Types.llama_context_params:
    """Get the default parameters for a llama context"""
    libllama.llama_context_default_params.argtypes = []
    libllama.llama_context_default_params.restype = ctypes.c_void_p
    return libllama.llama_context_default_params()

def llama_sampler_chain_default_params() -> Types.llama_sampler_chain_params:
    """Get the default parameters for a sampler chain"""
    libllama.llama_sampler_chain_default_params.argtypes = []
    libllama.llama_sampler_chain_default_params.restype = ctypes.c_void_p
    return libllama.llama_sampler_chain_default_params()

def llama_model_quantize_default_params() -> Types.llama_model_quantize_params:
    """Get the default parameters for model quantization"""
    libllama.llama_model_quantize_default_params.argtypes = []
    libllama.llama_model_quantize_default_params.restype = ctypes.c_void_p
    return libllama.llama_model_quantize_default_params()

#
# Setup and teardown
#

def llama_backend_init() -> None:
    """Initialize the llama + ggml backend"""
    libllama.llama_backend_init.argtypes = []
    libllama.llama_backend_init.restype = None
    libllama.llama_backend_init()

def llama_numa_init(numa: int) -> None:
    """Initialize NUMA optimizations globally"""
    libllama.llama_numa_init.argtypes = [ctypes.c_int]
    libllama.llama_numa_init.restype = None
    libllama.llama_numa_init(numa)

def llama_attach_threadpool(ctx: Types.llama_context, ggml_threadpool: ptr, threadpool_batch: ptr) -> None:
    """Attach a threadpool to a llama_context"""
    libllama.llama_attach_threadpool.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_attach_threadpool.restype = None
    libllama.llama_attach_threadpool(ctx, ggml_threadpool, threadpool_batch)

def llama_detach_threadpool(ctx: Types.llama_context) -> None:
    """Detach a threadpool from a llama_context"""
    libllama.llama_detach_threadpool.argtypes = [ctypes.c_void_p]
    libllama.llama_detach_threadpool.restype = None
    libllama.llama_detach_threadpool(ctx)

def llama_backend_free() -> None:
    """
    Free the llama + ggml backend
    
    Call once at the end of the program - currently only used for MPI
    """
    libllama.llama_backend_free.argtypes = []
    libllama.llama_backend_free.restype = None
    libllama.llama_backend_free()

def llama_load_model_from_file(path_model: str, params: Types.llama_model_params) -> Types.llama_model:
    """Load a llama model from a file"""
    libllama.llama_load_model_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    libllama.llama_load_model_from_file.restype = ctypes.c_void_p
    return libllama.llama_load_model_from_file(path_model.encode('utf-8'), params)

def llama_free_model(model: Types.llama_model) -> None:
    """Free a model"""
    libllama.llama_free_model.argtypes = [ctypes.c_void_p]
    libllama.llama_free_model.restype = None
    libllama.llama_free_model(model)

def llama_new_context_with_model(model: Types.llama_model, params: Types.llama_context_params) -> Types.llama_context:
    """Create a new llama context with a loaded model"""
    libllama.llama_new_context_with_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_new_context_with_model.restype = ctypes.c_void_p
    return libllama.llama_new_context_with_model(model, params)

def llama_free(ctx: ptr) -> None:
    """Frees all allocated memory"""
    libllama.llama_free.argtypes = [ctypes.c_void_p]
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
# Getters for llama_context attributes
#

def llama_n_ctx(ctx: Types.llama_context) -> int:
    """Get the context size"""
    libllama.llama_n_ctx.argtypes = [ctypes.c_void_p]
    libllama.llama_n_ctx.restype = ctypes.c_int
    return libllama.llama_n_ctx(ctx)

def llama_n_batch(ctx: Types.llama_context) -> int:
    """Get the logical maximum batch size"""
    libllama.llama_n_batch.argtypes = [ctypes.c_void_p]
    libllama.llama_n_batch.restype = ctypes.c_int
    return libllama.llama_n_batch(ctx)

def llama_n_ubatch(ctx: Types.llama_context) -> int:
    """Get the physical maximum batch size"""
    libllama.llama_n_ubatch.argtypes = [ctypes.c_void_p]
    libllama.llama_n_ubatch.restype = ctypes.c_int
    return libllama.llama_n_ubatch(ctx)

def llama_n_seq_max(ctx: Types.llama_context) -> int:
    """Get the maximum number of sequences"""
    libllama.llama_n_seq_max.argtypes = [ctypes.c_void_p]
    libllama.llama_n_seq_max.restype = ctypes.c_int
    return libllama.llama_n_seq_max(ctx)

#
# Getters for model attributes
#

def llama_n_vocab(model: Types.llama_model) -> int:
    """Get the number of tokens in the vocabulary"""
    libllama.llama_n_vocab.argtypes = [ctypes.c_void_p]
    libllama.llama_n_vocab.restype = ctypes.c_int
    return libllama.llama_n_vocab(model)

def llama_n_ctx_train(model: Types.llama_model) -> int:
    """Get the context size used during training"""
    libllama.llama_n_ctx_train.argtypes = [ctypes.c_void_p]
    libllama.llama_n_ctx_train.restype = ctypes.c_int
    return libllama.llama_n_ctx_train(model)

def llama_n_embd(model: Types.llama_model) -> int:
    """Get the embedding size"""
    libllama.llama_n_embd.argtypes = [ctypes.c_void_p]
    libllama.llama_n_embd.restype = ctypes.c_int
    return libllama.llama_n_embd(model)

def llama_n_layer(model: Types.llama_model) -> int:
    """Get the number of layers"""
    libllama.llama_n_layer.argtypes = [ctypes.c_void_p]
    libllama.llama_n_layer.restype = ctypes.c_int
    return libllama.llama_n_layer(model)

def llama_n_head(model: Types.llama_model) -> int:
    """Get the number of attention heads"""
    libllama.llama_n_head.argtypes = [ctypes.c_void_p]
    libllama.llama_n_head.restype = ctypes.c_int
    return libllama.llama_n_head(model)

# More getters for llama_context ...

def llama_get_model(ctx: Types.llama_context) -> Types.llama_model:
    """Get the model associated with a context"""
    libllama.llama_get_model.argtypes = [ctypes.c_void_p]
    libllama.llama_get_model.restype = ctypes.c_void_p
    return libllama.llama_get_model(ctx)

def llama_pooling_type(ctx: Types.llama_context) -> int:
    """Get the pooling type used by a context"""
    libllama.llama_pooling_type.argtypes = [ctypes.c_void_p]
    libllama.llama_pooling_type.restype = ctypes.c_int
    return libllama.llama_pooling_type(ctx)

# More getters for llama_model ...

def llama_vocab_type(model: Types.llama_model) -> int:
    """Get the vocabulary type used by a model"""
    libllama.llama_vocab_type.argtypes = [ctypes.c_void_p]
    libllama.llama_vocab_type.restype = ctypes.c_int
    return libllama.llama_vocab_type(model)

def llama_rope_type(model: Types.llama_model) -> int:
    """Get the RoPE type used by a model"""
    libllama.llama_rope_type.argtypes = [ctypes.c_void_p]
    libllama.llama_rope_type.restype = ctypes.c_int
    return libllama.llama_rope_type(model)

def llama_rope_freq_scale_train(model: Types.llama_model) -> float:
    """Get the RoPE frequency scaling factor used during training"""
    libllama.llama_rope_freq_scale_train.argtypes = [ctypes.c_void_p]
    libllama.llama_rope_freq_scale_train.restype = ctypes.c_float
    return libllama.llama_rope_freq_scale_train(model)

def llama_model_meta_val_str(model: Types.llama_model, key: str, buf: ptr, buf_size: int) -> int:
    """Get a metadata value as a string"""
    libllama.llama_model_meta_val_str.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_model_meta_val_str.restype = ctypes.c_int
    return libllama.llama_model_meta_val_str(model, key.encode('utf-8'), buf, buf_size)

def llama_model_meta_count(model: Types.llama_model) -> int:
    """Get the number of metadata key-value pairs"""
    libllama.llama_model_meta_count.argtypes = [ctypes.c_void_p]
    libllama.llama_model_meta_count.restype = ctypes.c_int
    return libllama.llama_model_meta_count(model)

def llama_model_meta_key_by_index(model: Types.llama_model, i: int, buf: ptr, buf_size: int) -> int:
    """Get a metadata key by index"""
    libllama.llama_model_meta_key_by_index.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_model_meta_key_by_index.restype = ctypes.c_int
    return libllama.llama_model_meta_key_by_index(model, i, buf, buf_size)

def llama_model_meta_val_str_by_index(model: Types.llama_model, i: int, buf: ptr, buf_size: int) -> int:
    """Get a metadata value by index"""
    libllama.llama_model_meta_val_str_by_index.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_model_meta_val_str_by_index.restype = ctypes.c_int
    return libllama.llama_model_meta_val_str_by_index(model, i, buf, buf_size)

def llama_model_desc(model: Types.llama_model, buf: ptr, buf_size: int) -> int:
    """Get a string describing the model type"""
    libllama.llama_model_desc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_model_desc.restype = ctypes.c_int
    return libllama.llama_model_desc(model, buf, buf_size)

def llama_model_size(model: Types.llama_model) -> int:
    """Get the total size of all tensors in the model"""
    libllama.llama_model_size.argtypes = [ctypes.c_void_p]
    libllama.llama_model_size.restype = ctypes.c_int
    return libllama.llama_model_size(model)

def llama_model_n_params(model: Types.llama_model) -> int:
    """Get the total number of parameters in the model"""
    libllama.llama_model_n_params.argtypes = [ctypes.c_void_p]
    libllama.llama_model_n_params.restype = ctypes.c_int
    return libllama.llama_model_n_params(model)

def llama_get_model_tensor(model: Types.llama_model, name: str) -> ptr:
    """Get a model tensor by name"""
    libllama.llama_get_model_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libllama.llama_get_model_tensor.restype = ctypes.c_void_p
    return libllama.llama_get_model_tensor(model, name.encode('utf-8'))

def llama_model_has_encoder(model: Types.llama_model) -> bool:
    """Check if the model has an encoder"""
    libllama.llama_model_has_encoder.argtypes = [ctypes.c_void_p]
    libllama.llama_model_has_encoder.restype = ctypes.c_bool
    return libllama.llama_model_has_encoder(model)

def llama_model_has_decoder(model: Types.llama_model) -> bool:
    """Check if the model has a decoder"""
    libllama.llama_model_has_decoder.argtypes = [ctypes.c_void_p]
    libllama.llama_model_has_decoder.restype = ctypes.c_bool
    return libllama.llama_model_has_decoder(model)

def llama_model_decoder_start_token(model: Types.llama_model) -> int:
    """Get the start token for the decoder"""
    libllama.llama_model_decoder_start_token.argtypes = [ctypes.c_void_p]
    libllama.llama_model_decoder_start_token.restype = ctypes.c_int
    return libllama.llama_model_decoder_start_token(model)

def llama_model_is_recurrent(model: Types.llama_model) -> bool:
    """Check if the model is recurrent"""
    libllama.llama_model_is_recurrent.argtypes = [ctypes.c_void_p]
    libllama.llama_model_is_recurrent.restype = ctypes.c_bool
    return libllama.llama_model_is_recurrent(model)

#
# Quantization
#

def llama_model_quantize(fname_inp: str, fname_out: str, params: Types.llama_model_quantize_params) -> int:
    """Quantize a model. Returns 0 on success"""
    libllama.llama_model_quantize.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    libllama.llama_model_quantize.restype = ctypes.c_int
    return libllama.llama_model_quantize(fname_inp.encode('utf-8'), fname_out.encode('utf-8'), params)

#
# LoRA
#

def llama_lora_adapter_init(model: Types.llama_model, path_lora: str) -> Types.llama_lora_adapter:
    """Initialize a LoRA adapter"""
    libllama.llama_lora_adapter_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libllama.llama_lora_adapter_init.restype = ctypes.c_void_p
    return libllama.llama_lora_adapter_init(model, path_lora.encode('utf-8'))

def llama_lora_adapter_set(ctx: Types.llama_context, adapter: ptr, scale: float) -> int:
    """Set a LoRA adapter for a context"""
    libllama.llama_lora_adapter_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float]
    libllama.llama_lora_adapter_set.restype = ctypes.c_int
    return libllama.llama_lora_adapter_set(ctx, adapter, scale)

def llama_lora_adapter_remove(ctx: Types.llama_context, adapter: ptr) -> int:
    """Remove a LoRA adapter from a context"""
    libllama.llama_lora_adapter_remove.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_lora_adapter_remove.restype = ctypes.c_int
    return libllama.llama_lora_adapter_remove(ctx, adapter)

def llama_lora_adapter_clear(ctx: Types.llama_context) -> None:
    """Clear all LoRA adapters from a context"""
    libllama.llama_lora_adapter_clear.argtypes = [ctypes.c_void_p]
    libllama.llama_lora_adapter_clear.restype = None
    libllama.llama_lora_adapter_clear(ctx)

def llama_lora_adapter_free(adapter: Types.llama_lora_adapter) -> None:
    """Free a LoRA adapter"""
    libllama.llama_lora_adapter_free.argtypes = [ctypes.c_void_p]
    libllama.llama_lora_adapter_free.restype = None
    libllama.llama_lora_adapter_free(adapter)

#
# Control vector
#

def llama_control_vector_apply(ctx: Types.llama_context, data: ptr, len: int, n_embd: int, il_start: int, il_end: int) -> int:
    """Apply a control vector to a context"""
    libllama.llama_control_vector_apply.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_control_vector_apply.restype = ctypes.c_int
    return libllama.llama_control_vector_apply(ctx, data, len, n_embd, il_start, il_end)

#
# KV cache
#

def llama_kv_cache_view_init(ctx: Types.llama_context, n_seq_max: int) -> Types.llama_kv_cache_view:
    """
    DEBUG ONLY

    Create an empty KV cache view (use only for debugging purposes)
    """
    libllama.llama_kv_cache_view_init.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_kv_cache_view_init.restype = ctypes.c_void_p
    return libllama.llama_kv_cache_view_init(ctx, n_seq_max)

def llama_kv_cache_view_free(view: Types.llama_kv_cache_view) -> None:
    """
    DEBUG ONLY

    Free a KV cache view
    """
    libllama.llama_kv_cache_view_free.argtypes = [ctypes.c_void_p]
    libllama.llama_kv_cache_view_free.restype = None
    libllama.llama_kv_cache_view_free(view)

def llama_kv_cache_view_update(ctx: Types.llama_context, view: Types.llama_kv_cache_view) -> None:
    """
    DEBUG ONLY
    
    Update a KV cache view with the current state of the KV cache
    """
    libllama.llama_kv_cache_view_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_kv_cache_view_update.restype = None
    libllama.llama_kv_cache_view_update(ctx, view)

def llama_get_kv_cache_token_count(ctx: Types.llama_context) -> int:
    """
    DEBUG ONLY & SLOW
    
    Get the number of tokens in the KV cache
    """
    libllama.llama_get_kv_cache_token_count.argtypes = [ctypes.c_void_p]
    libllama.llama_get_kv_cache_token_count.restype = ctypes.c_int
    return libllama.llama_get_kv_cache_token_count(ctx)

def llama_get_kv_cache_used_cells(ctx: Types.llama_context) -> int:
    """Get the number of used KV cells"""
    libllama.llama_get_kv_cache_used_cells.argtypes = [ctypes.c_void_p]
    libllama.llama_get_kv_cache_used_cells.restype = ctypes.c_int
    return libllama.llama_get_kv_cache_used_cells(ctx)

def llama_kv_cache_clear(ctx: Types.llama_context) -> None:
    """Clear the KV cache"""
    libllama.llama_kv_cache_clear.argtypes = [ctypes.c_void_p]
    libllama.llama_kv_cache_clear.restype = None
    libllama.llama_kv_cache_clear(ctx)

def llama_kv_cache_seq_rm(ctx: Types.llama_context, seq_id: int, p0: int, p1: int) -> bool:
    """Remove tokens from a sequence in the KV cache"""
    libllama.llama_kv_cache_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_kv_cache_seq_rm.restype = ctypes.c_bool
    return libllama.llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)

def llama_kv_cache_seq_cp(ctx: Types.llama_context, seq_id_src: int, seq_id_dst: int, p0: int, p1: int) -> None:
    """Copy tokens from one sequence to another in the KV cache"""
    libllama.llama_kv_cache_seq_cp.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_kv_cache_seq_cp.restype = None
    libllama.llama_kv_cache_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1)

def llama_kv_cache_seq_keep(ctx: Types.llama_context, seq_id: int) -> None:
    """Keep only the tokens of a sequence in the KV cache"""
    libllama.llama_kv_cache_seq_keep.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_kv_cache_seq_keep.restype = None
    libllama.llama_kv_cache_seq_keep(ctx, seq_id)

def llama_kv_cache_seq_add(ctx: Types.llama_context, seq_id: int, p0: int, p1: int, delta: int) -> None:
    """Add a relative position to tokens in a sequence in the KV cache"""
    libllama.llama_kv_cache_seq_add.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_kv_cache_seq_add.restype = None
    libllama.llama_kv_cache_seq_add(ctx, seq_id, p0, p1, delta)

def llama_kv_cache_seq_div(ctx: Types.llama_context, seq_id: int, p0: int, p1: int, d: int) -> None:
    """Divide the positions of tokens in a sequence in the KV cache by a factor"""
    libllama.llama_kv_cache_seq_div.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libllama.llama_kv_cache_seq_div.restype = None
    libllama.llama_kv_cache_seq_div(ctx, seq_id, p0, p1, d)

def llama_kv_cache_seq_pos_max(ctx: Types.llama_context, seq_id: int) -> int:
    """Get the maximum position of a sequence in the KV cache"""
    libllama.llama_kv_cache_seq_pos_max.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_kv_cache_seq_pos_max.restype = ctypes.c_int
    return libllama.llama_kv_cache_seq_pos_max(ctx, seq_id)

def llama_kv_cache_defrag(ctx: Types.llama_context) -> None:
    """Defragment the KV cache"""
    libllama.llama_kv_cache_defrag.argtypes = [ctypes.c_void_p]
    libllama.llama_kv_cache_defrag.restype = None
    libllama.llama_kv_cache_defrag(ctx)

def llama_kv_cache_update(ctx: Types.llama_context) -> None:
    """Apply KV cache updates"""
    libllama.llama_kv_cache_update.argtypes = [ctypes.c_void_p]
    libllama.llama_kv_cache_update.restype = None
    libllama.llama_kv_cache_update(ctx)

def llama_kv_cache_can_shift(ctx: Types.llama_context) -> bool:
    """Check if the context supports KV cache shifting"""
    libllama.llama_kv_cache_can_shift.argtypes = [ctypes.c_void_p]
    libllama.llama_kv_cache_can_shift.restype = ctypes.c_bool
    return libllama.llama_kv_cache_can_shift(ctx)

#
# State management
#

def llama_state_get_size(ctx: Types.llama_context) -> int:
    """Get the size of the state in bytes"""
    libllama.llama_state_get_size.argtypes = [ctypes.c_void_p]
    libllama.llama_state_get_size.restype = ctypes.c_int
    return libllama.llama_state_get_size(ctx)

def llama_get_state_size(*args):
    """
    DEPRECATED
    
    Use llama_state_get_size instead
    """
    raise Deprecated(
        f"function llama_get_state_size is marked as deprecated. use "
        f"llama_state_get_size instead."
    )

def llama_state_get_data(ctx: Types.llama_context, dst: ptr, size: int) -> int:
    """Copy the state to a destination address"""
    libllama.llama_state_get_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_state_get_data.restype = ctypes.c_int
    return libllama.llama_state_get_data(ctx, dst, size)

def llama_copy_state_data(*args):
    """
    DEPRECATED
    
    Use llama_state_get_data instead
    """
    raise Deprecated(
        f"function llama_copy_state_data is marked as deprecated. use "
        f"llama_state_get_data instead."
    )

def llama_state_set_data(ctx: Types.llama_context, src: ptr, size: int) -> int:
    """Set the state from a source address"""
    libllama.llama_state_set_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_state_set_data.restype = ctypes.c_int
    return libllama.llama_state_set_data(ctx, src, size)

def llama_set_state_data(*args):
    """
    DEPRECATED
    
    Use llama_state_set_dat instead
    """
    raise Deprecated(
        f"function llama_set_state_data is marked as deprecated. use "
        f"llama_state_set_dat instead."
    )

def llama_state_load_file(ctx: Types.llama_context, path_session: str, tokens_out: ptr, n_token_capacity: int, n_token_count_out: ptr) -> bool:
    """Load a state from a file"""
    libllama.llama_state_load_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    libllama.llama_state_load_file.restype = ctypes.c_bool
    return libllama.llama_state_load_file(ctx, path_session.encode('utf-8'), tokens_out, n_token_capacity, n_token_count_out)

def llama_load_session_file(*args):
    """
    DEPRECATED
    
    Use llama_state_load_file instead
    """
    raise Deprecated(
        f"function llama_load_session_file is marked as deprecated. use "
        f"llama_state_load_file instead."
    )

def llama_state_save_file(ctx: Types.llama_context, path_session: str, tokens: ptr, n_token_count: int) -> bool:
    """Save a state to a file"""
    libllama.llama_state_save_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_state_save_file.restype = ctypes.c_bool
    return libllama.llama_state_save_file(ctx, path_session.encode('utf-8'), tokens, n_token_count)

def llama_save_session_file(*args):
    """
    DEPRECATED
    
    Use llama_state_save_file instead
    """
    raise Deprecated(
        f"function llama_save_session_file is marked as deprecated. use "
        f"llama_state_save_file instead."
    )

def llama_state_seq_get_size(ctx: Types.llama_context, llama_seq_id: int) -> int:
    """Get the exact size needed to copy the KV cache of a single sequence"""
    libllama.llama_state_seq_get_size.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    libllama.llama_state_seq_get_size.restype = ctypes.c_ulong
    return libllama.llama_state_seq_get_size(ctx, llama_seq_id)

def llama_state_seq_get_data(ctx: Types.llama_context, dst: int, size: int, seq_id: int) -> int:
    """Copy the KV cache of a single sequence into the specified buffer"""
    libllama.llama_state_seq_get_data.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_ulong, ctypes.c_int32]
    libllama.llama_state_seq_get_data.restype = ctypes.c_ulong
    return libllama.llama_state_seq_get_data(ctx, dst, size, seq_id)

def llama_state_seq_set_data(ctx: Types.llama_context, src: int, size: int, dest_seq_id: int) -> int:
    """
    Copy the sequence data (originally copied with `llama_state_seq_get_data`)
    into the specified sequence
    
    Returns:
    - Positive: Ok
    - Zero: Failed to load
    """
    libllama.llama_state_seq_set_data.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_ulong, ctypes.c_int32]
    libllama.llama_state_seq_set_data.restype = ctypes.c_ulong
    return libllama.llama_state_seq_set_data(ctx, src, size, dest_seq_id)

def llama_state_seq_save_file(ctx: Types.llama_context, filepath: str, seq_id: int, tokens: ptr, n_token_count: int) -> int:
    libllama.llama_state_seq_save_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_ulong]
    libllama.llama_state_seq_save_file.restype = ctypes.c_ulong
    return libllama.llama_state_seq_save_file(ctx, filepath.encode('utf-8'), seq_id, tokens, n_token_count)

def llama_state_seq_load_file(ctx: Types.llama_context, filepath: str, dest_seq_id: int, tokens_out: ptr, n_token_capacity: int, n_token_count_out: ptr) -> int:
    libllama.llama_state_seq_load_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    libllama.llama_state_seq_load_file.restype = ctypes.c_ulong
    return libllama.llama_state_seq_load_file(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)

#
# Batch
#

def llama_batch_get_one(tokens: ptr, n_tokens: int) -> Types.llama_batch:
    """
    AVOID USING

    Return batch for single sequence of tokens
    """
    libllama.llama_batch_get_one.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    libllama.llama_batch_get_one.restype = ctypes.c_void_p
    return libllama.llama_batch_get_one(tokens, n_tokens)

def llama_batch_init(n_tokens: int, embd: int, n_seq_max: int) -> Types.llama_batch:
    """Allocates a batch of tokens"""
    libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    libllama.llama_batch_init.restype = ctypes.c_void_p
    return libllama.llama_batch_init(n_tokens, embd, n_seq_max)

def llama_batch_free(batch: Types.llama_batch) -> None:
    """Frees a batch of tokens"""
    libllama.llama_batch_free.argtypes = [ctypes.c_void_p]
    libllama.llama_batch_free.restype = None
    libllama.llama_batch_free(batch)

#
# Encode / decode
#

def llama_encode(ctx: Types.llama_context, batch: Types.llama_batch) -> int:
    """Process a batch of tokens with the encoder part of the encoder-decoder model"""
    libllama.llama_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_encode.restype = ctypes.c_int
    return libllama.llama_encode(ctx, batch)

def llama_decode(ctx: Types.llama_context, batch: Types.llama_batch) -> int:
    """Process a batch of tokens with the decoder part of the encoder-decoder model"""
    libllama.llama_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_decode.restype = ctypes.c_int
    return libllama.llama_decode(ctx, batch)

def llama_set_n_threads(ctx: Types.llama_context, n_threads: int, n_threads_batch: int) -> None:
    """Set the number of threads used for decoding"""
    libllama.llama_set_n_threads.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    libllama.llama_set_n_threads.restype = None
    libllama.llama_set_n_threads(ctx, n_threads, n_threads_batch)

def llama_n_threads(ctx: Types.llama_context) -> int:
    """Get the number of threads used for generation of a single token"""
    libllama.llama_n_threads.argtypes = [ctypes.c_void_p]
    libllama.llama_n_threads.restype = ctypes.c_int
    return libllama.llama_n_threads(ctx)

def llama_n_threads_batch(ctx: Types.llama_context) -> int:
    """Get the number of threads used for prompt and batch processing"""
    libllama.llama_n_threads_batch.argtypes = [ctypes.c_void_p]
    libllama.llama_n_threads_batch.restype = ctypes.c_int
    return libllama.llama_n_threads_batch(ctx)

def llama_set_embeddings(ctx: Types.llama_context, embeddings: bool) -> None:
    """Set whether to use embeddings mode or not"""
    libllama.llama_set_embeddings.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    libllama.llama_set_embeddings.restype = None
    libllama.llama_set_embeddings(ctx, embeddings)

def llama_set_causal_attn(ctx: Types.llama_context, causal_attn: bool) -> None:
    """Set whether to use causal attention or not"""
    libllama.llama_set_causal_attn.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    libllama.llama_set_causal_attn.restype = None
    libllama.llama_set_causal_attn(ctx, causal_attn)

def llama_set_abort_callback(ctx: Types.llama_context, abort_callback: ptr, abort_callback_data: ptr) -> None:
    """Set an abort callback"""
    libllama.llama_set_abort_callback.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_set_abort_callback.restype = None
    libllama.llama_set_abort_callback(ctx, abort_callback, abort_callback_data)

def llama_synchronize(ctx: Types.llama_context) -> None:
    """
    Wait until all computations are finished

    Not necessary to call explicitly in most cases
    """
    libllama.llama_synchronize.argtypes = [ctypes.c_void_p]
    libllama.llama_synchronize.restype = None
    libllama.llama_synchronize(ctx)

def llama_get_logits(ctx: Types.llama_context) -> ptr:
    """
    Get the token logits obtained from the last call to llama_decode()
    
    Rows: number of tokens for which llama_batch.logits[i] != 0
    Cols: n_vocab
    """
    libllama.llama_get_logits.argtypes = [ctypes.c_void_p]
    libllama.llama_get_logits.restype = ctypes.c_void_p
    return libllama.llama_get_logits(ctx)

def llama_get_logits_ith(ctx: Types.llama_context, i: int) -> ptr:
    """Get the logits for the ith token"""
    libllama.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_get_logits_ith.restype = ctypes.c_void_p
    return libllama.llama_get_logits_ith(ctx, i)

def llama_get_embeddings(ctx: Types.llama_context) -> ptr:
    """Get all output token embeddings"""
    libllama.llama_get_embeddings.argtypes = [ctypes.c_void_p]
    libllama.llama_get_embeddings.restype = ctypes.c_void_p
    return libllama.llama_get_embeddings(ctx)

def llama_get_embeddings_ith(ctx: Types.llama_context, i: int) -> ptr:
    """Get the embeddings for the ith token"""
    libllama.llama_get_embeddings_ith.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_get_embeddings_ith.restype = ctypes.c_void_p
    return libllama.llama_get_embeddings_ith(ctx, i)

def llama_get_embeddings_seq(ctx: ptr, seq_id: int) -> ptr:
    """Get the embeddings for a sequence id"""
    libllama.llama_get_embeddings_seq.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_get_embeddings_seq.restype = ctypes.c_void_p
    return libllama.llama_get_embeddings_seq(ctx, seq_id)

#
# Vocab
#

def llama_token_get_text(model: Types.llama_model, token: int) -> str:
    """Get the text representation of a token"""
    libllama.llama_token_get_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_token_get_text.restype = ctypes.c_char_p
    return libllama.llama_token_get_text(model, token).decode('utf-8')

def llama_token_get_score(model: Types.llama_model, token: int) -> float:
    """Get the score of a token"""
    libllama.llama_token_get_score.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_token_get_score.restype = ctypes.c_float
    return libllama.llama_token_get_score(model, token)

def llama_token_get_attr(model: Types.llama_model, token: int) -> int:
    """Get the attributes of a token"""
    libllama.llama_token_get_attr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_token_get_attr.restype = ctypes.c_int
    return libllama.llama_token_get_attr(model, token)

def llama_token_is_eog(model: Types.llama_model, token: int) -> bool:
    """Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)"""
    libllama.llama_token_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_token_is_eog.restype = ctypes.c_bool
    return libllama.llama_token_is_eog(model, token)

def llama_token_is_control(model: Types.llama_model, token: int) -> bool:
    """Identify if Token Id is a control token or a render-able token"""
    libllama.llama_token_is_control.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_token_is_control.restype = ctypes.c_bool
    return libllama.llama_token_is_control(model, token)

#
# Special tokens
#

def llama_token_bos(model: Types.llama_model) -> int:
    """Get the BOS token"""
    libllama.llama_token_bos.argtypes = [ctypes.c_void_p]
    libllama.llama_token_bos.restype = ctypes.c_int
    return libllama.llama_token_bos(model)

def llama_token_eos(model: Types.llama_model) -> int:
    """Get the EOS token"""
    libllama.llama_token_eos.argtypes = [ctypes.c_void_p]
    libllama.llama_token_eos.restype = ctypes.c_int
    return libllama.llama_token_eos(model)

def llama_token_eot(model: Types.llama_model) -> int:
    """Get the end-of-turn token"""
    libllama.llama_token_eot.argtypes = [ctypes.c_void_p]
    libllama.llama_token_eot.restype = ctypes.c_int
    return libllama.llama_token_eot(model)

def llama_token_cls(model: Types.llama_model) -> int:
    """Get the classification token"""
    libllama.llama_token_cls.argtypes = [ctypes.c_void_p]
    libllama.llama_token_cls.restype = ctypes.c_int
    return libllama.llama_token_cls(model)

def llama_token_sep(model: Types.llama_model) -> int:
    """Get the sentence separator token"""
    libllama.llama_token_sep.argtypes = [ctypes.c_void_p]
    libllama.llama_token_sep.restype = ctypes.c_int
    return libllama.llama_token_sep(model)

def llama_token_nl(model: Types.llama_model) -> int:
    """Get the next-line token"""
    libllama.llama_token_nl.argtypes = [ctypes.c_void_p]
    libllama.llama_token_nl.restype = ctypes.c_int
    return libllama.llama_token_nl(model)

def llama_token_pad(model: Types.llama_model) -> int:
    """Get the padding token"""
    libllama.llama_token_pad.argtypes = [ctypes.c_void_p]
    libllama.llama_token_pad.restype = ctypes.c_int
    return libllama.llama_token_pad(model)

def llama_add_bos_token(model: Types.llama_model) -> bool:
    """Whether BOS token should be added to tokenizations"""
    libllama.llama_add_bos_token.argtypes = [ctypes.c_void_p]
    libllama.llama_add_bos_token.restype = ctypes.c_bool
    return libllama.llama_add_bos_token(model)

def llama_add_eos_token(model: Types.llama_model) -> bool:
    """Whether EOS token should be added to tokenizations"""
    libllama.llama_add_eos_token.argtypes = [ctypes.c_void_p]
    libllama.llama_add_eos_token.restype = ctypes.c_bool
    return libllama.llama_add_eos_token(model)

def llama_token_prefix(*args):
    """
    DEPRECATED

    Use llama_token_fim_pre instead
    """
    raise Deprecated(
        f"function llama_token_prefix is marked as deprecated. use "
        f"llama_token_fim_pre instead."
    )

def llama_token_middle(*args):
    """
    DEPRECATED

    Use llama_token_fim_mid instead
    """
    raise Deprecated(
        f"function llama_token_middle is marked as deprecated. use "
        f"llama_token_fim_mid instead."
    )

def llama_token_suffix(*args):
    """
    DEPRECATED

    Use llama_token_fim_suf instead
    """
    raise Deprecated(
        f"function llama_token_suffix is marked as deprecated. use "
        f"llama_token_fim_suf instead."
    )

def llama_token_fim_pre(model: Types.llama_model) -> int:
    """Infill prefix token"""
    libllama.llama_token_fim_pre.argtypes = [ctypes.c_void_p]
    libllama.llama_token_fim_pre.restype = ctypes.c_int32
    return libllama.llama_token_fim_pre(model)

def llama_token_fim_suf(model: Types.llama_model) -> int:
    """Infill suffix token"""
    libllama.llama_token_fim_suf.argtypes = [ctypes.c_void_p]
    libllama.llama_token_fim_suf.restype = ctypes.c_int32
    return libllama.llama_token_fim_suf(model)

def llama_token_fim_mid(model: Types.llama_model) -> int:
    """Infill middle token"""
    libllama.llama_token_fim_mid.argtypes = [ctypes.c_void_p]
    libllama.llama_token_fim_mid.restype = ctypes.c_int32
    return libllama.llama_token_fim_mid(model)

def llama_token_fim_pad(model: Types.llama_model) -> int:
    """Infill pad token"""
    libllama.llama_token_fim_pad.argtypes = [ctypes.c_void_p]
    libllama.llama_token_fim_pad.restype = ctypes.c_int32
    return libllama.llama_token_fim_pad(model)

def llama_token_fim_rep(model: Types.llama_model) -> int:
    """Infill repo token"""
    libllama.llama_token_fim_rep.argtypes = [ctypes.c_void_p]
    libllama.llama_token_fim_rep.restype = ctypes.c_int32
    return libllama.llama_token_fim_rep(model)

def llama_token_fim_sep(model: Types.llama_model) -> int:
    """Infill sep token"""
    libllama.llama_token_fim_sep.argtypes = [ctypes.c_void_p]
    libllama.llama_token_fim_sep.restype = ctypes.c_int32
    return libllama.llama_token_fim_sep(model)

#
# Tokenization
#

def llama_tokenize(model: Types.llama_model, text: str, text_len: int, tokens: ptr, n_tokens_max: int, add_special: bool, parse_special: bool) -> int:
    """Tokenize a text into tokens"""
    libllama.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
    libllama.llama_tokenize.restype = ctypes.c_int
    return libllama.llama_tokenize(model, text.encode('utf-8'), text_len, tokens, n_tokens_max, add_special, parse_special)

def llama_token_to_piece(model: Types.llama_model, token: int, buf: ptr, length: int, lstrip: int, special: bool) -> int:
    """Convert a token to a piece of text"""
    libllama.llama_token_to_piece.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
    libllama.llama_token_to_piece.restype = ctypes.c_int
    return libllama.llama_token_to_piece(model, token, buf, length, lstrip, special)

def llama_detokenize(model: Types.llama_model, tokens: ptr, n_tokens: int, text: ptr, text_len_max: int, remove_special: bool, unparse_special: bool) -> int:
    """Detokenize tokens into a text"""
    libllama.llama_detokenize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
    libllama.llama_detokenize.restype = ctypes.c_int
    return libllama.llama_detokenize(model, tokens, n_tokens, text, text_len_max, remove_special, unparse_special)

#
# Chat templating (not implemented)
#

def llama_chat_apply_template(*args):
    raise NotImplementedError

def llama_chat_builtin_templates(*args):
    raise NotImplementedError

#
# Sampling
#

def llama_sampler_name(smpl: Types.llama_sampler) -> str:
    """Get the name of a sampler"""
    libllama.llama_sampler_name.argtypes = [ctypes.c_void_p]
    libllama.llama_sampler_name.restype = ctypes.c_char_p
    return libllama.llama_sampler_name(smpl).decode('utf-8')

def llama_sampler_accept(smpl: Types.llama_sampler, token: int) -> None:
    """Accept a token sampled by a sampler"""
    libllama.llama_sampler_accept.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_sampler_accept.restype = None
    libllama.llama_sampler_accept(smpl, token)

def llama_sampler_apply(smpl: Types.llama_sampler, cur_p: Types.llama_token_data_array) -> None:
    """Apply a sampler to a token data array"""
    libllama.llama_sampler_apply.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_sampler_apply.restype = None
    libllama.llama_sampler_apply(smpl, cur_p)

def llama_sampler_reset(smpl: Types.llama_sampler) -> None:
    """Reset a sampler"""
    libllama.llama_sampler_reset.argtypes = [ctypes.c_void_p]
    libllama.llama_sampler_reset.restype = None
    libllama.llama_sampler_reset(smpl)

def llama_sampler_clone(smpl: Types.llama_sampler) -> Types.llama_sampler:
    """Clone a sampler"""
    libllama.llama_sampler_clone.argtypes = [ctypes.c_void_p]
    libllama.llama_sampler_clone.restype = ctypes.c_void_p
    return libllama.llama_sampler_clone(smpl)

def llama_sampler_free(smpl: Types.llama_sampler) -> None:
    """
    Free a sampler
    
    NOTE: Do not free if the sampler has been added to a llama_sampler_chain
    (via llama_sampler_chain_add)
    """
    libllama.llama_sampler_free.argtypes = [ctypes.c_void_p]
    libllama.llama_sampler_free.restype = None
    libllama.llama_sampler_free(smpl)

#
# Sampler chains
#

def llama_sampler_chain_init(params: Types.llama_sampler_chain_params) -> Types.llama_sampler:
    """Initialize a sampler chain"""
    libllama.llama_sampler_chain_init.argtypes = [ctypes.c_void_p]
    libllama.llama_sampler_chain_init.restype = ctypes.c_void_p
    return libllama.llama_sampler_chain_init(params)

def llama_sampler_chain_add(chain: Types.llama_sampler, smpl: Types.llama_sampler) -> None:
    """
    Add a sampler to a sampler chain
    
    Takes ownership of the sampler object and will free it when llama_sampler_free is called
    """
    libllama.llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_sampler_chain_add.restype = None
    libllama.llama_sampler_chain_add(chain, smpl)

def llama_sampler_chain_get(chain: Types.llama_sampler, i: int) -> Types.llama_sampler:
    """Get a sampler from a sampler chain"""
    libllama.llama_sampler_chain_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_sampler_chain_get.restype = ctypes.c_void_p
    return libllama.llama_sampler_chain_get(chain, i)

def llama_sampler_chain_n(chain: Types.llama_sampler) -> int:
    """Get the number of samplers in a sampler chain"""
    libllama.llama_sampler_chain_n.argtypes = [ctypes.c_void_p]
    libllama.llama_sampler_chain_n.restype = ctypes.c_int
    return libllama.llama_sampler_chain_n(chain)

def llama_sampler_chain_remove(chain: Types.llama_sampler, i: int) -> Types.llama_sampler:
    """
    Remove a sampler from a sampler chain
    
    after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    """
    libllama.llama_sampler_chain_remove.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libllama.llama_sampler_chain_remove.restype = ctypes.c_void_p
    return libllama.llama_sampler_chain_remove(chain, i)

#
# Samplers
#

def llama_sampler_init_greedy() -> Types.llama_sampler:
    """Initialize a greedy sampler"""
    libllama.llama_sampler_init_greedy.argtypes = []
    libllama.llama_sampler_init_greedy.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_greedy()

def llama_sampler_init_dist(seed: int) -> Types.llama_sampler:
    """Initialize a distribution sampler"""
    libllama.llama_sampler_init_dist.argtypes = [ctypes.c_int]
    libllama.llama_sampler_init_dist.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_dist(seed)

def llama_sampler_init_softmax():
    """
    DEPRECATED

    Initialize a softmax sampler
    """
    raise Deprecated(
        f"function llama_sampler_init_softmax is marked as deprecated. "
        f"do not use."
    )

def llama_sampler_init_top_k(k: int) -> Types.llama_sampler:
    """Initialize a top-K sampler"""
    libllama.llama_sampler_init_top_k.argtypes = [ctypes.c_int]
    libllama.llama_sampler_init_top_k.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_top_k(k)

def llama_sampler_init_top_p(p: float, min_keep: int) -> Types.llama_sampler:
    """Initialize a top-p sampler"""
    libllama.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_int]
    libllama.llama_sampler_init_top_p.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_top_p(p, min_keep)

def llama_sampler_init_min_p(p: float, min_keep: int) -> Types.llama_sampler:
    """Initialize a min-p sampler"""
    libllama.llama_sampler_init_min_p.argtypes = [ctypes.c_float, ctypes.c_int]
    libllama.llama_sampler_init_min_p.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_min_p(p, min_keep)

def llama_sampler_init_typical(p: float, min_keep: int) -> Types.llama_sampler:
    """Initialize a locally typical sampler"""
    libllama.llama_sampler_init_typical.argtypes = [ctypes.c_float, ctypes.c_int]
    libllama.llama_sampler_init_typical.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_typical(p, min_keep)

def llama_sampler_init_temp(t: float) -> Types.llama_sampler:
    """
    Initialize a temperature sampler
    
    When `t` <= 0.0, the maximum logit is kept at it's original value, the rest are set to -inf
    """
    libllama.llama_sampler_init_temp.argtypes = [ctypes.c_float]
    libllama.llama_sampler_init_temp.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_temp(t)

def llama_sampler_init_temp_ext(t: float, delta: float, exponent: float) -> Types.llama_sampler:
    """Initialize an dynamic temperature / entropy sampler"""
    libllama.llama_sampler_init_temp_ext.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    libllama.llama_sampler_init_temp_ext.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_temp_ext(t, delta, exponent)

def llama_sampler_init_xtc(p: float, t: float, min_keep: int, seed: int) -> Types.llama_sampler:
    """Initialize an XTC sampler"""
    libllama.llama_sampler_init_xtc.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int]
    libllama.llama_sampler_init_xtc.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_xtc(p, t, min_keep, seed)

def llama_sampler_init_mirostat(seed: int, tau: float, eta: float) -> Types.llama_sampler:
    """Initialize a Mirostat sampler"""
    libllama.llama_sampler_init_mirostat.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
    libllama.llama_sampler_init_mirostat.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_mirostat(seed, tau, eta)

def llama_sampler_init_mirostat_v2(seed: int, tau: float, eta: float) -> Types.llama_sampler:
    """Initialize a Mirostat v2 sampler"""
    libllama.llama_sampler_init_mirostat_v2.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
    libllama.llama_sampler_init_mirostat_v2.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_mirostat_v2(seed, tau, eta)

def llama_sampler_init_grammar(model: Types.llama_model, grammar_str: str, grammar_root: str) -> Types.llama_sampler:
    """Initialize a grammar sampler"""
    libllama.llama_sampler_init_grammar.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    libllama.llama_sampler_init_grammar.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_grammar(model, grammar_str.encode('utf-8'), grammar_root.encode('utf-8'))

def llama_sampler_init_penalties(n_vocab: int, special_eos_id: int, linefeed_id: int, penalty_last_n: int, penalty_repeat: float, penalty_freq: float, penalty_present: float, penalize_nl: bool, ignore_eos: bool) -> Types.llama_sampler:
    """Initialize a penalties sampler"""
    libllama.llama_sampler_init_penalties.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool, ctypes.c_bool]
    libllama.llama_sampler_init_penalties.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_penalties(n_vocab, special_eos_id, linefeed_id, penalty_last_n, penalty_repeat, penalty_freq, penalty_present, penalize_nl, ignore_eos)

def llama_sampler_init_dry(model: Types.llama_model, dry_multiplier: float, dry_base: float, dry_allowed_length: int, dry_penalty_last_n: int, seq_breakers: ptr, num_breakers: int) -> Types.llama_sampler:
    """Initialize a DRY sampler"""
    libllama.llama_sampler_init_dry.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_sampler_init_dry.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_dry(model, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers)

def llama_sampler_init_logit_bias(n_vocab: int, n_logit_bias: int, logit_bias: ptr) -> Types.llama_sampler:
    """Initialize a logit bias sampler"""
    libllama.llama_sampler_init_logit_bias.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libllama.llama_sampler_init_logit_bias.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias)

def llama_sampler_init_infill(model: Types.llama_model) -> Types.llama_sampler:
    """
    Initialize an infill sampler
    
    This sampler is meant to be used for fill-in-the-middle infilling. It's supposed to be used after top_k + top_p sampling"""
    libllama.llama_sampler_init_infill.argtypes = [ctypes.c_void_p]
    libllama.llama_sampler_init_infill.restype = ctypes.c_void_p
    return libllama.llama_sampler_init_infill(model)

def llama_sampler_get_seed(smpl: Types.llama_sampler) -> int:
    """
    Get the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    """
    libllama.llama_sampler_get_seed.argtypes = [ctypes.c_void_p]
    libllama.llama_sampler_get_seed.restype = ctypes.c_int
    return libllama.llama_sampler_get_seed(smpl)

def llama_sampler_sample(smpl: Types.llama_sampler, ctx: Types.llama_context, idx: int) -> int:
    """
    Sample and accept a token from the idx-th output of the last evaluation
    """
    libllama.llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    libllama.llama_sampler_sample.restype = ctypes.c_int
    return libllama.llama_sampler_sample(smpl, ctx, idx)

#
# Model split
#

# llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
def llama_split_path(split_path: ptr, maxlen: int, path_prefix: str, split_no: int, split_count: int) -> int:
    """Build a split GGUF final path for a chunk"""
    libllama.llama_split_path.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    libllama.llama_split_path.restype = ctypes.c_int
    return libllama.llama_split_path(split_path, maxlen, path_prefix.encode('utf-8'), split_no, split_count)

# llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
def llama_split_prefix(split_prefix: ptr, maxlen: int, split_path: str, split_no: int, split_count: int) -> int:
    """Extract the path prefix from a split path"""
    libllama.llama_split_prefix.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    libllama.llama_split_prefix.restype = ctypes.c_int
    return libllama.llama_split_prefix(split_prefix, maxlen, split_path.encode('utf-8'), split_no, split_count)

#
# Print system info
#

def llama_print_system_info() -> None:
    """Print system information"""
    libllama.llama_print_system_info.argtypes = []
    libllama.llama_print_system_info.restype = ctypes.c_char_p
    libllama.llama_print_system_info()

#
# Log callback
#

def llama_log_set(log_callback: ptr, user_data: ptr) -> None:
    """Set a callback for logging events"""
    libllama.llama_log_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libllama.llama_log_set.restype = None
    libllama.llama_log_set(log_callback, user_data)

#
# Performance utils
#

# NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.

def llama_perf_context(ctx: Types.llama_context) -> Types.llama_perf_context_data:
    """Get performance data for a context"""
    libllama.llama_perf_context.argtypes = [ctypes.c_void_p]
    libllama.llama_perf_context.restype = ctypes.c_void_p
    return libllama.llama_perf_context(ctx)

def llama_perf_context_print(ctx: Types.llama_context) -> None:
    """Print performance data for a context"""
    libllama.llama_perf_context_print.argtypes = [ctypes.c_void_p]
    libllama.llama_perf_context_print.restype = None
    libllama.llama_perf_context_print(ctx)

def llama_perf_context_reset(ctx: Types.llama_context) -> None:
    """Reset performance data for a context"""
    libllama.llama_perf_context_reset.argtypes = [ctypes.c_void_p]
    libllama.llama_perf_context_reset.restype = None
    libllama.llama_perf_context_reset(ctx)

# NOTE: the following work only with samplers constructed via llama_sampler_chain_init

def llama_perf_sampler(smpl: Types.llama_sampler) -> Types.llama_perf_sampler_data:
    """Get performance data for a sampler"""
    libllama.llama_perf_sampler.argtypes = [ctypes.c_void_p]
    libllama.llama_perf_sampler.restype = ctypes.c_void_p
    return libllama.llama_perf_sampler(smpl)

def llama_perf_sampler_print(smpl: Types.llama_sampler) -> None:
    """Print performance data for a sampler"""
    libllama.llama_perf_sampler_print.argtypes = [ctypes.c_void_p]
    libllama.llama_perf_sampler_print.restype = None
    libllama.llama_perf_sampler_print(smpl)

def llama_perf_sampler_reset(smpl: Types.llama_sampler) -> None:
    """Reset performance data for a sampler"""
    libllama.llama_perf_sampler_reset.argtypes = [ctypes.c_void_p]
    libllama.llama_perf_sampler_reset.restype = None
    libllama.llama_perf_sampler_reset(smpl)
