# llama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from _version import __version__

import os
import sys
import time
import struct
import ctypes

import numpy    as np
import libllama as lib

from utils    import (
    print_info, print_warning, print_error, print_stopwatch, null_ptr_check,
    softmax
)
from libllama import _internals, GGUFValueType
from typing   import Optional, Iterable
from io       import BufferedReader
from sampling import SamplerParams

_SUPPORTED_KV_TYPES = [
    lib.GGMLType.GGML_TYPE_F32,   # lib only supports static types, not
    lib.GGMLType.GGML_TYPE_F16,   # k-types
    lib.GGMLType.GGML_TYPE_Q8_1,
    lib.GGMLType.GGML_TYPE_Q8_0,  # BF16 is also sometimes supported, but not
    lib.GGMLType.GGML_TYPE_Q5_1,  # always, and offers no benefit compared
    lib.GGMLType.GGML_TYPE_Q5_0,  # to F16, so is not included here
    lib.GGMLType.GGML_TYPE_Q4_1,
    lib.GGMLType.GGML_TYPE_Q4_0
]

_DEFAULT_KV_TYPE = lib.GGMLType.GGML_TYPE_F16

_cpu_count = None

def _init_backend_if_needed() -> None:

    # if already initialized, no need to do anything
    if lib._BACKEND_INIT is True:
        return
    
    print_info(
        f'easy_llama package version: {__version__}'
    )
    
    global _cpu_count
    _cpu_count = int(os.cpu_count())
    
    # most cases
    if sys.byteorder == 'little':
        print_info(
            "host is little-endian"
        )
    # rare
    elif sys.byteorder == 'big':
        print_warning(
            "host is big-endian, please ensure your GGUF file is also "
            "big-endian"
        )
    # extremely rare
    else:
        print_error(
            f"unexpected value for sys.byteorder: {sys.byteorder!r}; "
            f"expected 'little' for little-endian host or 'big' for "
            f"big-endian host"
        )
    
    # actually load the backend
    lib.llama_backend_init() # this sets libllama._BACKEND_INIT to True

# NOTE: the optimal n_threads value (for text generation) is equal
#       to the number of physical cores (for homogenous CPUs) or
#       to the number of performance cores (for heterogenous CPUs)
#
#       the optimal n_threads_batch value (for prompt processing) is equal
#       to the total number of logical cores, regardless of
#       their type

def _get_optimal_n_threads() -> int:
    global _cpu_count
    return max(_cpu_count//2, 1)

def _get_optimal_n_threads_batch() -> int:
    global _cpu_count
    return _cpu_count

# def _get_random_seed() -> int:
#     # uint32_t
#     return int.from_bytes(
#         bytes=os.urandom(4),
#         byteorder=sys.byteorder,
#         signed=False
#     )

def _calculate_rope_freq_base(
        n_ctx_train: int,
        n_ctx_load: int,
        rope_freq_base_train: Optional[float]
    ) -> float:
    """
    Returns the rope_freq_base value at which a model should be loaded
    """

    # n_ctx does not exceed n_ctx_train - simply return native value

    if n_ctx_load <= n_ctx_train:
        if rope_freq_base_train is None:
            return 0.0
        else:
            return rope_freq_base_train
    
    # n_ctx exceeds n_ctx_train, but native value is unknown, so automatic
    # adjustment cannot be applied - show error and return 0.0

    if rope_freq_base_train in [None, 0.0]:
        print_error(
            f'n_ctx value {n_ctx_load} > n_ctx_train value {n_ctx_train}, and '
            f'automatic rope_freq_base adjustment is not supported for this '
            f'model; model loading might fail, or the model might not work '
            f'correctly'
        )
        return 0.0
    
    # n_ctx exceeds n_ctx_train, and native value is known, so automatic
    # adjustment can be applied - show warning and return adjusted value

    # standard formula -- proportional increase
    adjusted_rope_freq = (n_ctx_load/n_ctx_train)*rope_freq_base_train
    # experimental formula -- slightly above proportional increase
    #adjusted_rope_freq = \
    #    ((n_ctx_load/n_ctx_train)**(2**(1/4)))*rope_freq_base_train
    
    print_warning(
        f"n_ctx value {n_ctx_load} exceeds n_ctx_train value "
        f"{n_ctx_train}; using adjusted rope_freq_base value "
        f"{adjusted_rope_freq}, native value is "
        f"{rope_freq_base_train}; model will function with "
        f"potentially degraded output quality"
    )

    return adjusted_rope_freq

def _round_n_ctx(n_ctx: int, n_ctx_train: int) -> int:

    if n_ctx % 512 == 0:
        return n_ctx
    else:
        rounded = (n_ctx + 511) // 512 * 512
        if (rounded > n_ctx_train) and (n_ctx <= n_ctx_train):
            # do not round beyond n_ctx_train if not already exceeded
            return n_ctx_train
        else:
            return rounded

class LlamaStopwatch:
    """Track elapsed time for prompt processing and text generation"""
    #
    # Q: why don't you use llama_perf_context?
    #
    # A: comments in llama.h state to only use that in llama.cpp examples,
    #    and to do your own performance measurements instead.
    #
    #    trying to use llama_perf_context leads to output with
    #    "0.00 ms per token" and "inf tokens per second"
    #
    def __init__(self):
        self.pp_start_time = None
        self.tg_start_time = None
        self.pp_elapsed_time = 0
        self.tg_elapsed_time = 0
        self.n_pp_tokens = 0
        self.n_tg_tokens = 0

    def start_pp(self):
        """Start prompt processing stopwatch"""
        self.pp_start_time = time.time_ns()

    def stop_pp(self):
        """Stop prompt processing stopwatch"""
        if self.pp_start_time is not None:
            self.pp_elapsed_time += time.time_ns() - self.pp_start_time
            self.pp_start_time = None

    def start_tg(self):
        """Start text generation stopwatch"""
        self.tg_start_time = time.time_ns()

    def stop_tg(self):
        """Stop text generation stopwatch"""
        if self.tg_start_time is not None:
            self.tg_elapsed_time += time.time_ns() - self.tg_start_time
            self.tg_start_time = None

    def get_elapsed_time_pp(self) -> int:
        """Total nanoseconds elapsed during prompt processing"""
        return self.pp_elapsed_time

    def get_elapsed_time_tg(self) -> int:
        """Total nanoseconds elapsed during text generation"""
        return self.tg_elapsed_time

    def increment_pp_tokens(self, n: int):
        self.n_pp_tokens += max(n, 0) # do not allow negative increment

    def increment_tg_tokens(self, n: int):
        self.n_tg_tokens += max(n, 0) # do not allow negative increment

    def reset(self):
        """Reset the stopwatch to its original state"""
        self.pp_start_time = None
        self.tg_start_time = None
        self.pp_elapsed_time = 0
        self.tg_elapsed_time = 0
        self.n_pp_tokens = 0
        self.n_tg_tokens = 0

    def print_stats(self):
        """
        Print performance statistics using current stopwatch state
        
        #### NOTE:
        The `n_tg_tokens` value will be equal to the number of calls to
        llama_decode which have a batch size of 1, which is technically not
        always equal to the number of tokens generated - it may be off by one.
        """

        print(f"\n", end='', file=sys.stderr, flush=True)

        if self.n_pp_tokens + self.n_tg_tokens == 0:
            print_stopwatch(
                f'print_stats was called but no tokens were processed or '
                f'generated'
            )
            return

        if self.n_pp_tokens > 0:
            pp_elapsed_ns = self.get_elapsed_time_pp()
            pp_elapsed_ms = pp_elapsed_ns / 1e6
            pp_elapsed_s = pp_elapsed_ns / 1e9
            pp_tps = self.n_pp_tokens / pp_elapsed_s
            print_stopwatch(
                f'prompt processing: {self.n_pp_tokens:>7} tokens in '
                f'{pp_elapsed_ms:>13.3f}ms '
                f'({pp_tps:>10.2f} tok/s)'
            )

        if self.n_tg_tokens > 0:
            tg_elapsed_ns = self.get_elapsed_time_tg()
            tg_elapsed_ms = tg_elapsed_ns / 1e6
            tg_elapsed_s = tg_elapsed_ns / 1e9
            tg_tps = self.n_tg_tokens / tg_elapsed_s
            print_stopwatch(
                f'  text generation: {self.n_tg_tokens:>7} tokens in '
                f'{tg_elapsed_ms:>13.3f}ms '
                f'({tg_tps:>10.2f} tok/s)'
            )

class QuickGGUFReader:
    # ref: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

    # the GGUF format versions that this class supports
    SUPPORTED_GGUF_VERSIONS = [2, 3]
    
    # arguments for struct.unpack() based on gguf value type
    value_packing = {
        GGUFValueType.UINT8   : "=B",
        GGUFValueType.INT8    : "=b",
        GGUFValueType.UINT16  : "=H",
        GGUFValueType.INT16   : "=h",
        GGUFValueType.UINT32  : "=I",
        GGUFValueType.INT32   : "=i",
        GGUFValueType.FLOAT32 : "=f",
        GGUFValueType.UINT64  : "=Q",
        GGUFValueType.INT64   : "=q",
        GGUFValueType.FLOAT64 : "=d",
        GGUFValueType.BOOL    : "?"
    }

    # length in bytes for each gguf value type
    value_lengths = {
        GGUFValueType.UINT8   : 1,
        GGUFValueType.INT8    : 1,
        GGUFValueType.UINT16  : 2,
        GGUFValueType.INT16   : 2,
        GGUFValueType.UINT32  : 4,
        GGUFValueType.INT32   : 4,
        GGUFValueType.FLOAT32 : 4,
        GGUFValueType.UINT64  : 8,
        GGUFValueType.INT64   : 8,
        GGUFValueType.FLOAT64 : 8,
        GGUFValueType.BOOL    : 1
    }

    @staticmethod
    def unpack(value_type: GGUFValueType, file: BufferedReader):
        return struct.unpack(
            QuickGGUFReader.value_packing.get(value_type),
            file.read(QuickGGUFReader.value_lengths.get(value_type))
        )[0]

    @staticmethod
    def get_single(
            value_type: GGUFValueType,
            file: BufferedReader
        ) -> str | int | float | bool:
        """Read a single value from an open file"""
        if value_type == GGUFValueType.STRING:
            string_length = QuickGGUFReader.unpack(
                GGUFValueType.UINT64, file=file
            )
            value = file.read(string_length)
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                print_warning(
                    f'UnicodeDecodeError was raised while reading a string '
                    f'from the GGUF metadata. the GGUF format specifies that '
                    f'all strings in file metadata should be valid UTF-8. the '
                    f'affected string will be left blank.'
                )
                value = ''
        else:
            value = QuickGGUFReader.unpack(value_type, file=file)
        return value
    
    @staticmethod
    def load_metadata(
            path_model: os.PathLike[str] | str
        ) -> dict[str, str | int | float | bool | list]:
        """
        Given a path to a GGUF file, peek at its header for metadata

        Return a dictionary where all keys are strings, and values can be
        strings, ints, floats, bools, or lists
        """

        metadata: dict[str, str | int | float | bool | list] = {}
        with open(path_model, "rb") as file:
            magic = file.read(4)

            if magic != lib.GGUF_MAGIC_BYTES:
                raise ValueError(
                    f"your model file is not a valid GGUF file "
                    f"(magic number mismatch, got {magic}, "
                    f"expected {lib.GGUF_MAGIC_BYTES})"
                )
            
            version = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)

            if version not in QuickGGUFReader.SUPPORTED_GGUF_VERSIONS:
                raise ValueError(
                    f"your model file reports GGUF version {version}, but "
                    f"only versions {QuickGGUFReader.SUPPORTED_GGUF_VERSIONS} "
                    f"are supported. re-convert your model or download a newer "
                    f"version"
                )
            
            tensor_count = QuickGGUFReader.unpack(
                GGUFValueType.UINT64, file=file
            )
            if version == 3:
                metadata_kv_count = QuickGGUFReader.unpack(
                    GGUFValueType.UINT64, file=file
                )
            elif version == 2:
                metadata_kv_count = QuickGGUFReader.unpack(
                    GGUFValueType.UINT32, file=file
                )
            for _ in range(metadata_kv_count):
                if version == 3:
                    key_length = QuickGGUFReader.unpack(
                        GGUFValueType.UINT64, file=file
                    )
                elif version == 2:
                    key_length = 0
                    while key_length == 0:
                        # seek until next key is found
                        key_length = QuickGGUFReader.unpack(
                            GGUFValueType.UINT32, file=file
                        )
                    file.read(4) # 4 byte offset for GGUFv2
                key = file.read(key_length)
                value_type = GGUFValueType(
                    QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                )
                if value_type == GGUFValueType.ARRAY:
                    array_value_type = GGUFValueType(
                        QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                    )
                    # array_length is the number of items in the array
                    if version == 3:
                        array_length = QuickGGUFReader.unpack(
                            GGUFValueType.UINT64, file=file
                        )
                    elif version == 2:
                        array_length = QuickGGUFReader.unpack(
                            GGUFValueType.UINT32, file=file
                        )
                        file.read(4) # 4 byte offset for GGUFv2
                    array = [
                        QuickGGUFReader.get_single(
                            array_value_type,
                            file
                        ) for _ in range(array_length)
                    ]
                    metadata[key.decode()] = array
                else:
                    value = QuickGGUFReader.get_single(
                        value_type,
                        file
                    )
                    metadata[key.decode()] = value

        return metadata

#
# Simple python wrappers
#

class _LlamaModel:

    def __init__(
        self,
        path_model: str,
        devices: Optional[lib.ptr] = None,
        n_gpu_layers: Optional[int] = None,
        split_mode: Optional[int] = None,
        main_gpu: Optional[int] = None,
        tensor_split: Optional[lib.ptr] = None,
        rpc_servers: Optional[str] = None,
        progress_callback: Optional[lib.ptr] = None,
        progress_callback_user_data: Optional[lib.ptr] = None,
        kv_overrides: Optional[lib.llama_model_kv_override] = None,
        vocab_only: Optional[bool] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        check_tensors: Optional[bool] = None
    ):
        _init_backend_if_needed()
        self.path_model = path_model
        self.params = lib.llama_model_default_params()
        null_ptr_check(self.params, "self.params", "_LlamaModel.__init__")
        if devices is not None:
            self.params.devices = (
                ctypes.c_void_p * (len(devices) + 1)
            )(*devices, None)
        if n_gpu_layers is not None:
            self.params.n_gpu_layers = (
                n_gpu_layers
            ) if n_gpu_layers >= 0 else lib.MAX_OFFLOAD_LAYERS
        if split_mode is not None:
            self.params.split_mode = split_mode
        if main_gpu is not None:
            self.params.main_gpu = main_gpu
        if tensor_split is not None:
            self.params.tensor_split = (
                ctypes.c_float * len(tensor_split)
            )(*tensor_split)
        if rpc_servers is not None:
            self.params.rpc_servers = rpc_servers.encode('utf-8')
        if progress_callback is not None:
            self.params.progress_callback = progress_callback
        if progress_callback_user_data is not None:
            self.params.progress_callback_user_data = progress_callback_user_data
        if kv_overrides is not None:
            self.params.kv_overrides = (
                lib.llama_model_kv_override * len(kv_overrides)
            )(*kv_overrides)
        if vocab_only is not None:
            self.params.vocab_only = vocab_only
        if use_mmap is not None:
            self.params.use_mmap = use_mmap
        if use_mlock is not None:
            self.params.use_mlock = use_mlock
        if check_tensors is not None:
            self.params.check_tensors = check_tensors

        # refuse to load files with incorrect extension
        if not path_model.lower().endswith('.gguf'):
            raise ValueError(
                f"_LlamaModel.__init__: the given path_model {path_model!r} "
                f"does not end in '.gguf'. easy-llama refuses to load from "
                f"files that do not have the correct file extension."
            )
        
        # load model
        self.model = lib.llama_load_model_from_file(path_model, self.params)
        null_ptr_check(self.model, "self.model", "_LlamaModel.__init__")
    
    def __del__(self):
        self.free()

    def free(self):
        if self.model is not None:
            lib.llama_free_model(self.model)
            self.model = None


class _LlamaCtx:

    def __init__(
        self,
        model: _LlamaModel,
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_ubatch: Optional[int] = None,
        n_seq_max: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        rope_scaling_type: Optional[int] = None,
        pooling_type: Optional[int] = None,
        attention_type: Optional[int] = None,
        rope_freq_base: Optional[float] = None,
        rope_freq_scale: Optional[float] = None,
        yarn_ext_factor: Optional[float] = None,
        yarn_attn_factor: Optional[float] = None,
        yarn_beta_fast: Optional[float] = None,
        yarn_beta_slow: Optional[float] = None,
        yarn_orig_ctx: Optional[int] = None,
        defrag_thold: Optional[float] = None,
        cb_eval: Optional[lib.ptr] = None,
        cb_eval_user_data: Optional[lib.ptr] = None,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        logits_all: Optional[bool] = None,
        embeddings: Optional[bool] = None,
        offload_kqv: Optional[bool] = None,
        flash_attn: Optional[bool] = None,
        no_perf: Optional[bool] = None,
        abort_callback: Optional[lib.ptr] = None,
        abort_callback_data: Optional[lib.ptr] = None
    ):
        _init_backend_if_needed()
        self.params = lib.llama_context_default_params()
        null_ptr_check(self.params, "self.params", "_LlamaCtx.__init__")
        if n_ctx is not None:
            self.params.n_ctx = n_ctx
        if n_batch is not None:
            self.params.n_batch = n_batch
        if n_ubatch is not None:
            self.params.n_ubatch = n_ubatch
        if n_seq_max is not None:
            if n_seq_max != 1:
                raise NotImplementedError(
                    f'n_seq_max value {n_seq_max} != 1; this is not yet supported'
                )
            self.params.n_seq_max = n_seq_max
        if n_threads is not None:
            self.params.n_threads = n_threads
        if n_threads_batch is not None:
            self.params.n_threads_batch = n_threads_batch
        if rope_scaling_type is not None:
            self.params.rope_scaling_type = rope_scaling_type
        if pooling_type is not None:
            self.params.pooling_type = pooling_type
        if attention_type is not None:
            self.params.attention_type =  attention_type
        if rope_freq_base is not None:
            self.params.rope_freq_base = rope_freq_base
        if rope_freq_scale is not None:
            self.params.rope_freq_scale = rope_freq_scale
        if yarn_ext_factor is not None:
            self.params.yarn_ext_factor = yarn_ext_factor
        if yarn_attn_factor is not None:
            self.params.yarn_attn_factor = yarn_attn_factor
        if yarn_beta_fast is not None:
            self.params.yarn_beta_fast = yarn_beta_fast
        if yarn_beta_slow is not None:
            self.params.yarn_beta_slow = yarn_beta_slow
        if yarn_orig_ctx is not None:
            self.params.yarn_orig_ctx = yarn_orig_ctx
        if defrag_thold is not None:
            self.params.defrag_thold = defrag_thold
        if cb_eval is not None:
            self.params.cb_eval = cb_eval
        if cb_eval_user_data is not None:
            self.params.cb_eval_user_data = cb_eval_user_data
        _k = _DEFAULT_KV_TYPE
        if type_k is not None:
            self.params.type_k = _k = type_k
        _v = _DEFAULT_KV_TYPE
        if type_v is not None:
            self.params.type_v = _v = type_v
        if _k != _v:
            print_warning(
                f'type_k value {_k} != type_v value {_v}; this is rarely '
                f'supported, program may fail'
            )
        if _k not in _SUPPORTED_KV_TYPES:
            print_warning(
                f'type_k value {_k} is unsupported; program may fail'
            )
        if _v not in _SUPPORTED_KV_TYPES:
            print_warning(
                f'type_v value {_v} is unsupported; program may fail'
            )
        if flash_attn and _v not in [
            lib.GGMLType.GGML_TYPE_F32, lib.GGMLType.GGML_TYPE_F16
        ]:
            print_warning(
                f'V cache quantization requires flash_attn; program may fail'
            )
        if logits_all is not None:
            self.params.logits_all = logits_all
        if embeddings is not None:
            self.params.embeddings = embeddings
        if offload_kqv is not None:
            self.params.offload_kqv = offload_kqv
        if flash_attn is not None:
            self.params.flash_attn = flash_attn
        if no_perf is not None:
            self.params.no_perf = no_perf
        if abort_callback is not None:
            self.params.abort_callback = abort_callback
        if abort_callback_data is not None:
            self.params.abort_callback_data = abort_callback_data
        
        null_ptr_check(model.model, "model.model", "_LlamaCtx.__init__")
        self.ctx = lib.llama_new_context_with_model(
            model.model, self.params
        )
        null_ptr_check(self.ctx, "self.ctx", "_LlamaCtx.__init__")
    
    def __del__(self):
        self.free()

    def free(self):
        if self.ctx is not None:
            lib.llama_free(self.ctx)
            self.ctx = None

#
# Llama
#

class Llama:
    """
    Simplified interface for general-purpose Llama model usage

    The `easy_llama.Llama` class provides a high-level Python interface to
    a llama_model and its associated llama_context.

    Example usage:
    >>> import easy_llama as ez
    >>> Llama = ez.Llama('/path/to/model.gguf', n_ctx=8192)
    >>> in_txt = "The apple doesn't fall far from"
    >>> in_toks = Llama.tokenize(in_txt.encode(), add_special=True, parse_special=False)
    >>> out_toks = Llama.generate(in_toks, n_predict=16)
    >>> out_txt = Llama.detokenize(out_toks, special=True)
    >>> print(out_txt)
    b" the tree, as the saying goes, and I think that's especially true when"
    """
    
    def __init__(
        self,
        path_model: str,
        n_gpu_layers: int = 0, # use < 0 to offload all layers
        use_mmap: bool = True,
        use_mlock: bool = False,
        n_ctx: int = 512, # use <= 0 for n_ctx_train
        n_batch: int = 2048,
        rope_freq_base: float = 0.0, # use 0.0 for auto
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        offload_kqv: bool = False,
        flash_attn: bool = False,
        warmup: bool = True
    ):
        """
        Load a llama model from a file

        - path_model:
            The path to the GGUF model file you wish to load from
        - n_gpu_layers:
            How many of the model's layers should be offloaded from CPU to GPU.
            Values less than 0 will attempt to offload all layers. Default is 0.
        - use_mmap:
            Whether to memory-map the model. Changing this to False will cause
            slower load times. Default is True. 
        - use_mlock:
            Whether to lock the model into memory, which can prevents page-outs.
            Changing this to True can cause slower load times and increased
            memory usage. Default is False.
        - n_ctx:
            The context length at which to load the model, in tokens. Default is
            512, which is very small. Increase as needed. Values 0 or less will
            attempt to load the native context length of the model (which may be
            very large).
        - n_batch:
            The maximum number of tokens to process at once. Higher values
            will increase prompt processing speed at expense of increased memory
            usage. Values must be between 32 and n_ctx inclusive.
        - rope_freq_base:
            The RoPE frequency base (theta) to use when loading the model.
            Default is 0.0, which will determine the correct value
            automatically. Recommended to leave at 0.0 unless you know what
            you're doing.
        - type_k:
            The `libllama.GGMLType` to use for the K cache. Default is 1 (f16).
            In most cases, this must be the same as `type_v`.
        - type_v:
            The `libllama.GGMLType` to use for the V cache. Default is 1 (f16).
            In most cases, this must be the same as `type_k`. Values other than
            1 are not compatible with `flash_attn=True`.
        - offload_kqv:
            Whether to offload the K, Q, V caches to the GPU, which can greatly
            improve prompt processing speed at the cost of increased VRAM usage.
            Default is False for compatability reasons. Recommended to set to
            True if possible.
        - flash_attn:
            Whether to use Flash Attention, which decreases memory usage and
            can increase both prompt processing and text generation speed,
            especially at long context lengths. Default is False. Recommended
            to set to True if possible.
        - warmup:
            Whether to warm-up the model with an empty run. This reduces the
            latency of the first generation at the cost of a slightly slower
            load time.
        """
        if not os.path.exists(path_model):
            raise FileNotFoundError(
                f"Llama: the given path_model {path_model!r} does not exist"
            )
        if os.path.isdir(path_model):
            raise IsADirectoryError(
                f"Llama: the given path_model {path_model!r} is a directory, "
                f"not a GGUF file"
            )
        
        # peek at metadata from GGUF file header before loading model

        self.metadata = QuickGGUFReader.load_metadata(path_model)
        
        #
        # Load model from file
        #

        self._model = _LlamaModel(
            path_model=path_model,
            n_gpu_layers=n_gpu_layers,
            use_mmap=use_mmap,
            use_mlock=use_mlock
        )

        n_ctx_train = lib.llama_n_ctx_train(self._model.model)

        # use n_ctx unless it's 0 or negative, in that case use n_ctx_train

        if n_ctx <= 0:
            print_info(
                f'n_ctx value {n_ctx}; using n_ctx_train value '
                f'{n_ctx_train}'
            )
            _n_ctx = int(n_ctx_train)
        else:
            _n_ctx = int(n_ctx)

        # use rope_freq_base unless it == 0.0, in that case use the native
        # rope_freq_base found in the GGUF metadata

        if rope_freq_base == 0.0:
            rope_freq_base_train = None
            for key in self.metadata.keys():
                if key.endswith('.rope.freq_base'):
                    rope_freq_base_train = float(self.metadata[key])
            
            # NOTE: if n_ctx > n_ctx_train, then rope_freq_base must also be
            #       increased by at least a proportional amount to guarantee a
            #       usable kv cache throughout the entire context
            #
            #       the function _calculate_rope_freq_base handles this

            _rope_freq_base = _calculate_rope_freq_base(
                n_ctx_train=n_ctx_train,
                n_ctx_load=_n_ctx,
                rope_freq_base_train=rope_freq_base_train # can be None
            )
        else:
            _rope_freq_base = rope_freq_base

        #
        # New context with model
        #
        
        self._ctx = _LlamaCtx(
            model=self._model,
            n_ctx=_n_ctx,
            n_batch=n_batch,
            n_threads=_get_optimal_n_threads(),
            n_threads_batch=_get_optimal_n_threads_batch(),
            rope_freq_base=_rope_freq_base,
            type_k=type_k,
            type_v=type_v,
            offload_kqv=offload_kqv,
            flash_attn=flash_attn
        )

        #
        # Display warnings about n_ctx if necessary
        #

        actual_n_ctx = self.n_ctx()
        requested_n_ctx = _n_ctx

        if actual_n_ctx != requested_n_ctx:
            print_warning(
                f"requested n_ctx value differs from actual n_ctx value; "
                f"requested {requested_n_ctx}, actual {actual_n_ctx}"
            )
        if actual_n_ctx < 512:
            print_warning(
                f"n_ctx value {actual_n_ctx} is less than 512, which can "
                f"sometimes cause problems with llama.cpp - consider "
                f"increasing it to at least 512"
            )
        if actual_n_ctx % 512 != 0:
            print_warning(
                f"n_ctx value {actual_n_ctx} is not divisible by 512, which "
                f"can sometimes cause problems with llama.cpp - consider "
                f"changing it to "
                f"{_round_n_ctx(actual_n_ctx, n_ctx_train)}."
            )
        if actual_n_ctx == 512:
            print_warning(
                f'you are using the default n_ctx value {actual_n_ctx}, which '
                f'is very small. increase n_ctx as needed to support longer '
                f'inputs and outputs.'
            )
        
        self.stopwatch = LlamaStopwatch()

        #
        # Store immutable Llama metadata as attributes for faster access
        #

        self._n_ctx                 = self.n_ctx()
        self._n_batch               = self.n_batch()
        self._n_ubatch              = self.n_ubatch()
        self._n_seq_max             = self.n_seq_max()
        self._n_vocab               = self.n_vocab()
        self._n_ctx_train           = self.n_ctx_train()
        self._n_embd                = self.n_embd()
        self._n_layer               = self.n_layer()
        self._n_head                = self.n_head() # attn heads, not KV
        self._pooling_type          = self.pooling_type()
        self._vocab_type            = self.vocab_type()
        self._rope_type             = self.rope_type()
        self._rope_freq_scale_train = self.rope_freq_scale_train()
        self._model_size            = self.model_size()
        self._model_n_params        = self.model_n_params()
        self._model_has_encoder     = self.model_has_encoder()
        self._model_has_decoder     = self.model_has_decoder()
        self._model_is_recurrent    = self.model_is_recurrent()
        self._token_bos             = self.token_bos()
        self._token_eos             = self.token_eos()
        self._token_eot             = self.token_eot()
        self._token_cls             = self.token_cls()
        self._token_sep             = self.token_sep()
        self._token_nl              = self.token_nl()
        self._token_pad             = self.token_pad()
        self._add_bos_token         = self.add_bos_token()
        self._add_eos_token         = self.add_eos_token()
        self._token_fim_pre         = self.token_fim_pre()
        self._token_fim_suf         = self.token_fim_suf()
        self._token_fim_mid         = self.token_fim_mid()
        self._token_fim_pad         = self.token_fim_pad()
        self._token_fim_rep         = self.token_fim_rep()
        self._token_fim_sep         = self.token_fim_sep()

        self.eog_tokens = [i for i in range(self._n_vocab) if self.token_is_eog(i)]
        """
        A list of all tokens in the vocab that are marked as EOG
        (End-Of-Generation)
        """

        # internal use only - the default SamplerParams with this model
        self._default_sampler_params = SamplerParams(self)

        self.pos = 0
        """The number of tokens in the context window that have been processed"""

        self.context_tokens = []
        """A list of all tokens currently in the context window"""

        if warmup:
            # warm up the model with an empty run
            print_info('warming up the model with an empty run ...')
            _internals.decode_tg(self._ctx.ctx, 0, 0)
            print_info('model is warm')

        # End of Llama.__init__
    
    def __repr__(self) -> str:
        return (
            f"Llama("
            f"path_model={self._model.path_model!r}, "
            f"n_gpu_layers={self._model.params.n_gpu_layers}, "
            f"use_mmap={self._model.params.use_mmap}, "
            f"use_mlock={self._model.params.use_mlock}, "
            f"n_ctx={self._n_ctx}, "
            f"n_batch={self._n_batch}, "
            f"rope_freq_base={self._ctx.params.rope_freq_base}, "
            f"type_k={self._ctx.params.type_k}, "
            f"type_v={self._ctx.params.type_v}, "
            f"offload_kqv={self._ctx.params.offload_kqv}, "
            f"flash_attn={self._ctx.params.flash_attn}"
            f")"
        )
    
    def free(self):
        """Deallocate the context and model"""
        self._ctx.free()
        self._model.free()

    def n_ctx(self) -> int:
        """Get the current context length"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_ctx")
        return lib.llama_n_ctx(self._ctx.ctx)

    def n_batch(self) -> int:
        """Get the current batch size"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_batch")
        return lib.llama_n_batch(self._ctx.ctx)

    def n_ubatch(self) -> int:
        """Get the current micro-batch size"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_batch")
        return lib.llama_n_ubatch(self._ctx.ctx)

    def n_seq_max(self) -> int:
        """Get the max number of sequences"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_seq_max")
        return lib.llama_n_seq_max(self._ctx.ctx)

    def n_vocab(self) -> int:
        """Get the vocab size"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_vocab")
        return lib.llama_n_vocab(self._model.model)

    def n_ctx_train(self) -> int:
        """Get the trained context length"""
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.n_ctx_train'
        )
        return lib.llama_n_ctx_train(self._model.model)

    def n_embd(self) -> int:
        """Get the embedding size"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_embd")
        return lib.llama_n_embd(self._model.model)

    def n_layer(self) -> int:
        """Get the number of layers"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_layer")
        return lib.llama_n_layer(self._model.model)

    def n_head(self) -> int:
        """Get the number of attention heads"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_head")
        return lib.llama_n_head(self._model.model)

    def pooling_type(self) -> int:
        """Get the pooling type"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.pooling_type")
        return lib.llama_pooling_type(self._ctx.ctx)

    def vocab_type(self) -> int:
        """Get the vocab type"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.vocab_type"
        )
        return lib.llama_vocab_type(self._model.model)

    def rope_type(self) -> int:
        """Get the RoPE type"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.rope_type"
        )
        return lib.llama_rope_type(self._model.model)

    def rope_freq_scale_train(self) -> float:
        """Get the trained RoPE frequency scale"""
        null_ptr_check(
            self._model.model,
            "self._model.model",
            "Llama.rope_freq_scale_train"
        )
        return lib.llama_rope_freq_scale_train(self._model.model)

    def model_size(self) -> int:
        """Get the total size of the model in bytes"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_size"
        )
        return lib.llama_model_size(self._model.model)

    def model_n_params(self) -> int:
        """Get the total number of parameters in the model"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_n_params"
        )
        return lib.llama_model_n_params(self._model.model)

    def model_has_encoder(self) -> bool:
        """If the model has an encoder"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_has_encoder"
        )
        return lib.llama_model_has_encoder(self._model.model)

    def model_has_decoder(self) -> bool:
        """If the model has a decoder"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_has_decoder"
        )
        return lib.llama_model_has_decoder(self._model.model)

    def model_is_recurrent(self) -> bool:
        """If the model is recurrent"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_is_recurrent"
        )
        return lib.llama_model_is_recurrent(self._model.model)

    def kv_cache_clear(self) -> None:
        """Clear the KV cache"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_clear")
        lib.llama_kv_cache_clear(self._ctx.ctx)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        """Remove tokens from a sequence in the KV cache"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_rm")
        return lib.llama_kv_cache_seq_rm(self._ctx.ctx, seq_id, p0, p1)

    def kv_cache_seq_cp(
        self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int
    ) -> None:
        """Copy tokens between sequences in the KV cache"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_cp")
        lib.llama_kv_cache_seq_cp(self._ctx.ctx, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int) -> None:
        """Remove all tokens except for the ones in this sequence"""
        null_ptr_check(
            self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_keep"
        )
        lib.llama_kv_cache_seq_keep(self._ctx.ctx, seq_id)

    def kv_cache_seq_add(
        self, seq_id: int, p0: int, p1: int, delta: int
    ) -> None:
        """Add relative position "delta" to the tokens"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_add")
        lib.llama_kv_cache_seq_add(self._ctx.ctx, seq_id, p0, p1, delta)

    def kv_cache_seq_div(self, seq_id: int, p0: int, p1: int, d: int) -> None:
        """Integer division of the positions by factor of `d > 1`"""
        null_ptr_check(
            self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_div"
        )
        lib.llama_kv_cache_seq_div(self._ctx.ctx, seq_id, p0, p1, d)

    def kv_cache_seq_pos_max(self, seq_id: int) -> int:
        """Returns the largest position present in the KV cache for the specified sequence"""
        null_ptr_check(
            self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_pos_max"
        )
        return lib.llama_kv_cache_seq_pos_max(self._ctx.ctx, seq_id)

    def kv_cache_defrag(self) -> None:
        """Defragment the KV cache"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_defrag")
        lib.llama_kv_cache_defrag(self._ctx.ctx)

    def kv_cache_update(self) -> None:
        """Apply the KV cache updates (K-shifts, defragmentation, etc.)"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_update")
        lib.llama_kv_cache_update(self._ctx.ctx)

    def kv_cache_can_shift(self) -> bool:
        """Check if the context supports KV cache shifting"""
        null_ptr_check(
            self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_can_shift"
        )
        return lib.llama_kv_cache_can_shift(self._ctx.ctx)

    def n_threads(self) -> int:
        """Get the number of threads used for batch size 1"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_threads")
        return lib.llama_n_threads(self._ctx.ctx)

    def n_threads_batch(self) -> int:
        """Get the number of threads used for batch sizes > 1"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_threads_batch")
        return lib.llama_n_threads_batch(self._ctx.ctx)

    # NOTE: this is disabled until i figure out what it does
    # def token_get_score(self, token: int) -> float:
    #     """Get the score of a token"""
    #     null_ptr_check(
    #         self._model.model, "self._model.model", "Llama.token_get_score"
    #     )
    #     return lib.llama_token_get_score(self._model.model, token)

    def token_is_eog(self, token: int) -> bool:
        """If the token is marked as EOG (End-Of-Generation)"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_is_eog"
        )
        return lib.llama_token_is_eog(self._model.model, token)

    def token_is_control(self, token: int) -> bool:
        """If the token is marked as a control token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_is_control"
        )
        return lib.llama_token_is_control(self._model.model, token)

    def token_bos(self) -> int:
        """Get the BOS (Beginning-Of-Sequence) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_bos"
        )
        return lib.llama_token_bos(self._model.model)

    def token_eos(self) -> int:
        """Get the EOS (End-Of-Sequence) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_eos"
        )
        return lib.llama_token_eos(self._model.model)

    def token_eot(self) -> int:
        """Get the EOT (End-Of-Turn) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_eot"
        )
        return lib.llama_token_eot(self._model.model)

    def token_cls(self) -> int:
        """Get the CLS (Classification) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_cls"
        )
        return lib.llama_token_cls(self._model.model)

    def token_sep(self) -> int:
        """Get the SEP (Separator) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_sep"
        )
        return lib.llama_token_sep(self._model.model)

    def token_nl(self) -> int:
        """Get the NL (Newline) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_nl"
        )
        return lib.llama_token_nl(self._model.model)

    def token_pad(self) -> int:
        """Get the PAD (Padding) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_pad"
        )
        return lib.llama_token_pad(self._model.model)

    def add_bos_token(self) -> bool:
        """If the model is configured to add a BOS token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.add_bos_token"
        )
        return lib.llama_add_bos_token(self._model.model)

    def add_eos_token(self) -> bool:
        """If the model is configured to add an EOS token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.add_eos_token"
        )
        return lib.llama_add_eos_token(self._model.model)

    def token_fim_pre(self) -> int:
        """Get the FIM PRE (Fill-In-Middle Prefix) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_pre"
        )
        return lib.llama_token_fim_pre(self._model.model)

    def token_fim_suf(self) -> int:
        """Get the FIM SUF (Fill-In-Middle Suffix) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_suf"
        )
        return lib.llama_token_fim_suf(self._model.model)

    def token_fim_mid(self) -> int:
        """Get the FIM MID (Fill-In-Middle Middle) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_mid"
        )
        return lib.llama_token_fim_mid(self._model.model)

    def token_fim_pad(self) -> int:
        """Get the FIM PAD (Fill-In-Middle Padding) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_pad"
        )
        return lib.llama_token_fim_pad(self._model.model)

    def token_fim_rep(self) -> int:
        """Get the FIM REP (Fill-In-Middle Repository) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_rep"
        )
        return lib.llama_token_fim_rep(self._model.model)

    def token_fim_sep(self) -> int:
        """Get the FIM SEP (Fill-In-Middle Separator) token"""
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_sep"
        )
        return lib.llama_token_fim_sep(self._model.model)
    
    def tokenize(
        self,
        text_bytes: bytes,
        add_special: bool,
        parse_special: bool,
    ) -> list[int]:
        """
        Convert the provided UTF-8 encoded text into tokens

        - text_bytes:
            The text to be tokenized
        - add_special:
            Allow to add BOS and EOS tokens if model is configured to do so.
        - parse_special:
            Allow tokenizing special and/or control tokens which otherwise are
            not exposed and treated as plaintext. Does not insert a leading
            space.
        """
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.tokenize'
        )
        n_tokens = _internals.get_length(
            model=self._model.model,
            text_bytes=text_bytes,
            add_special=add_special,
            parse_special=parse_special
        )
        return _internals.tokenize(
            model=self._model.model,
            text_bytes=text_bytes,
            n_tokens_max=n_tokens,
            add_special=add_special,
            parse_special=parse_special
        )

    def token_to_piece(self, token: int, special: bool) -> bytes:
        """
        Convert a single token ID into utf-8 bytes

        - special:
            If True, special tokens are rendered in the output
        """
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.token_to_piece'
        )
        return _internals.token_to_piece(
            model=self._model.model,
            token=token,
            special=special
        )

    def detokenize(
        self,
        tokens: Iterable[int],
        special: bool
    ) -> bytes:
        """
        Convert the provided tokens into UTF-8 encoded text

        - special:
            If True, special tokens are rendered in the output
        """
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.detokenize'
        )
        return _internals.detokenize(
            model=self._model.model,
            tokens=tokens,
            special=special
        )
    
    def get_length(
        self,
        text_bytes: bytes,
        add_special: bool,
        parse_special: bool,
    ) -> int:
        """
        Return the length of a given text as measured in tokens
        """
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.get_length'
        )
        return _internals.get_length(
            model=self._model.model,
            text_bytes=text_bytes,
            add_special=add_special,
            parse_special=parse_special
        )

    def first_valid_pos(self, tokens: Iterable[int]) -> int:
        """
        Given a list of tokens, and using `Llama.context_tokens`, find the first
        valid `Llama.pos`

        In other words, return length of the longest common prefix between the
        two iterables of tokens

        Returns 0 if none of the tokens match, 1 if one token matches, etc.
        """
        i = 0
        for c, t in zip(self.context_tokens, tokens):
            if c == t:
                i += 1
            else:
                break
        return i

    def generate_single(
        self,
        input_tokens: Iterable[int],
        sampler_params: Optional[SamplerParams] = None,
    ) -> int:
        """
        Generate a single token

        - input_tokens:
            The tokens to evaluate
        - sampler:
            The `SamplerParams` object to use for sampling. If not specified,
            use the model's default sampler parameters
        """

        n_tokens = len(input_tokens)

        if n_tokens == 0:
            raise ValueError(
                f'Llama.generate_single: input_tokens cannot be empty'
            )
        
        # find how many tokens in the input are already in the KV cache
        self.pos = self.first_valid_pos(input_tokens)

        # remove all tokens that are past that point
        self.kv_cache_seq_rm(0, self.pos, -1)
        actual_input_tokens = input_tokens[self.pos:] # tokens after self.pos
        self.context_tokens = input_tokens[:self.pos] # tokens already processed

        n_cache_miss_tokens = len(actual_input_tokens)
        n_cache_hit_tokens = len(input_tokens) - n_cache_miss_tokens

        print_info(
            f'Llama.generate_single: {n_cache_hit_tokens} tokens in cache, '
            f'{n_cache_miss_tokens} tokens to eval ...'
        )

        # split the input into batches of tokens
        batch_splits = range(0, n_tokens, self._n_batch)
        # print(f'batch_split: {batch_splits=}')
        batches = []
        for i in batch_splits:
            # print(f'this batch split index: {i}')
            batch_tokens = actual_input_tokens[i : i + self._n_batch]
            # print(f'batch_split: {batch_tokens=}')
            if len(batch_tokens) > 0:
                batches.append(batch_tokens)
                # print(
                #     f'batch_split: appended batch with {len(batch_tokens)} '
                #     f'tokens'
                # )

        # n_batches = len(batches)
        # print(f'batch_split: {n_batches=}')

        # set up the stopwatch
        self.stopwatch.reset()

        # process each batch one-by-one
        for batch in batches:

            n_batch_tokens = len(batch)
            
            if n_batch_tokens > 1:
                # print(
                #     f'ppdecode: {batch_tokens=}\n'
                #     f'ppdecode: {n_batch_tokens=}'
                # )
                self.stopwatch.start_pp()
                _internals.decode_pp(
                    self._ctx.ctx, self.pos, batch, n_batch_tokens
                )
                self.stopwatch.stop_pp()
                self.stopwatch.increment_pp_tokens(n_batch_tokens)
            elif n_batch_tokens == 1:
                # print(
                #     f'tgdecode: {batch_tokens=}\n'
                #     f'tgdecode: {n_batch_tokens=}'
                # )
                self.stopwatch.start_tg()
                _internals.decode_tg(self._ctx.ctx, self.pos, batch[0])
                self.stopwatch.stop_tg()
                self.stopwatch.increment_tg_tokens(1)
            else:
                raise RuntimeError(
                    f'Llama.generate_single: unexpected n_batch_tokens value '
                    f'{n_batch_tokens}'
                )
            
            # update the Llama position and context
            self.pos += n_batch_tokens
            self.context_tokens.extend(batch)
        
        # sample and return
        self.stopwatch.print_stats()
        return self.sample(sampler_params)

    def generate(
        self,
        input_tokens: Iterable[int],
        n_predict: int,
        stop_tokens: Optional[Iterable[int]] = None,
        sampler_params: Optional[SamplerParams] = None
    ) -> list[int]:
        """
        Generate one or more tokens and return them all at once

        - input_tokens:
            The tokens to evaluate
        - n_predict:
            The number of tokens to predict. If `n_predict <= 0`, then the
            number of tokens predicted is only limited by the context length.
        - stop_tokens:
            A list of token IDs that will end the generation early. Note that
            the stop token will be included in the output. If this parameter is
            None, all built-in stop tokens for the model will be used. Pass an
            empty list `[]` to ignore all stop tokens.
        - sampler_params:
            The `SamplerParams` object to use for sampling. If not specified,
            use the default sampler parameters
        """

        n_tokens = len(input_tokens)

        if n_tokens == 0:
            raise ValueError('Llama.generate: input_tokens cannot be empty')
        
        # find how many tokens in the input are already in the KV cache
        self.pos = self.first_valid_pos(input_tokens)

        # remove all tokens that are past that point
        self.kv_cache_seq_rm(0, self.pos, -1)
        actual_input_tokens = input_tokens[self.pos:] # tokens after self.pos
        self.context_tokens = input_tokens[:self.pos] # tokens already processed

        n_cache_miss_tokens = len(actual_input_tokens)
        n_cache_hit_tokens = len(input_tokens) - n_cache_miss_tokens

        print_info(
            f'Llama.generate: {n_cache_hit_tokens} tokens in cache, '
            f'{n_cache_miss_tokens} tokens to eval ...'
        )

        stop_tokens = stop_tokens if stop_tokens is not None else self.eog_tokens

        # split the input into batches of tokens
        batch_splits = range(0, n_tokens, self._n_batch)
        # print(f'batch_split: {batch_splits=}')
        batches = []
        for i in batch_splits:
            # print(f'this batch split index: {i}')
            batch_tokens = actual_input_tokens[i : i + self._n_batch]
            # print(f'batch_split: {batch_tokens=}')
            if len(batch_tokens) > 0:
                batches.append(batch_tokens)
                # print(
                #     f'batch_split: appended batch with {len(batch_tokens)} '
                #     f'tokens'
                # )
        
        # n_batches = len(batches)
        # print(f'batch_split: {n_batches=}')

        # set up the loop
        self.stopwatch.reset()
        output_tokens = []
        n_predicted = 0

        # process each input batch one-by-one
        for batch in batches:

            n_batch_tokens = len(batch)
            
            if n_batch_tokens > 1:
                # print(
                #     f'ppdecode: {batch_tokens=}\n'
                #     f'ppdecode: {n_batch_tokens=}'
                # )
                self.stopwatch.start_pp()
                _internals.decode_pp(
                    self._ctx.ctx, self.pos, batch, n_batch_tokens
                )
                self.stopwatch.stop_pp()
                self.stopwatch.increment_pp_tokens(n_batch_tokens)
            elif n_batch_tokens == 1:
                # print(
                #     f'tgdecode: {batch_tokens=}\n'
                #     f'tgdecode: {n_batch_tokens=}'
                # )
                self.stopwatch.start_tg()
                _internals.decode_tg(self._ctx.ctx, self.pos, batch[0])
                self.stopwatch.stop_tg()
                self.stopwatch.increment_tg_tokens(1)
            else:
                raise RuntimeError(
                    f'Llama.generate: unexpected n_batch_tokens value '
                    f'{n_batch_tokens}'
                )
            
            # update the Llama position and context
            self.pos += n_batch_tokens
            self.context_tokens.extend(batch)
        
        # continue generating until n_predict or n_ctx is reached
        # print(f'start while loop')
        while (n_predicted < n_predict) if n_predict > 0 else (self.pos < self._n_ctx):
            self.stopwatch.start_tg()
            _internals.decode_tg(self._ctx.ctx, self.pos, self.context_tokens[-1])
            self.stopwatch.stop_tg()
            self.stopwatch.increment_tg_tokens(1)
            self.pos += 1

            id = self.sample(sampler_params)
            self.context_tokens.append(id)
            output_tokens.append(id)
            n_predicted += 1

            if id in stop_tokens:
                self.stopwatch.print_stats()
                return output_tokens
        
        self.stopwatch.print_stats()
        return output_tokens

    def stream(
        self,
        input_tokens: Iterable[int],
        n_predict: int,
        stop_tokens: Optional[Iterable[int]] = None,
        sampler_params: Optional['SamplerParams'] = None
    ) -> Iterable[int]:
        """
        Return a Generator which yields one or more tokens

        - input_tokens:
            The tokens to evaluate
        - n_predict:
            The number of tokens to predict. If `n_predict <= 0`, then the
            number of tokens predicted is only limited by the context length.
        - stop_tokens:
            A list of token IDs that will end the generation early. Note that
            the stop token will be included in the output. If this parameter is
            None, all built-in stop tokens for the model will be used. Pass an
            empty list `[]` to ignore all stop tokens.
        - sampler_params:
            The `SamplerParams` object to use for sampling. If not specified,
            use the default sampler parameters
        """

        n_tokens = len(input_tokens)

        if n_tokens == 0:
            raise ValueError('Llama.stream: input_tokens cannot be empty')
        
        # find how many tokens in the input are already in the KV cache
        self.pos = self.first_valid_pos(input_tokens)

        # remove all tokens that are past that point
        self.kv_cache_seq_rm(0, self.pos, -1)
        actual_input_tokens = input_tokens[self.pos:] # tokens after self.pos
        self.context_tokens = input_tokens[:self.pos] # tokens already processed

        n_cache_miss_tokens = len(actual_input_tokens)
        n_cache_hit_tokens = len(input_tokens) - n_cache_miss_tokens

        print_info(
            f'Llama.stream: {n_cache_hit_tokens} tokens in cache, '
            f'{n_cache_miss_tokens} tokens to eval ...'
        )

        stop_tokens = stop_tokens if stop_tokens is not None else self.eog_tokens

        # split the input into batches of tokens
        batch_splits = range(0, n_tokens, self._n_batch)
        # print(f'batch_split: {batch_splits=}')
        batches = []
        for i in batch_splits:
            # print(f'this batch split index: {i}')
            batch_tokens = actual_input_tokens[i : i + self._n_batch]
            # print(f'batch_split: {batch_tokens=}')
            if len(batch_tokens) > 0:
                batches.append(batch_tokens)
                # print(
                #     f'batch_split: appended batch with {len(batch_tokens)} '
                #     f'tokens'
                # )
        
        # n_batches = len(batches)
        # print(f'batch_split: {n_batches=}')

        # set up the loop
        self.stopwatch.reset()
        n_predicted = 0

        # process each input batch one-by-one
        for batch in batches:

            n_batch_tokens = len(batch)
            
            if n_batch_tokens > 1:
                # print(
                #     f'ppdecode: {batch_tokens=}\n'
                #     f'ppdecode: {n_batch_tokens=}'
                # )
                self.stopwatch.start_pp()
                _internals.decode_pp(
                    self._ctx.ctx, self.pos, batch, n_batch_tokens
                )
                self.stopwatch.stop_pp()
                self.stopwatch.increment_pp_tokens(n_batch_tokens)
            elif n_batch_tokens == 1:
                # print(
                #     f'tgdecode: {batch_tokens=}\n'
                #     f'tgdecode: {n_batch_tokens=}'
                # )
                self.stopwatch.start_tg()
                _internals.decode_tg(self._ctx.ctx, self.pos, batch[0])
                self.stopwatch.stop_tg()
                self.stopwatch.increment_tg_tokens(1)
            else:
                raise RuntimeError(
                    f'Llama.stream: unexpected n_batch_tokens value '
                    f'{n_batch_tokens}'
                )
            
            # update the Llama position and context
            self.pos += n_batch_tokens
            self.context_tokens.extend(batch)
        
        # continue generating until n_predict or n_ctx is reached
        # print(f'start while loop')
        while (n_predicted < n_predict) if n_predict > 0 else (self.pos < self._n_ctx):
            self.stopwatch.start_tg()
            _internals.decode_tg(self._ctx.ctx, self.pos, self.context_tokens[-1])
            self.stopwatch.stop_tg()
            self.stopwatch.increment_tg_tokens(1)
            self.pos += 1

            id = self.sample(sampler_params)
            self.context_tokens.append(id)
            yield id
            n_predicted += 1

            if id in stop_tokens:
                self.stopwatch.print_stats()
                return
        
        self.stopwatch.print_stats()
        return
    
    def sample_greedy(self) -> int:
        id = _internals.sample_greedy(self._ctx.ctx)
        lib.llama_sampler_accept(_internals.greedy_sampler, id)
        return id
    
    def sample(self, params: Optional[SamplerParams] = None) -> int:
      """
      Sample a token using the current context

      - params
            The `sampling.SamplerParams` object which defines the sampling
            parameters to use. If this parameter is None, the default sampler
            paramater values will be used.
      """
      params = params if params is not None else self._default_sampler_params
      id = lib.llama_sampler_sample(params.smpl, self._ctx.ctx, -1)
      lib.llama_sampler_accept(params.smpl, id)
      return id
    
    def get_logits(self) -> np.ndarray:
        """
        Return the raw logits for the last token in the context

        The returned array has shape `(n_vocab)`.
        """
        null_ptr_check(self._ctx.ctx, 'self._ctx.ctx', 'Llama.logits')
        raw_logits = lib.llama_get_logits_ith(self._ctx.ctx, -1)
        return np.ctypeslib.as_array(raw_logits, shape=[1, self._n_vocab])[0]

    def get_scores(self, temp: Optional[float] = None) -> np.ndarray:
        """
        Return the softmaxed logits for the last token in the context.
        Optionally apply temperature `temp` if specified.

        Any floating-point value for temperature `temp` is valid, including 0.0
        and negative numbers.

        The returned array has shape `(n_vocab)`.
        """
        logits = self.get_logits()
        return softmax(logits, T=temp)

    def reset(self) -> None:
        """Reset the position of the model and clear the KV cache"""
        self.kv_cache_clear()
        self.pos = 0
        self.context_tokens = []
        print_info('model was reset')

class InferenceLock:
    """
    A context manager that can be used to prevent a `llama.Llama` instance from
    accepting more than one generation at a time, which is not supported and can
    cause a hard crash.

    This is mostly useful in asychronous / multi-threaded contexts
    """

    class LockFailure(Exception):
        pass

    def __init__(self):
        self.locked = False

    def __enter__(self):
        return self.acquire()
    
    def __exit__(self, *_):
        return self.release()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *_):
       return self.__exit__()
    
    def acquire(self):
        if self.locked:
            raise self.LockFailure(
                'failed to acquire InferenceLock (already locked)'
            )
        self.locked = True
        return self
    
    def release(self):
        if not self.locked:
            raise self.LockFailure(
                'tried to release InferenceLock that is not acquired'
            )
        self.locked = False

#
# End of functions / Begin test
#

def main() -> int:

    #test_model_path = "/Users/dylan/Documents/AI/models/Llama-3.2-1B-Instruct-q8_0-q8_0.gguf"
    test_model_path = '/Users/dylan/Documents/AI/models/Meta-Llama-3.1-8B-Instruct-q8_0-q6_K.gguf'

    TestLlama = Llama(
        path_model=test_model_path,
        n_gpu_layers=-1,
        use_mmap=True,
        use_mlock=False,
        n_ctx=8192,
        offload_kqv=True,
        flash_attn=True
    )

    chktxt_a = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is Einstein famous for?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    chktxt_b = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is Einstein's full name?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    tokens_a = TestLlama.tokenize(chktxt_a.encode(), add_special=True, parse_special=True)
    tokens_b = TestLlama.tokenize(chktxt_b.encode(), add_special=True, parse_special=True)

    print('-' * 80)

    stream = TestLlama.stream(tokens_a, n_predict=128)
    for tok in stream:
        txt = TestLlama.token_to_piece(tok, True).decode()
        print(txt, end='', file=sys.stderr, flush=True)
    print('\n', end='', file=sys.stderr, flush=True)

    return 0


if __name__ == '__main__':
    exit(main())
