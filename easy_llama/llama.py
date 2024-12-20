# llama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import os
import sys
import time
import struct
import ctypes

from utils import print_verbose, print_info, print_warning, print_error
from typing    import NoReturn, Optional, Iterable
from io        import BufferedReader
from enum      import IntEnum
from constants import Colors

import libllama as lib
import numpy    as np

from libllama import null_ptr_check, NULL, NULLPTR

RESET  = Colors.RESET
GREEN  = Colors.GREEN
BLUE   = Colors.BLUE
GREY   = Colors.GREY
YELLOW = Colors.YELLOW
RED    = Colors.RED

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
    global _cpu_count

    # if already initialized, no need to do anything
    if lib._BACKEND_INIT is True:
        return
    
    _cpu_count = os.cpu_count()
    
    # most cases
    if sys.byteorder == 'little':
        print_verbose(
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
        print_warning(
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
#       the optimal n_threads_batch value (for prompt eval) is equal
#       to the total number of logical cores, regardless of
#       their type

def _get_optimal_n_threads() -> int:
    global _cpu_count
    return max(_cpu_count//2, 1)

def _get_optimal_n_threads_batch() -> int:
    global _cpu_count
    return _cpu_count

def _get_random_seed() -> int:
    # uint32_t
    return int.from_bytes(
        bytes=os.urandom(4),
        byteorder=sys.byteorder,
        signed=False
    )

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

class LlamaPerformanceTracker:
    """Track elapsed time, input tokens, output tokens"""
    #
    # Q: why don't you use llama_perf_context?
    #
    # A: comments in llama.h state to only use that in llama.cpp examples,
    #    and to do your own performance measurements instead
    #
    def __init__(self):
        self.pp_start_time = None
        self.pp_end_time = None
        self.tg_start_time = None
        self.tg_end_time = None
        self.n_pp_tokens = 0
        self.n_tg_tokens = 0
    
    def start_pp(self):
        """Start prompt processing timer"""
        self.pp_start_time = time.time_ns()
    
    def stop_pp(self):
        """Stop prompt processing timer"""
        self.pp_end_time = time.time_ns()
    
    def start_tg(self):
        """Start text generation timer"""
        self.tg_start_time = time.time_ns()
    
    def stop_tg(self):
        """Stop text generation timer"""
        self.tg_end_time = time.time_ns()

    def get_elapsed_time_pp(self) -> int:
        """Return the number of nanoseconds elapsed during prompt processing"""
        return self.pp_end_time - self.pp_start_time
    
    def get_elapsed_time_tg(self) -> int:
        """Return the number of nanoseconds elapsed during text generation"""
        return self.tg_end_time - self.tg_start_time
    
    def increment_pp_tokens(self, n: int):
        self.n_pp_tokens += max(n, 0) # do not allow negative increment
    
    def increment_tg_tokens(self, n: int):
        self.n_tg_tokens += max(n, 0) # do not allow negative increment
    
    def reset(self):
        """Reset the tracker to its original state"""
        self.pp_start_time = None
        self.tg_end_time = None
        self.pp_start_time = None
        self.tg_end_time = None
        self.n_pp_tokens = 0
        self.n_tg_tokens = 0
    
    def print_stats(self):
        """Print performance statistics using current tracker state"""

        if self.n_pp_tokens + self.n_tg_tokens == 0:
            print_info(
                f'LlamaPerformanceTracker: print_stats was called but no '
                f'tokens were processed or generated'
            )
            return
        
        if self.n_pp_tokens > 0:
            pp_elapsed_ns = self.get_elapsed_time_pp()
            pp_elapsed_ms = pp_elapsed_ns / 1e6
            pp_elapsed_s = pp_elapsed_ns / 1e9
            pp_tps = self.n_pp_tokens / pp_elapsed_s
            print_info(
                f'prompt processing for {self.n_pp_tokens} tokens took '
                f'{pp_elapsed_ms:.3f}ms ({pp_tps:.2f} tok/s)'
            )

        if self.n_tg_tokens > 0:
            tg_elapsed_ns = self.get_elapsed_time_tg()
            tg_elapsed_ms = tg_elapsed_ns / 1e6
            tg_elapsed_s = tg_elapsed_ns / 1e9
            tg_tps = self.n_tg_tokens / tg_elapsed_s
            print_info(
                f'text generation of {self.n_tg_tokens} tokens took '
                f'{tg_elapsed_ms:.3f}ms ({tg_tps:.2f} tok/s)'
            )

        if (self.n_pp_tokens > 0) and (self.n_tg_tokens > 0):
            total_elapsed_ns = pp_elapsed_ns + tg_elapsed_ns
            total_elapsed_ms = total_elapsed_ns / 1e6
            total_elapsed_s = total_elapsed_ns / 1e9

            total_tokens = self.n_pp_tokens + self.n_tg_tokens
            total_average_tps = total_tokens / total_elapsed_s

            print_info(
                f'total of {total_tokens} tokens took {total_elapsed_ms:.3f}ms '
                f'({total_average_tps:.2f} tok/s average)'
            )

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
        devices = None,
        n_gpu_layers = 0, # if < 0, then will use MAX_OFFLOAD_LAYERS
        split_mode: Optional[int] = None,
        main_gpu: Optional[int] = None,
        tensor_split = None,
        rpc_servers = None,
        progress_callback = None,
        progress_callback_user_data = None,
        kv_overrides = None,
        vocab_only = False,
        use_mmap = True,
        use_mlock = False,
        check_tensors = True # does some validation on tensor data
    ):
        _init_backend_if_needed()
        self.path_model = path_model
        self.params = lib.llama_model_default_params()
        null_ptr_check(self.params, "self.params", "_LlamaModel.__init__")
        self.params.devices = (
            ctypes.c_void_p * (len(devices) + 1)
        )(*devices, None) if devices is not None else NULL
        self.params.n_gpu_layers = (
            n_gpu_layers
        ) if n_gpu_layers >= 0 else lib.MAX_OFFLOAD_LAYERS
        self.params.split_mode = (
            split_mode
        ) if split_mode is not None else (
            lib.LlamaSplitMode.LLAMA_SPLIT_MODE_NONE
        )
        self.params.main_gpu = main_gpu if main_gpu is not None else 0
        self.params.tensor_split = (
            ctypes.c_float * len(tensor_split)
        )(*tensor_split) if tensor_split is not None else NULL
        self.params.rpc_servers = (
            rpc_servers.encode('utf-8')
        ) if rpc_servers is not None else NULL
        self.params.progress_callback = (
            progress_callback
        ) if progress_callback is not None else lib.dummy_progress_callback()
        self.params.progress_callback_user_data = (
            progress_callback_user_data
        ) if progress_callback_user_data is not None else NULL
        self.params.kv_overrides = (
            lib.llama_model_kv_override * len(kv_overrides)
        )(*kv_overrides) if kv_overrides is not None else NULL
        self.params.vocab_only = vocab_only
        self.params.use_mmap = use_mmap
        self.params.use_mlock = use_mlock
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
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        n_seq_max = 1,
        n_threads = 0,
        n_threads_batch = 0,
        rope_scaling_type: Optional[int] = None,
        pooling_type: Optional[int] = None,
        attention_type: Optional[int] = None,
        rope_freq_base = 0.0,
        rope_freq_scale = 0.0,
        yarn_ext_factor = 0.0,
        yarn_attn_factor = 0.0,
        yarn_beta_fast = 0.0,
        yarn_beta_slow = 0.0,
        yarn_orig_ctx = 0,
        defrag_thold = 0.0,
        cb_eval = None,
        cb_eval_user_data = None,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        logits_all = False,
        embeddings = False,
        offload_kqv = False,
        flash_attn = False,
        no_perf = False,
        abort_callback = None,
        abort_callback_data = None
    ):
        _init_backend_if_needed()
        self.params = lib.llama_context_default_params()
        null_ptr_check(self.params, "self.params", "_LlamaCtx.__init__")
        self.params.n_ctx = n_ctx
        self.params.n_batch = n_batch
        self.params.n_ubatch = n_ubatch
        if n_seq_max != 1:
            raise NotImplementedError(
                f'n_seq_max value {n_seq_max} != 1; this is not yet supported'
            )
        self.params.n_seq_max = n_seq_max
        self.params.n_threads = n_threads
        self.params.n_threads_batch = n_threads_batch
        self.params.rope_scaling_type = (
            rope_scaling_type
        ) if rope_scaling_type is not None else (
            lib.LlamaRopeScalingType.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
        )
        self.params.pooling_type = (
            pooling_type
        ) if pooling_type is not None else (
            lib.LlamaPoolingType.LLAMA_POOLING_TYPE_UNSPECIFIED
        )
        self.params.attention_type = (
            attention_type
        ) if attention_type is not None else (
            lib.LlamaAttentionType.LLAMA_ATTENTION_TYPE_UNSPECIFIED
        )
        self.params.rope_freq_base = rope_freq_base
        self.params.rope_freq_scale = rope_freq_scale
        self.params.yarn_ext_factor = yarn_ext_factor
        self.params.yarn_attn_factor = yarn_attn_factor
        self.params.yarn_beta_fast = yarn_beta_fast
        self.params.yarn_beta_slow = yarn_beta_slow
        self.params.yarn_orig_ctx = yarn_orig_ctx
        self.params.defrag_thold = defrag_thold
        self.params.cb_eval = (
            cb_eval
        ) if cb_eval is not None else lib.dummy_eval_callback()
        self.params.cb_eval_user_data = cb_eval_user_data
        self.params.type_k = type_k if type_k is not None else _DEFAULT_KV_TYPE
        self.params.type_v = type_v if type_v is not None else _DEFAULT_KV_TYPE
        _k, _v = self.params.type_k, self.params.type_v
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
                f'V cache quantization requires flash_attn, program may fail'
            )
        self.params.logits_all = logits_all
        self.params.embeddings = embeddings
        self.params.offload_kqv = offload_kqv
        self.params.flash_attn = flash_attn
        self.params.no_perf = no_perf
        self.params.abort_callback = (
                abort_callback
            ) if abort_callback is not None else lib.dummy_abort_callback()
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


class _LlamaBatch:

    def __init__(
        self,
        n_tokens: int,
        embd: int = 0,
        n_seq_max: int = 1
    ):
        _init_backend_if_needed()
        if embd != 0:
            raise NotImplementedError(
                f'embd value {embd} != 0; this is not yet supported'
            )
        if n_seq_max != 1:
            raise NotImplementedError(
                f'n_seq_max value {n_seq_max} != 1; this is not yet supported'
            )
        self.batch = lib.llama_batch_init(
            n_tokens=n_tokens,
            embd=embd,
            n_seq_max=n_seq_max
        )
        null_ptr_check(self.batch, "self.batch", "_LlamaBatch.__init__")
    
    # NOTE: not supposed to use this. see libllama for details
    # def get_one(self, tokens: Iterable[int]) -> None:
    #     """
    #     Allocate a batch for the given tokens
    #     """
    #     self.free() # free the old self.batch, if necessary
    #     n_tokens = len(tokens)
    #     tokens_array = (lib.llama_token * n_tokens)(*tokens)
    #     self.batch = lib.llama_batch_get_one(
    #         tokens=tokens_array, n_tokens=n_tokens
    #     )
    #     null_ptr_check(self.batch, 'self.batch', '_LlamaBatch.get_one')
    
    def __del__(self):
        self.free()

    def free(self):
        if self.batch is not None:
            lib.llama_batch_free(self.batch)
            self.batch = None
    
    def n_tokens(self) -> int:
        return self.batch.n_tokens

    def reset(self):
        self.batch.n_tokens = 0
    
    def set(self, batch: Iterable[int], n_past: int, logits_all: bool):
        self.free()
        n_tokens = len(batch)
        self.batch = lib.llama_batch_init(
            n_tokens=n_tokens,
            embd=0,
            n_seq_max=1
        )
        null_ptr_check(self.batch, 'self.batch', '_LlamaBatch.set')
        self.batch.n_tokens = n_tokens
        print(f'batch set: n_tokens is {n_tokens}')
        for i in range(n_tokens):
            print(f'batch set: {i=}')
            self.batch.token[i]     = batch[i]
            print(f'batch set: token[{i}] = {batch[i]}')
            self.batch.pos[i]       = n_past + i
            print(f'batch set: pos[{i}] = {n_past + i}')
            self.batch.seq_id[i][0] = 0
            print(f'batch set: seq_id[{i}][0] = 0')
            self.batch.n_seq_id[i]  = 1
            print(f'batch set: n_seq_id[{i}] = 1')
            self.batch.logits[i]    = logits_all
            print(f'batch set: logits[{i}] = {logits_all}')
        self.batch.logits[n_tokens - 1] = True
        print(f'batch set: logits[{n_tokens - 1}] = True')

#
# Llama
#

class Llama:
    """
    Simplified interface for general-purpose Llama model usage
    """
    
    def __init__(
        self,
        path_model: str,
        n_gpu_layers: int = 0, # use < 0 to offload all layers
        use_mmap: bool = True,
        use_mlock: bool = False,
        n_ctx: int = 512, # use <= 0 for n_ctx_train, otherwise unmodified
        rope_freq_base: float = 0.0, # use 0.0 for auto, otherwise unmodified
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        offload_kqv: bool = False,
        flash_attn: bool = False
    ):
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
            
            # NOTE: if n_ctx is > n_ctx_train, then rope_freq_base must also
            #       be increased by at least a proportional amount to guarantee
            #       a usable kv cache throughout the entire context
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
        
        #
        # Set the tracker and batch for this model
        #

        # TODO: support multiple batches per Llama
        
        self.tracker = LlamaPerformanceTracker()
        self._batch = _LlamaBatch(
            n_tokens=self.n_batch(),
            embd=0,
            n_seq_max=1
        )

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

        self.pos = 0
        """The current position of the model within the context window"""

        self.context_tokens = []
        """A list of all tokens currently in context"""

        # End of Llama.__init__
    
    def free(self):
        """Deallocate the batch, context, and model"""
        self._batch.free()
        self._ctx.free()
        self._model.free()

    def n_ctx(self) -> int:
        """Return the currently loaded context length of this model"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_ctx")
        return lib.llama_n_ctx(self._ctx.ctx)

    def n_batch(self) -> int:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_batch")
        return lib.llama_n_batch(self._ctx.ctx)

    def n_ubatch(self) -> int:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_batch")
        return lib.llama_n_ubatch(self._ctx.ctx)

    def n_seq_max(self) -> int:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_seq_max")
        return lib.llama_n_seq_max(self._ctx.ctx)

    def n_vocab(self) -> int:
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_vocab")
        return lib.llama_n_vocab(self._model.model)

    def n_ctx_train(self) -> int:
        """Return the native context length of this model"""
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.n_ctx_train'
        )
        return lib.llama_n_ctx_train(self._model.model)

    def n_embd(self) -> int:
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_embd")
        return lib.llama_n_embd(self._model.model)

    def n_layer(self) -> int:
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_layer")
        return lib.llama_n_layer(self._model.model)

    def n_head(self) -> int:
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_head")
        return lib.llama_n_head(self._model.model)

    def pooling_type(self) -> int:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.pooling_type")
        return lib.llama_pooling_type(self._ctx.ctx)

    def vocab_type(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.vocab_type"
        )
        return lib.llama_vocab_type(self._model.model)

    def rope_type(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.rope_type"
        )
        return lib.llama_rope_type(self._model.model)

    def rope_freq_scale_train(self) -> float:
        null_ptr_check(
            self._model.model,
            "self._model.model",
            "Llama.rope_freq_scale_train"
        )
        return lib.llama_rope_freq_scale_train(self._model.model)

    def model_size(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_size"
        )
        return lib.llama_model_size(self._model.model)

    def model_n_params(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_n_params"
        )
        return lib.llama_model_n_params(self._model.model)

    def model_has_encoder(self) -> bool:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_has_encoder"
        )
        return lib.llama_model_has_encoder(self._model.model)

    def model_has_decoder(self) -> bool:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_has_decoder"
        )
        return lib.llama_model_has_decoder(self._model.model)

    def model_is_recurrent(self) -> bool:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.model_is_recurrent"
        )
        return lib.llama_model_is_recurrent(self._model.model)

    def kv_cache_clear(self) -> None:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_clear")
        lib.llama_kv_cache_clear(self._ctx.ctx)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        if seq_id != 0:
            raise NotImplementedError(
                f'Llama.kv_cache_seq_rm: seq_id {seq_id} != 0; this is not yet '
                f'supported'
            )
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_rm")
        return lib.llama_kv_cache_seq_rm(self._ctx.ctx, 0, p0, p1)

    def kv_cache_seq_cp(
        self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int
    ) -> None:
        raise NotImplementedError(
            f'Llama.kv_cache_seq_cp is not yet implemented'
        )
        # null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_cp")
        # lib.llama_kv_cache_seq_cp(self._ctx.ctx, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int) -> None:
        raise NotImplementedError(
            f'Llama.kv_cache_seq_keep is not yet implemented'
        )
        # null_ptr_check(
        #     self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_keep"
        # )
        # lib.llama_kv_cache_seq_keep(self._ctx.ctx, seq_id)

    def kv_cache_seq_add(
        self, seq_id: int, p0: int, p1: int, delta: int
    ) -> None:
        raise NotImplementedError(
            f'Llama.kv_cache_seq_add is not yet implemented'
        )
        # null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_add")
        # lib.llama_kv_cache_seq_add(self._ctx.ctx, seq_id, p0, p1, delta)

    def kv_cache_seq_div(self, seq_id: int, p0: int, p1: int, d: int) -> None:
        raise NotImplementedError(
            f'Llama.kv_cache_seq_div is not yet implemented'
        )
        # null_ptr_check(
        #     self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_div"
        # )
        # lib.llama_kv_cache_seq_div(self._ctx.ctx, seq_id, p0, p1, d)

    def kv_cache_seq_pos_max(self, seq_id: int) -> int:
        raise NotImplementedError(
            f'Llama.kv_cache_seq_pos_max is not yet implemented'
        )
        # null_ptr_check(
        #     self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_pos_max"
        # )
        # return lib.llama_kv_cache_seq_pos_max(self._ctx.ctx, seq_id)

    def kv_cache_defrag(self) -> None:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_defrag")
        lib.llama_kv_cache_defrag(self._ctx.ctx)

    def kv_cache_update(self) -> None:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_update")
        lib.llama_kv_cache_update(self._ctx.ctx)

    def kv_cache_can_shift(self) -> bool:
        null_ptr_check(
            self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_can_shift"
        )
        return lib.llama_kv_cache_can_shift(self._ctx.ctx)

    def decode(self, batch: lib.llama_batch) -> None:
        """
        Decode a batch of tokens

        - batch:
            The `llama_batch` to decode
        """
        # NOTE: this function does not return the return code because if this
        #       function completes, the return code must be 0
        null_ptr_check(self._ctx.ctx, 'self._ctx.ctx', 'Llama.decode')
        null_ptr_check(batch, 'batch', 'Llama.decode')
        ret = lib.llama_decode(self._ctx.ctx, batch)
        if ret < 0:
            raise RuntimeError(
                f'Llama.decode: llama_decode failed with return code {ret}; '
                f'the KV cache state is restored to the state before this call'
            )
        elif ret == 1:
            raise RuntimeError(
                f'Llama.decode: llama_decode could not find a KV slot for the '
                f'batch (try reducing the size of the batch or increase the '
                f'context)'
            )
        elif ret > 1:
            raise RuntimeError(
                f'Llama.decode: llama_decode returned {ret}'
            )
        else:
            # print_info(
            #     f'Llama.decode: llama_decode success'
            # )
            pass

    def n_threads(self) -> int:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_threads")
        return lib.llama_n_threads(self._ctx.ctx)

    def n_threads_batch(self) -> int:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_threads_batch")
        return lib.llama_n_threads_batch(self._ctx.ctx)

    def get_logits(self) -> lib.ptr:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.get_logits")
        return lib.llama_get_logits(self._ctx.ctx)

    def get_logits_ith(self, i: int) -> lib.ptr:
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.get_logits_ith")
        return lib.llama_get_logits_ith(self._ctx.ctx, i)

    def token_get_score(self, token: int) -> float:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_get_score"
        )
        return lib.llama_token_get_score(self._model.model, token)

    def token_is_eog(self, token: int) -> bool:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_is_eog"
        )
        return lib.llama_token_is_eog(self._model.model, token)

    def token_is_control(self, token: int) -> bool:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_is_control"
        )
        return lib.llama_token_is_control(self._model.model, token)

    def token_bos(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_bos"
        )
        return lib.llama_token_bos(self._model.model)

    def token_eos(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_eos"
        )
        return lib.llama_token_eos(self._model.model)

    def token_eot(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_eot"
        )
        return lib.llama_token_eot(self._model.model)

    def token_cls(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_cls"
        )
        return lib.llama_token_cls(self._model.model)

    def token_sep(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_sep"
        )
        return lib.llama_token_sep(self._model.model)

    def token_nl(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_nl"
        )
        return lib.llama_token_nl(self._model.model)

    def token_pad(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_pad"
        )
        return lib.llama_token_pad(self._model.model)

    def add_bos_token(self) -> bool:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.add_bos_token"
        )
        return lib.llama_add_bos_token(self._model.model)

    def add_eos_token(self) -> bool:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.add_eos_token"
        )
        return lib.llama_add_eos_token(self._model.model)

    def token_fim_pre(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_pre"
        )
        return lib.llama_token_fim_pre(self._model.model)

    def token_fim_suf(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_suf"
        )
        return lib.llama_token_fim_suf(self._model.model)

    def token_fim_mid(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_mid"
        )
        return lib.llama_token_fim_mid(self._model.model)

    def token_fim_pad(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_pad"
        )
        return lib.llama_token_fim_pad(self._model.model)

    def token_fim_rep(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_rep"
        )
        return lib.llama_token_fim_rep(self._model.model)

    def token_fim_sep(self) -> int:
        null_ptr_check(
            self._model.model, "self._model.model", "Llama.token_fim_sep"
        )
        return lib.llama_token_fim_sep(self._model.model)
    
    def tokenize(
        self,
        text_bytes: bytes,
        n_tokens_max: int,
        add_special: bool,
        parse_special: bool,
    ) -> list[int]:
        """
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
        tokens_buf = (ctypes.c_int32 * n_tokens_max)()
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.tokenize'
        )
        n_tokens = lib.llama_tokenize(
            model=self._model.model,
            text=text_bytes,
            text_len=len(text_bytes),
            tokens=tokens_buf,
            n_tokens_max=n_tokens_max,
            add_special=add_special,
            parse_special=parse_special
        )
        if n_tokens < 0:
            raise ValueError(
                f'_LlamaModel.tokenize: n_tokens value {-n_tokens} exceeds '
                f'n_tokens_max value {n_tokens_max}'
            )
        ret = list(tokens_buf[:n_tokens])
        del tokens_buf
        return ret

    def token_to_piece(self, token: int, special: bool) -> bytes:
        """
        Convert a single token ID into utf-8 bytes

        - special:
            If True, special tokens are rendered in the output
        """
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.token_to_piece'
        )
        n_bytes = lib.llama_token_to_piece(
            model=self._model.model,
            token=token,
            buf=self._detok_buffer,
            length=_MAX_SINGLE_TOKEN_TEXT_LENGTH,
            lstrip=0, # skip up to 'lstrip' leading spaces
            special=special
        )
        if n_bytes > _MAX_SINGLE_TOKEN_TEXT_LENGTH:
            raise ValueError(
                f"_LlamaModel.token_to_piece: the token with ID {token} "
                f"requires a buffer of size {n_bytes}, but the maximum "
                f"buffer size is {_MAX_SINGLE_TOKEN_TEXT_LENGTH}"
            )
        # NOTE: do not just do str_buf.value.decode() because the token could
        #       possibly be a part of a utf-8 bytestring, but not a valid utf-8
        #       string itself. let the caller handle this
        return self._detok_buffer.raw[:n_bytes]

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
        # this function is just like token_to_piece but in a loop
        detok_bytes = b""
        null_ptr_check(
            self._model.model, 'self._model.model', 'Llama.detokenize'
        )
        for token in tokens:
            n_bytes = lib.llama_token_to_piece(
                model=self._model.model,
                token=token,
                buf=self._detok_buffer,
                length=_MAX_SINGLE_TOKEN_TEXT_LENGTH,
                lstrip=0, # skip up to 'lstrip' leading spaces
                special=special
            )
            if n_bytes > _MAX_SINGLE_TOKEN_TEXT_LENGTH:
                raise ValueError(
                    f"_LlamaModel.detokenize: the token with ID {token} "
                    f"requires a buffer of size {n_bytes}, but the maximum "
                    f"buffer size is {_MAX_SINGLE_TOKEN_TEXT_LENGTH}"
                )
            detok_bytes += self._detok_buffer.raw[:n_bytes]
        return detok_bytes
    
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
        return -lib.llama_tokenize(
            model=self._model.model,
            text=text_bytes,
            text_len=len(text_bytes),
            tokens=NULL,
            n_tokens_max=0,
            add_special=add_special,
            parse_special=parse_special
        )

    def eval() -> list[int]:
        pass # TODO
    
    def first_valid_pos(self, tokens: Iterable[int]) -> int:
        """
        Given a list of tokens, and using `Llama.context_tokens`, find the first
        valid `Llama.pos`

        In other words, return longest common prefix length between the two
        iterables of tokens

        Returns -1 if none of the tokens match
        """
        i = -1
        for c, t in zip(self.context_tokens, tokens):
            if c == t:
                i += 1
            else:
                break
        return i

    def reset(self) -> None:
        self.kv_cache_clear()
        self.pos = 0
        self.context_tokens = []

#
# End of functions / Begin test
#

if __name__ == '__main__':

    # Handy-dandy basic test of wrappers
    # Assumes model.gguf is available in the current working directory

    test_model_path = '/Users/dylan/Documents/AI/models/Meta-Llama-3.1-8B-Instruct-q8_0-q6_K.gguf'
    #test_model_path = '/Users/dylan/Documents/AI/models/Llama-3.2-1B-q8_0-q8_0.gguf'

    if not os.path.exists(test_model_path):
        raise FileNotFoundError(f'the file {test_model_path!r} was not found')
    
    this_module_name = os.path.splitext(os.path.basename(__file__))[0]

    def test_print(text: str) -> None:
        print_verbose(f'{this_module_name} test: {text}')

    print("-" * 80)

    TestLlama = Llama(
        path_model=test_model_path,
        n_gpu_layers=-1,
        use_mmap=True,
        use_mlock=False,
        n_ctx=8192,
        offload_kqv=True,
        flash_attn=True
    )

    print("-" * 80)
    #chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'
    chktxt = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nHello, tell me a short story about two potatoes in love.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    test_print(f'prompt:\n\n{chktxt!r}')
    tokens = TestLlama.tokenize(chktxt.encode(), n_tokens_max=1024, add_special=True, parse_special=True)
    print()
    test_print(f'tokenized prompt:\n\n{tokens!r}')
    detok = TestLlama.detokenize(tokens, special=True).decode()
    print()
    test_print(f'detokenized prompt:\n\n{detok!r}')
    print("-" * 80)

    print(
        f"num prompt characters - {len(chktxt)}\n"
        f"num prompt tokens ----- {len(tokens)}\n"
        f"num detok characters -- {len(detok)}\n"
        f"exact string match? --- {'yes' if chktxt == detok else 'no'}"
    )
    print("-" * 80)
    test_print('using tokenized prompt as input for eval')
    output_tokens = TestLlama.eval(tokens, n_predict=-1, logits_all=False, stops=[])
    print("-" * 80)
    print(TestLlama.detokenize(output_tokens, special=True).decode())
    print("-" * 80)
    TestLlama.free()
    print("-" * 80)
