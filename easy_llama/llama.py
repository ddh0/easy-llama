# llama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""This file provides a high-level Python interface to LLAMA_API ("libllama")."""

import re
import os
import sys
import time
import tqdm
import struct
import ctypes
import asyncio
import threading

import numpy as np

from .utils    import (
    null_ptr_check, softmax, suppress_output, _SupportsWriteAndFlush, ptr, log, ez_decode,
    log_verbose, get_verbose, set_verbose, log_debug, get_debug 
)
from .sampling import SamplerParams, SamplerPreset
from .libllama import _internals, GGUFValueType
from typing    import Optional, Iterable, Union
from io        import BufferedReader

from . import libllama as lib

#
# Constants, etc.
#

PROGRESS_BAR_N_BATCHES = 16
"""Show a tqdm progress bar if processing at least this many batches to be decoded"""

PROGRESS_BAR_N_TOKENS = 20480
"""Show a tqdm progress bar if processing at least this many tokens to be decoded"""

PROGRESS_BAR_N_TOKENS_MIN = 8192
"""Never show a tqdm progress bar if there are fewer than this many tokens to be decoded"""

_SUPPORTED_KV_TYPES = [
    lib.GGMLType.GGML_TYPE_F32,   # lib only supports static types, not
    lib.GGMLType.GGML_TYPE_F16,   # k-types
    lib.GGMLType.GGML_TYPE_Q8_0,
    lib.GGMLType.GGML_TYPE_Q5_1,  # BF16 is also sometimes supported, but not
    lib.GGMLType.GGML_TYPE_Q5_0,  # always, and offers no benefit compared
    lib.GGMLType.GGML_TYPE_Q4_1,  # to F16, so is not included here
    lib.GGMLType.GGML_TYPE_Q4_0
]

_DEFAULT_KV_TYPE = lib.GGMLType.GGML_TYPE_F16

_cpu_count = None

#
# Functions
#

def _init_backend_if_needed() -> None:

    # if already initialized, no need to do anything
    if lib._BACKEND_INIT:
        return
    
    global _cpu_count
    _cpu_count = int(os.cpu_count())
    
    # most cases
    if sys.byteorder == 'little':
        log_debug("host is little-endian")
    # rare
    elif sys.byteorder == 'big':
        log("host is big-endian, please ensure your GGUF file is also big-endian", 2)
    # extremely rare, maybe impossible?
    else:
        raise OSError(
            f"unexpected value for sys.byteorder: {sys.byteorder!r}; expected 'little' for "
            f"little-endian host or 'big' for big-endian host"
        )
    
    # actually load the backend
    with suppress_output(disable=get_verbose()):
        lib.llama_backend_init()

def _kv_type_to_str(t: int) -> str:
    # can't use match statement because we support Py3.9
    if t == lib.GGMLType.GGML_TYPE_F32:
        return "F32"
    elif t == lib.GGMLType.GGML_TYPE_BF16:
        return "BF16"
    elif t == lib.GGMLType.GGML_TYPE_F16:
        return "F16"
    elif t == lib.GGMLType.GGML_TYPE_Q8_0:
        return "Q8_0"
    elif t == lib.GGMLType.GGML_TYPE_Q5_1:
        return "Q5_1"
    elif t == lib.GGMLType.GGML_TYPE_Q5_0:
        return "Q5_0"
    elif t == lib.GGMLType.GGML_TYPE_Q4_1:
        return "Q4_1"
    elif t == lib.GGMLType.GGML_TYPE_Q4_0:
        return "Q4_0"
    else:
        return str(t) # integer as string

# NOTE: the optimal n_threads value (for text generation) is equal to the number of physical
#       cores (for homogenous CPUs) or to the number of performance cores (for heterogenous
#       CPUs)
#
#       the optimal n_threads_batch value (for prompt processing) is equal to the total number
#       of logical cores, regardless of their type
#
#       the following two functions are not universally optimal, but provide a reasonable
#       default number of threads for most machines

def _get_optimal_n_threads() -> int:
    global _cpu_count
    return max(_cpu_count//2, 1)

def _get_optimal_n_threads_batch() -> int:
    global _cpu_count
    return _cpu_count

def _calculate_rope_freq_base(
    n_ctx_train: int,
    n_ctx_load: int,
    rope_freq_base_train: Optional[float]
) -> float:
    """Returns the rope_freq_base value at which a model should be loaded"""

    # n_ctx does not exceed n_ctx_train - simply return native value

    if n_ctx_load <= n_ctx_train:
        if rope_freq_base_train is None:
            return 0.0
        else:
            return rope_freq_base_train
    
    # n_ctx exceeds n_ctx_train, but native value is unknown, so automatic
    # adjustment cannot be applied - show error and return 0.0

    if rope_freq_base_train in [None, 0.0]:
        log(
            f'n_ctx value {n_ctx_load} > n_ctx_train value {n_ctx_train}, and automatic '
            f'rope_freq_base adjustment is not supported for this model; model loading might '
            f'fail, or the model might not work correctly', 3
        )
        return 0.0
    
    # n_ctx exceeds n_ctx_train, and native value is known, so automatic
    # adjustment can be applied - show warning and return adjusted value

    # standard formula -- proportional increase
    adjusted_rope_freq = (n_ctx_load/n_ctx_train)*rope_freq_base_train
    # experimental formula -- slightly above proportional increase
    #adjusted_rope_freq = ((n_ctx_load/n_ctx_train)**(2**(1/4)))*rope_freq_base_train
    
    log(
        f"n_ctx value {n_ctx_load} exceeds n_ctx_train value {n_ctx_train}; using adjusted "
        f"rope_freq_base value {adjusted_rope_freq}, native value is {rope_freq_base_train}; "
        f"model will function with potentially degraded output quality", 2
    )

    return adjusted_rope_freq

def _round_n_ctx(n_ctx: int, n_ctx_train: int) -> int:
    if n_ctx % 512 == 0:
        return n_ctx
    else:
        rounded = (n_ctx + 511) // 512 * 512
        # do not round beyond n_ctx_train if not already exceeded
        if (rounded > n_ctx_train) and (n_ctx <= n_ctx_train):
            return n_ctx_train
        else:
            return rounded

def _batches_with_progress_bar(batches: list[list[int]]) -> Union[tqdm.tqdm, list[list[int]]]:
    """Wrap this around an iterable of batches to show a progress bar if there are over
    `PROGRESS_BAR_N_BATCHES` batches or `PROGRESS_BAR_N_TOKENS` tokens. Never show a progress
    bar if there are fewer then `PROGRESS_BAR_N_TOKENS_MIN` tokens."""

    n_batches = len(batches)
    n_tokens = sum(len(batch) for batch in batches)

    if n_tokens < PROGRESS_BAR_N_TOKENS_MIN:
        if (n_tokens > PROGRESS_BAR_N_TOKENS) or (n_batches > PROGRESS_BAR_N_BATCHES):
            return tqdm.tqdm(batches, desc='decoding input batches', unit="batch")
    return batches

def split_tokens_into_batches(tokens: list[int], n_batch: int) -> list[list[int]]:
    """Split a list of tokens into batches of size `n_batch`."""
    return [tokens[i:i + n_batch] for i in range(0, len(tokens), n_batch)]

#
# Exceptions and other classes
#

class ExceededContextLengthException(Exception):
    """Exception raised when an input exceeds a model's context length"""

class _LlamaStopwatch:
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
        self.wall_start_time = None
        self.generic_start_time = None
        self.pp_elapsed_time = 0
        self.tg_elapsed_time = 0
        self.wall_elapsed_time = 0
        self.generic_elapsed_time = 0
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
    
    def start_wall_time(self):
        """Start wall-time stopwatch"""
        self.wall_start_time = time.time_ns()

    def stop_wall_time(self):
        """Stop wall-time stopwatch"""
        if self.wall_start_time is not None:
            self.wall_elapsed_time += time.time_ns() - self.wall_start_time
            self.wall_start_time = None

    def start_generic(self):
        """Start generic stopwatch (not shown in print_stats)"""
        self.generic_start_time = time.time_ns()
    
    def stop_generic(self):
        """Stop generic stopwatch"""
        if self.generic_start_time is not None:
            self.generic_elapsed_time += time.time_ns() - self.generic_start_time
            self.generic_start_time = None
    
    def get_elapsed_time_pp(self) -> int:
        """Total nanoseconds elapsed during prompt processing"""
        return self.pp_elapsed_time

    def get_elapsed_time_tg(self) -> int:
        """Total nanoseconds elapsed during text generation"""
        return self.tg_elapsed_time
    
    def get_elapsed_wall_time(self) -> int:
        """Total wall-time nanoseconds elapsed"""
        return self.wall_elapsed_time

    def get_elapsed_time_generic(self) -> int:
        """Total generic nanoseconds elapsed"""
        return self.generic_elapsed_time

    def increment_pp_tokens(self, n: int):
        if n < 0:
            raise ValueError('negative increments are not allowed')
        self.n_pp_tokens += n

    def increment_tg_tokens(self, n: int):
        if n < 0:
            raise ValueError('negative increments are not allowed')
        self.n_tg_tokens += n

    def reset(self):
        """Reset the stopwatch to its original state"""
        self.pp_start_time = None
        self.tg_start_time = None
        self.wall_start_time = None
        self.generic_start_time = None
        self.pp_elapsed_time = 0
        self.tg_elapsed_time = 0
        self.wall_elapsed_time = 0
        self.generic_elapsed_time = 0
        self.n_pp_tokens = 0
        self.n_tg_tokens = 0

    def print_stats(self):
        """Print performance statistics using current stopwatch state
        
        #### NOTE:
        The `n_tg_tokens` value will be equal to the number of calls to
        llama_decode which have a batch size of 1, which is technically not
        always equal to the number of tokens generated - it may be off by one."""

        print(f"\n", end='', file=sys.stderr, flush=True)

        if self.n_pp_tokens + self.n_tg_tokens == 0:
            log(f'print_stats was called but no tokens were processed or generated', 4)

        if self.n_pp_tokens > 0:
            pp_elapsed_ns = self.get_elapsed_time_pp()
            pp_elapsed_ms = pp_elapsed_ns / 1e6
            pp_elapsed_s = pp_elapsed_ns / 1e9
            pp_tps = self.n_pp_tokens / pp_elapsed_s
            log(
                f'prompt processing: {self.n_pp_tokens:>7} tokens in {pp_elapsed_ms:>13.3f}ms '
                f'({pp_tps:>10.2f} tok/s)', 4
            )

        if self.n_tg_tokens > 0:
            tg_elapsed_ns = self.get_elapsed_time_tg()
            tg_elapsed_ms = tg_elapsed_ns / 1e6
            tg_elapsed_s = tg_elapsed_ns / 1e9
            tg_tps = self.n_tg_tokens / tg_elapsed_s
            log(
                f'  text generation: {self.n_tg_tokens:>7} tokens in {tg_elapsed_ms:>13.3f}ms '
                f'({tg_tps:>10.2f} tok/s)', 4
            )
        
        wall_elapsed_ns = self.get_elapsed_wall_time()
        wall_elapsed_ms = wall_elapsed_ns / 1e6
        log(f"        wall time:{' ' * 19}{wall_elapsed_ms:>13.3f}ms", 4)

class QuickGGUFReader:
    # ref: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

    ValueType = Union[str, int, float, bool, list]
    """Type hint for any GGUF metadata value type"""
    
    # the GGUF format versions that this class supports
    SUPPORTED_GGUF_VERSIONS = [2, 3]
    
    # arguments for struct.unpack() based on GGUF value type
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

    # length in bytes for each GGUF value type
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
    def get_single(value_type: GGUFValueType, file: BufferedReader) -> ValueType:
        """Read a single value from an open file"""
        if value_type == GGUFValueType.STRING:
            string_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            value = ez_decode(file.read(string_length))
        else:
            value = QuickGGUFReader.unpack(value_type, file=file)
        return value
    
    @staticmethod
    def load_metadata(path_model: os.PathLike[str] | str) -> dict[str, ValueType]:
        """Given a path to a GGUF file, peek at its header for metadata

        Return a dictionary where all keys are strings, and values can be
        strings, ints, floats, bools, or lists"""

        metadata: dict[str, QuickGGUFReader.ValueType] = {}
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
            
            tensor_count = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            log_debug(f'tensor count: {tensor_count}')
            if version == 3:
                metadata_kv_count = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            elif version == 2:
                metadata_kv_count = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
            for _ in range(metadata_kv_count):
                if version == 3:
                    key_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
                elif version == 2:
                    key_length = 0
                    while key_length == 0:
                        # seek until next key is found
                        key_length = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
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
                        array_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
                    elif version == 2:
                        array_length = QuickGGUFReader.unpack(GGUFValueType.UINT32, file=file)
                        file.read(4) # 4 byte offset for GGUFv2
                    array = [
                        QuickGGUFReader.get_single(array_value_type, file)
                        for _ in range(array_length)
                    ]
                    metadata[ez_decode(key)] = array
                else:
                    value = QuickGGUFReader.get_single(value_type, file)
                    metadata[ez_decode(key)] = value

        return metadata

#
# InferenceLock
#

class InferenceLockException(Exception):
    pass

class _InferenceLock:
    """A context manager which is used to prevent an `ez.Llama` instance from accepting
    more than one generation at a time, which is not supported and can cause a hard crash.

    - Safe if only used synchronously (`__enter__`/`__exit__`)
    - Safe if only used asynchronously (`__aenter__`/`__aexit__`)
    - Not safe for concurrent sync/async"""
    
    def __init__(self):
        self._locked = False
        self._sync_lock = threading.Lock()  # for thread safety
        self._async_lock = asyncio.Lock()   # for async safety

    def __enter__(self):
        with self._sync_lock:
            if self._locked:
                raise InferenceLockException(
                    'sync: failed to acquire InferenceLock (already locked)'
                )
            self._locked = True
        return self

    def __exit__(self, *_):
        with self._sync_lock:
            if not self._locked:
                raise InferenceLockException(
                    'sync: tried to release InferenceLock that is not acquired'
                )
            self._locked = False

    async def __aenter__(self):
        async with self._async_lock:
            if self._locked:
                raise InferenceLockException(
                    'async: failed to acquire InferenceLock (already locked)'
                )
            self._locked = True
        return self

    async def __aexit__(self, *_):
        async with self._async_lock:
            if not self._locked:
                raise InferenceLockException(
                    'async: tried to release InferenceLock that is not acquired'
                )
            self._locked = False

#
# Simple python wrappers
#

class _LlamaModel:
    """Low-level Python wrapper over `llama_model`"""

    def __init__(
        self,
        path_model: str,

        devices:                     Optional[list[ptr]]   = None,
        tensor_buft_override:        Optional[list[ptr]]   = None,
        n_gpu_layers:                Optional[int]         = None,
        split_mode:                  Optional[int]         = None,
        main_gpu:                    Optional[int]         = None,
        tensor_split:                Optional[list[float]] = None,
        rpc_servers:                 Optional[str]         = None,
        # progress_callback & progress_callback_user_data are not supported by easy-llama
        kv_overrides:                Optional[list[ptr]]   = None,
        vocab_only:                  Optional[bool]        = None,
        use_mmap:                    Optional[bool]        = None,
        use_mlock:                   Optional[bool]        = None,
        check_tensors:               Optional[bool]        = None
    ):
        # refuse to load files with incorrect extension
        if not path_model.lower().endswith('.gguf'):
            raise ValueError(
                f"_LlamaModel.__init__: the given path_model {path_model!r} does not end "
                f"in '.gguf'. easy-llama refuses to load from files that do not have the "
                f"correct file extension."
            )
        
        _init_backend_if_needed()
        self.path_model = path_model
        self.params = lib.llama_model_default_params()
        null_ptr_check(self.params, "self.params", "_LlamaModel.__init__")
        if devices is not None:
            self.params.devices = (ctypes.c_void_p * (len(devices) + 1))(*devices, None)
        if tensor_buft_override is not None:
            self.params.tensor_buft_overrides = (
                lib.llama_model_tensor_buft_override_p * (len(tensor_buft_override) + 1)
            )(*tensor_buft_override, None)
        if n_gpu_layers is not None:
            self.params.n_gpu_layers = (
                n_gpu_layers
            ) if n_gpu_layers >= 0 else lib.MAX_OFFLOAD_LAYERS
        if split_mode is not None:
            self.params.split_mode = split_mode
        if main_gpu is not None:
            self.params.main_gpu = main_gpu
        if tensor_split is not None:
            self.params.tensor_split = (ctypes.c_float * len(tensor_split))(*tensor_split)
        if rpc_servers is not None:
            self.params.rpc_servers = rpc_servers.encode('utf-8')
        
        # create the ctypes function pointer instance and store it as an attribute of this
        # `_LlamaModel` to keep it alive
        # def _py_progress_callback(progress: ctypes.c_float, user_data: ptr) -> ctypes.c_bool:
        #     return False
        # self._progress_callback_cfunc_instance = lib.progress_callback_functype(_py_progress_callback)
        # self.params.progress_callback = self._progress_callback_cfunc_instance
        # self.params.progress_callback_user_data = lib.NULLPTR
        
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
        
        # load model
        with suppress_output(disable=get_verbose()):
            self.model = lib.llama_model_load_from_file(path_model, self.params)
        
        null_ptr_check(self.model, "self.model", "_LlamaModel.__init__")
    
    def __del__(self):
        self.free()

    def free(self):
        if self.model is not None:
            with suppress_output(disable=get_verbose()):
                lib.llama_model_free(self.model)
            self.model = None

class _LlamaCtx:
    """Low-level Python wrapper over `llama_context`"""

    def __init__(
        self,
        model: _LlamaModel,

        n_ctx:               Optional[int]     = None,
        n_batch:             Optional[int]     = None,
        n_ubatch:            Optional[int]     = None,
        n_seq_max:           Optional[int]     = None,
        n_threads:           Optional[int]     = None,
        n_threads_batch:     Optional[int]     = None,
        rope_scaling_type:   Optional[int]     = None,
        pooling_type:        Optional[int]     = None,
        attention_type:      Optional[int]     = None,
        rope_freq_base:      Optional[float]   = None,
        rope_freq_scale:     Optional[float]   = None,
        yarn_ext_factor:     Optional[float]   = None,
        yarn_attn_factor:    Optional[float]   = None,
        yarn_beta_fast:      Optional[float]   = None,
        yarn_beta_slow:      Optional[float]   = None,
        yarn_orig_ctx:       Optional[int]     = None,
        defrag_thold:        Optional[float]   = None,
        # cb_eval & cb_eval_user_data are not supported by easy-llama
        type_k:              Optional[int]     = None,
        type_v:              Optional[int]     = None,
        embeddings:          Optional[bool]    = None,
        offload_kqv:         Optional[bool]    = None,
        flash_attn:          Optional[bool]    = None,
        no_perf:             Optional[bool]    = None,
        # abort_callback & abort_callback_data are not supported by easy-llama
        op_offload:          Optional[bool]    = None,
        swa_full:            Optional[bool]    = None
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
        
        # create the ctypes function pointer instance and store it as an attribute of this
        # `_LlamaCtx` to keep it alive
        # def _py_eval_callback(is_eval: bool, user_data: ptr) -> None:
        #     return
        # self._eval_callback_cfunc_instance = lib.eval_callback_functype(_py_eval_callback)
        # self.params.cb_eval = self._eval_callback_cfunc_instance
        # self.params.cb_eval_user_data = lib.NULLPTR

        _k = _DEFAULT_KV_TYPE
        if type_k is not None:
            self.params.type_k = _k = type_k
        _v = _DEFAULT_KV_TYPE
        if type_v is not None:
            self.params.type_v = _v = type_v
        if _k != _v:
            log(
                f'type_k value {_k} != type_v value {_v}; this is rarely '
                f'supported, program may fail', 2
            )
        if _k not in _SUPPORTED_KV_TYPES:
            log(f'type_k value {_k} is unsupported; program may fail', 2)
        if _v not in _SUPPORTED_KV_TYPES:
            log(f'type_v value {_v} is unsupported; program may fail', 2)
        if (not flash_attn) and (_v not in [
            lib.GGMLType.GGML_TYPE_F32, lib.GGMLType.GGML_TYPE_F16, lib.GGMLType.GGML_TYPE_BF16
        ]):
            log(f'V cache quantization requires flash_attn; program may fail', 2)
        if embeddings is not None:
            self.params.embeddings = embeddings
        if offload_kqv is not None:
            self.params.offload_kqv = offload_kqv
        if flash_attn is not None:
            self.params.flash_attn = flash_attn
        if no_perf is not None:
            self.params.no_perf = no_perf
        if op_offload is not None:
            self.params.op_offload = op_offload

        # enable proper SWA support unless explicitly disabled
        # (`swa_full` disables SWA for models which are designed to use it)
        self.params.swa_full = False if swa_full is None else swa_full

        # create the ctypes function pointer instance and store it as an attribute of this
        # `_LlamaCtx` to keep it alive
        # def _py_abort_callback(user_data: ptr) -> ctypes.c_bool:
        #     return False
        # self._abort_callback_cfunc_instance = lib.abort_callback_functype(_py_abort_callback)
        # self.params.abort_callback = self._abort_callback_cfunc_instance
        # self.params.abort_callback_data = lib.NULLPTR
        
        null_ptr_check(model.model, "model.model", "_LlamaCtx.__init__")
        with suppress_output(disable=get_verbose()):
            self.ctx = lib.llama_init_from_model(model.model, self.params)
        null_ptr_check(self.ctx, "self.ctx", "_LlamaCtx.__init__")
    
    def __del__(self):
        self.free()

    def free(self):
        if self.ctx is not None:
            with suppress_output(disable=get_verbose()):
                lib.llama_free(self.ctx)
            self.ctx = None

#
# Llama
#

class Llama:
    """Simplified interface for general-purpose Llama model usage

    The `easy_llama.Llama` class provides a high-level Python interface to
    a llama_model and its associated llama_context.

    Example usage:
    >>> import easy_llama as ez
    >>> MyLlama = ez.Llama('/path/to/model.gguf', n_ctx=8192)
    >>> in_txt = b"The apple doesn't fall far from"
    >>> in_toks = MyLlama.tokenize(in_txt, add_special=True, parse_special=False)
    >>> out_toks = MyLlama.generate(in_toks, n_predict=16)
    >>> out_txt = MyLlama.detokenize(out_toks, special=True)
    >>> out_txt
    b" the tree, as the saying goes, and I think that's especially true when\""""
    
    def __init__(
        self,
        path_model: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 512,
        n_threads: int = 0,
        n_threads_batch: int = 0,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        offload_kqv: bool = False,
        flash_attn: bool = False,
        warmup: bool = False,
        verbose: bool = True,
        vocab_only: bool = True,
        **kwargs
    ):
        """Load a llama model from a file

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
            usage. Values must be between 32 and n_ctx, inclusive.
        - n_threads:
            Number of threads to use for batch size == 1.
        - n_threads_batch:
            Number of threads to use for batch sizes > 1.
        - type_k:
            The `libllama.GGMLType` to use for the K cache. Default is 1 (f16).
            In most cases, this must be the same as `type_v`.
        - type_v:
            The `libllama.GGMLType` to use for the V cache. Default is 1 (f16).
            In most cases, this must be the same as `type_k`. Values other than
            0 and 1 are not compatible with `flash_attn=True`.
        - offload_kqv:
            Whether to offload the K, Q, V caches to the GPU, which can greatly
            improve prompt processing speed at the cost of increased VRAM usage.
            Default is False for compatability reasons. Recommended to set to
            True if possible.
        - flash_attn:
            Whether to use Flash Attention, which decreases memory usage and
            can increase both prompt processing and text generation speed,
            especially at long context lengths. Default is False for compatability reasons.
            Recommended to set to True if possible.
        - warmup:
            Whether to warm-up the model with an empty run. This reduces the
            latency of the first generation at the cost of a slower load time.
        - verbose:
            Print informational output when loading model as well as at
            runtime. Default is True. If set to False, warnings and errors
            will still be shown."""
        
        if not os.path.exists(path_model):
            raise FileNotFoundError(
                f"Llama: the given path_model {path_model!r} does not exist"
            )
        if os.path.isdir(path_model):
            raise IsADirectoryError(
                f"Llama: the given path_model {path_model!r} is a directory, "
                f"not a GGUF file"
            )
        
        set_verbose(verbose)
        
        # peek at metadata from GGUF file header before loading model

        self.metadata = QuickGGUFReader.load_metadata(path_model)
        
        #
        # Load model from file
        #

        self.vocab_only = vocab_only

        self._model = _LlamaModel(
            path_model           = path_model,
            devices              = kwargs.get('devices'),
            tensor_buft_override = kwargs.get('tensor_buft_override'),
            n_gpu_layers         = n_gpu_layers,
            split_mode           = kwargs.get('split_mode'),
            main_gpu             = kwargs.get('main_gpu'),
            tensor_split         = kwargs.get('tensor_split'),
            rpc_servers          = kwargs.get('rpc_servers'),
            kv_overrides         = kwargs.get('kv_overrides'),
            vocab_only           = self.vocab_only,
            use_mmap             = kwargs.get('use_mmap'),
            use_mlock            = kwargs.get('use_mlock'),
            check_tensors        = kwargs.get('check_tensors')
        )
        
        self._vocab = lib.llama_model_get_vocab(self._model.model)
        """A pointer to this model's `llama_vocab`"""
        null_ptr_check(self._vocab, 'self._vocab', 'Llama.__init__')

        if self.vocab_only:
            log('Llama instance has vocab_only=True, most functionality will not be available')
        
        else:

            n_ctx_train = lib.llama_model_n_ctx_train(self._model.model)
            
            # use n_ctx unless it's 0 or negative, in that case use n_ctx_train

            if n_ctx <= 0:
                log_verbose(f'n_ctx value {n_ctx}; using n_ctx_train value {n_ctx_train}')
                _n_ctx = int(n_ctx_train)
            else:
                _n_ctx = int(n_ctx)

            # use rope_freq_base unless it == 0.0, in that case use the native
            # rope_freq_base found in the GGUF metadata
            rope_freq_base = kwargs.get('rope_freq_base', 0.0)

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
            
            _n_threads = n_threads if n_threads > 0 else _get_optimal_n_threads()
            _n_threads_batch = n_threads_batch if n_threads_batch > 0 else (
                _get_optimal_n_threads_batch()
            )

            #
            # New context with model
            #
            
            self._ctx = _LlamaCtx(
                model             = self._model,
                n_ctx             = _n_ctx,
                n_batch           = kwargs.get('n_batch'),
                n_ubatch          = kwargs.get('n_ubatch'),
                n_seq_max         = kwargs.get('n_seq_max'),
                n_threads         = _n_threads,
                n_threads_batch   = _n_threads_batch,
                rope_scaling_type = kwargs.get('rope_scaling_type'),
                pooling_type      = kwargs.get('pooling_type'),
                attention_type    = kwargs.get('attention_type'),
                rope_freq_base    = _rope_freq_base,
                rope_freq_scale   = kwargs.get('rope_freq_scale'),
                yarn_ext_factor   = kwargs.get('yarn_ext_factor'),
                yarn_attn_factor  = kwargs.get('yarn_attn_factor'),
                yarn_beta_fast    = kwargs.get('yarn_beta_fast'),
                yarn_beta_slow    = kwargs.get('yarn_beta_slow'),
                yarn_orig_ctx     = kwargs.get('yarn_orig_ctx'),
                defrag_thold      = kwargs.get('defrag_thold'),
                type_k            = type_k,
                type_v            = type_v,
                embeddings        = kwargs.get('embeddings'),
                offload_kqv       = offload_kqv,
                flash_attn        = flash_attn,
                no_perf           = kwargs.get('no_perf'),
                op_offload        = kwargs.get('op_offload'),
                swa_full          = kwargs.get('swa_full')
            )

            #
            # Display warnings about n_ctx if necessary
            #

            actual_n_ctx = self.n_ctx()
            requested_n_ctx = _n_ctx

            if actual_n_ctx != requested_n_ctx:
                log(
                    f"requested n_ctx value differs from actual n_ctx value; "
                    f"requested {requested_n_ctx}, actual {actual_n_ctx}", 2
                )
            if actual_n_ctx < 512:
                log(
                    f"n_ctx value {actual_n_ctx} is less than 512, which can "
                    f"sometimes cause problems with llama.cpp - consider "
                    f"increasing it to at least 512", 2
                )
            if actual_n_ctx % 512 != 0:
                log(
                    f"n_ctx value {actual_n_ctx} is not divisible by 512, which "
                    f"can sometimes cause problems with llama.cpp - consider "
                    f"changing it to "
                    f"{_round_n_ctx(actual_n_ctx, n_ctx_train)}", 2
                )
            # warn about default context length
            if actual_n_ctx == 512:
                log(
                    f'you are using the default n_ctx value {actual_n_ctx}, which '
                    f'is very small. increase n_ctx as needed to support longer '
                    f'inputs and outputs (this model supports up to {self.n_ctx_train()})', 2
                )
            
            self._stopwatch = _LlamaStopwatch()

            #
            # Store immutable Llama metadata as attributes for faster access internally
            #

            self._name                  = self.name()
            self._n_ctx                 = self.n_ctx()
            self._n_batch               = self.n_batch()
            self._n_ubatch              = self.n_ubatch()
            self._n_seq_max             = self.n_seq_max()
            self._n_vocab               = self.n_vocab()
            self._n_ctx_train           = self.n_ctx_train()
            self._n_embd                = self.n_embd()
            self._n_layer               = self.n_layer()
            self._n_head                = self.n_head()
            self._n_head_kv             = self.n_head_kv()
            self._n_swa                 = self.n_swa()
            self._vocab_type            = self.vocab_type()
            self._rope_type             = self.rope_type()
            self._rope_freq_scale_train = self.rope_freq_scale_train()
            self._model_size_bytes      = self.model_size_bytes()
            self._chat_template         = self.chat_template()
            self._n_params              = self.n_params()
            self._bpw                   = self.bpw()
            self._has_encoder           = self.has_encoder()
            self._has_decoder           = self.has_decoder()
            self._is_recurrent          = self.is_recurrent()
            self._token_bos             = self.token_bos()
            self._token_eos             = self.token_eos()
            self._token_eot             = self.token_eot()
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
            """A list of all tokens in the vocab that are marked as EOG (End-Of-Generation)"""

            # internal use only - the default SamplerParams with this model
            self._default_sampler_params = SamplerParams(self)

            self.pos = 0
            """The current position of the model within the context window"""

            self.context_tokens = []
            """A list of all tokens currently in the context window"""

            self._lock = _InferenceLock()

            if warmup:
                self.warmup()

        # End of Llama.__init__
    
    def __repr__(self) -> str:
        if self.vocab_only:
            return (
                f"Llama("
                f"path_model={self._model.path_model!r}, "
                f"vocab_only={self.vocab_only}"
                f")"
            )
        return (
            f"Llama("
            f"path_model={self._model.path_model!r}, "
            f"n_gpu_layers={self._model.params.n_gpu_layers}, "
            f"n_ctx={self._n_ctx}, "
            f"type_k={self._ctx.params.type_k}, "
            f"type_v={self._ctx.params.type_v}, "
            f"offload_kqv={self._ctx.params.offload_kqv}, "
            f"flash_attn={self._ctx.params.flash_attn}, "
            f"vocab_only={self.vocab_only}"
            f")"
        )
    
    def free(self):
        """Deallocate the context and model"""
        # must free the context first, then the model
        if not self.vocab_only:
            self._ctx.free()
        self._model.free()
    
    def _validate_model_state(self) -> None:
        """Ensure `llama_model`, `llama_vocab` and `llama_context` are not NULL and validate
        `Llama.pos`"""
        null_ptr_check(self._model.model, 'self._model.model', '_validate_model_state')
        null_ptr_check(self._vocab,       'self._vocab',       '_validate_model_state')

        if not self.vocab_only:
            null_ptr_check(self._ctx.ctx,     'self._ctx.ctx',     '_validate_model_state')
        
            _n_context_tokens = len(self.context_tokens)
            _pos = self.pos

            if _pos < 0:
                self.reset()
                log_verbose(
                    f'self.pos value was {self.pos} - clamping to 0. the KV cache has been reset.',
                    2
                )
            if _pos != _n_context_tokens:
                self.reset()
                log_verbose(
                    f'n_context_tokens {_n_context_tokens} did not match self.pos {_pos}. the KV '
                    f'cache has been reset.', 2
                )
            if not hasattr(self, '_default_sampler_params'):
                self._default_sampler_params = SamplerParams(self)
                log_verbose(
                    "Llama._default_sampler_params was destroyed but has been recreated", 2
                )
    
    def warmup(self) -> None:
        """Warm-up the model. This also resets the model state."""
        if self.vocab_only:
            raise RuntimeError('Llama.warmup: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.warmup")

        with suppress_output(disable=get_verbose()):
            self.reset()

        lib.llama_set_warmup(self._ctx.ctx, True)

        log_verbose('warmup: single token decode ...')
        with self._lock:
            _internals.decode_tg(self._ctx.ctx, 0, 0)
        
        # This section decodes a full batch of tokens, but is probably unnecessary.
        #
        # with suppress_output(disable=get_verbose()):
        #     self.reset()
        # 
        # log_verbose('warmup: full batch decode ...')
        # with self._lock:
        #     _internals.decode_pp(self._ctx.ctx, 0, [0] * self._n_batch, self._n_batch)

        lib.llama_set_warmup(self._ctx.ctx, False)

        with suppress_output(disable=get_verbose()):
            self.reset()
        
        log_verbose('warmup: done')

    def n_ctx(self) -> int:
        """Get the current context length"""
        if self.vocab_only:
            raise RuntimeError('Llama.n_ctx: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_ctx")
        return lib.llama_n_ctx(self._ctx.ctx)

    def n_batch(self) -> int:
        """Get the current batch size"""
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_batch")
        if self.vocab_only:
            raise RuntimeError('Llama.n_batch: this instance is vocab-only')
        return lib.llama_n_batch(self._ctx.ctx)

    def n_ubatch(self) -> int:
        """Get the current micro-batch size"""
        if self.vocab_only:
            raise RuntimeError('Llama.n_ubatch: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_batch")
        return lib.llama_n_ubatch(self._ctx.ctx)

    def n_seq_max(self) -> int:
        """Get the max number of sequences"""
        if self.vocab_only:
            raise RuntimeError('Llama.n_seq_max: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_seq_max")
        return lib.llama_n_seq_max(self._ctx.ctx)

    def n_vocab(self) -> int:
        """Get the vocab size"""
        null_ptr_check(self._vocab, "self._vocab", "Llama.n_vocab")
        return lib.llama_vocab_n_tokens(self._vocab)

    def n_ctx_train(self) -> int:
        """Get the trained context length"""
        null_ptr_check(self._model.model, 'self._model.model', 'Llama.n_ctx_train')
        return lib.llama_model_n_ctx_train(self._model.model)

    def n_embd(self) -> int:
        """Get the embedding size"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_embd")
        return lib.llama_model_n_embd(self._model.model)

    def n_layer(self) -> int:
        """Get the number of layers"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_layer")
        return lib.llama_model_n_layer(self._model.model)

    def n_head(self) -> int:
        """Get the number of attention heads"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_head")
        return lib.llama_model_n_head(self._model.model)
    
    def n_head_kv(self) -> int:
        """Get the number of KV heads"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_head_kv")
        return lib.llama_model_n_head_kv(self._model.model)

    def pooling_type(self) -> int:
        """Get the pooling type used by the context"""
        if self.vocab_only:
            raise RuntimeError('Llama.pooling_type: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.pooling_type")
        return lib.llama_pooling_type(self._ctx.ctx)

    def n_swa(self) -> int:
        """Get the sliding window size, for models which use SWA."""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_swa")
        return lib.llama_model_n_swa(self._model.model)

    def vocab_type(self) -> int:
        """Get the vocab type"""
        null_ptr_check(self._vocab, "self._vocab", "Llama.vocab_type")
        return lib.llama_vocab_type(self._vocab)

    def rope_type(self) -> int:
        """Get the RoPE type"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.rope_type")
        return lib.llama_model_rope_type(self._model.model)

    def rope_freq_scale_train(self) -> float:
        """Get the trained RoPE frequency scale"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.rope_freq_scale_train")
        return lib.llama_model_rope_freq_scale_train(self._model.model)

    def model_size_bytes(self) -> int:
        """Get the total size of all tensors in the model, in bytes"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.model_size_bytes")
        return lib.llama_model_size(self._model.model)
    
    def chat_template(self) -> Optional[str]:
        """Get the model's built-in chat template string. Returns None if not available."""
        null_ptr_check(self._model.model, "self._model.model", "Llama.chat_template")
        return lib.llama_model_chat_template(self._model.model, name=None)

    def n_params(self) -> int:
        """Get the total number of parameters in the model"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.n_params")
        return lib.llama_model_n_params(self._model.model)

    def has_encoder(self) -> bool:
        """If the model has an encoder"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.has_encoder")
        return lib.llama_model_has_encoder(self._model.model)

    def has_decoder(self) -> bool:
        """If the model has a decoder"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.has_decoder")
        return lib.llama_model_has_decoder(self._model.model)

    def is_recurrent(self) -> bool:
        """If the model is recurrent"""
        null_ptr_check(self._model.model, "self._model.model", "Llama.is_recurrent")
        return lib.llama_model_is_recurrent(self._model.model)
    
    #
    # KV cache management methods
    #

    def kv_cache_clear(self) -> None:
        """Clear the KV cache"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_clear: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_clear")
        lib.llama_kv_self_clear(self._ctx.ctx)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        """Remove tokens from a sequence in the KV cache"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_seq_rm: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_rm")
        return lib.llama_kv_self_seq_rm(self._ctx.ctx, seq_id, p0, p1)

    def kv_cache_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int) -> None:
        """Copy tokens between sequences in the KV cache"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_seq_cp: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_cp")
        lib.llama_kv_self_seq_cp(self._ctx.ctx, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int) -> None:
        """Remove all tokens except for the ones in this sequence"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_seq_keep: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_keep")
        lib.llama_kv_self_seq_keep(self._ctx.ctx, seq_id)

    def kv_cache_seq_add(self, seq_id: int, p0: int, p1: int, delta: int) -> None:
        """Add relative position "delta" to the tokens"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_seq_add: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_add")
        lib.llama_kv_self_seq_add(self._ctx.ctx, seq_id, p0, p1, delta)

    def kv_cache_seq_div(self, seq_id: int, p0: int, p1: int, d: int) -> None:
        """Integer division of the positions by factor of `d > 1`"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_seq_div: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_div")
        lib.llama_kv_self_seq_div(self._ctx.ctx, seq_id, p0, p1, d)
    
    def kv_cache_seq_pos_min(self, seq_id: int) -> int:
        """Returns the earliest valid position in the KV cache for the specified sequence
        (relevant for models which use SWA)"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_seq_pos_min: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_pos_max")
        return lib.llama_kv_self_seq_pos_min(self._ctx.ctx, seq_id)

    def kv_cache_seq_pos_max(self, seq_id: int) -> int:
        """Returns the largest position present in the KV cache for the specified sequence"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_seq_pos_max: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_seq_pos_max")
        return lib.llama_kv_self_seq_pos_max(self._ctx.ctx, seq_id)

    def kv_cache_can_shift(self) -> bool:
        """Check if the context supports KV cache shifting"""
        if self.vocab_only:
            raise RuntimeError('Llama.kv_cache_can_shift: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.kv_cache_can_shift")
        return lib.llama_kv_self_can_shift(self._ctx.ctx)

    def n_threads(self) -> int:
        """Get the number of threads used for batch size == 1"""
        if self.vocab_only:
            raise RuntimeError('Llama.n_threads: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_threads")
        return lib.llama_n_threads(self._ctx.ctx)

    def n_threads_batch(self) -> int:
        """Get the number of threads used for batch sizes > 1"""
        if self.vocab_only:
            raise RuntimeError('Llama.n_threads_batch: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, "self._ctx.ctx", "Llama.n_threads_batch")
        return lib.llama_n_threads_batch(self._ctx.ctx)

    def token_get_score(self, token: int) -> float:
        """Get the score of a token"""
        null_ptr_check(self._vocab, "self._vocabl", "Llama.token_get_score")
        return lib.llama_vocab_get_score(self._vocab, token)

    def token_is_eog(self, token: int) -> bool:
        """If the token is marked as EOG (End-Of-Generation)"""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_is_eog")
        return lib.llama_vocab_is_eog(self._vocab, token)

    def token_is_control(self, token: int) -> bool:
        """If the token is marked as a control token"""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_is_control")
        return lib.llama_vocab_is_control(self._vocab, token)

    def token_bos(self) -> Optional[int]:
        """Get the BOS (Beginning-Of-Sequence) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_bos")
        tok_id = lib.llama_vocab_bos(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_eos(self) -> Optional[int]:
        """Get the EOS (End-Of-Sequence) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_eos")
        tok_id = lib.llama_vocab_eos(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_eot(self) -> Optional[int]:
        """Get the EOT (End-Of-Turn) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_eot")
        tok_id = lib.llama_vocab_eot(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_sep(self) -> Optional[int]:
        """Get the SEP (Separator) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_sep")
        tok_id = lib.llama_vocab_sep(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_nl(self) -> Optional[int]:
        """Get the NL (Newline) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_nl")
        tok_id = lib.llama_vocab_nl(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_pad(self) -> Optional[int]:
        """Get the PAD (Padding) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_pad")
        tok_id = lib.llama_vocab_pad(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def add_bos_token(self) -> bool:
        """If the model is configured to add a BOS token"""
        null_ptr_check(self._vocab, "self._vocab", "Llama.add_bos_token")
        return lib.llama_vocab_get_add_bos(self._vocab)

    def add_eos_token(self) -> bool:
        """If the model is configured to add an EOS token"""
        null_ptr_check(self._vocab, "self._vocab", "Llama.add_eos_token")
        return lib.llama_vocab_get_add_eos(self._vocab)

    def token_fim_pre(self) -> Optional[int]:
        """Get the FIM PRE (Fill-In-Middle Prefix) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_fim_pre")
        tok_id = lib.llama_vocab_fim_pre(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_fim_suf(self) -> Optional[int]:
        """Get the FIM SUF (Fill-In-Middle Suffix) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_fim_suf")
        tok_id = lib.llama_vocab_fim_suf(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_fim_mid(self) -> Optional[int]:
        """Get the FIM MID (Fill-In-Middle Middle) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_fim_mid")
        tok_id = lib.llama_vocab_fim_mid(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_fim_pad(self) -> Optional[int]:
        """Get the FIM PAD (Fill-In-Middle Padding) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_fim_pad")
        tok_id = lib.llama_vocab_fim_pad(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_fim_rep(self) -> Optional[int]:
        """Get the FIM REP (Fill-In-Middle Repository) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_fim_rep")
        tok_id = lib.llama_vocab_fim_rep(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None

    def token_fim_sep(self) -> Optional[int]:
        """Get the FIM SEP (Fill-In-Middle Separator) token. Return None if not available."""
        null_ptr_check(self._vocab, "self._vocab", "Llama.token_fim_sep")
        tok_id = lib.llama_vocab_fim_sep(self._vocab)
        return tok_id if tok_id != lib.LLAMA_TOKEN_NULL else None
    
    def tokenize(
        self,
        text_bytes: bytes,
        add_special: bool,
        parse_special: bool,
    ) -> list[int]:
        """Convert the provided UTF-8 encoded text into tokens

        - text_bytes:
            The text to be tokenized
        - add_special:
            Allow to add BOS and EOS tokens if model is configured to do so.
        - parse_special:
            Allow tokenizing special and/or control tokens which otherwise are
            not exposed and treated as plaintext. Does not insert a leading
            space."""
        null_ptr_check(self._vocab, 'self._vocab', 'Llama.tokenize')
        n_tokens = _internals.get_length(
            vocab=self._vocab,
            text_bytes=text_bytes,
            add_special=add_special,
            parse_special=parse_special
        )
        return _internals.tokenize(
            vocab=self._vocab,
            text_bytes=text_bytes,
            n_tokens_max=n_tokens,
            add_special=add_special,
            parse_special=parse_special
        )

    def token_to_piece(self, token: int, special: bool) -> bytes:
        """Convert a single token ID into UTF-8 bytes

        - special:
            If True, special tokens are rendered in the output"""
        null_ptr_check(self._vocab, 'self._vocab', 'Llama.token_to_piece')
        return _internals.token_to_piece(
            vocab=self._vocab,
            token=token,
            special=special
        )

    def detokenize(
        self,
        tokens: list[int],
        special: bool
    ) -> str:
        """Convert the provided tokens into UTF-8 encoded text

        - special:
            If True, special tokens are rendered in the output"""
        null_ptr_check(self._vocab, 'self._vocab', 'Llama.detokenize')
        return _internals.detokenize(
            vocab=self._vocab,
            tokens=tokens,
            special=special
        )
    
    def get_length(
        self,
        text_bytes: bytes,
        add_special: bool,
        parse_special: bool,
    ) -> int:
        """Return the length of a given text as measured in tokens"""
        null_ptr_check(self._vocab, 'self._vocab', 'Llama.get_length')
        return _internals.get_length(
            vocab=self._vocab,
            text_bytes=text_bytes,
            add_special=add_special,
            parse_special=parse_special
        )

    def _first_valid_pos(self, tokens: list[int]) -> int:
        """Given a list of tokens, and using `Llama.context_tokens`, find the first
        valid `Llama.pos`

        In other words, return length of the longest common prefix between the
        two lists of tokens.

        Returns 0 if none of the tokens match, 1 if one token matches, etc."""
        i = 0
        for c, t in zip(self.context_tokens, tokens):
            if c == t:
                i += 1
            else:
                break
        return i

    def _set_cache_tokens(self, input_tokens: list[int]) -> list[int]:
        if self.vocab_only:
            raise RuntimeError('Llama._set_cache_tokens: this instance is vocab-only')
        
        n_input_tokens = len(input_tokens)

        if n_input_tokens == 0:
            raise ValueError(f'Llama._set_cache_tokens: input_tokens cannot be empty')
        
        # find how many tokens in the input are already in the KV cache
        self.pos = self._first_valid_pos(input_tokens)

        if self.pos > self._n_ctx:
            raise ExceededContextLengthException(
                f'Llama._set_cache_tokens: no valid position within context window '
                f'{self._n_ctx} (self.pos = {self.pos})'
            )

        # remove all tokens in the KV cache that are past that point
        self.kv_cache_seq_rm(0, self.pos, -1)

        # tokens already in the KV cache
        self.context_tokens = input_tokens[:self.pos]

        # tokens returned to caller for processing
        actual_input_tokens = input_tokens[self.pos:]

        return actual_input_tokens

    def _decode_batch(
        self,
        batch_tokens: list[int],
        logits_all: bool = False
    ) -> Optional[np.ndarray]:
        """Decode a batch of one or more tokens, up to `Llama.n_batch()`. If `logits_all` is
        True, return the logits for all tokens in the batch. Otherwise, return None.
        
        This function is used by `Llama.eval`, `Llama.generate`, `Llama.stream`, etc."""

        if self.vocab_only:
            raise RuntimeError('Llama._decode_batch: this instance is vocab-only')

        batch_logits = None
        n_batch_tokens = len(batch_tokens)
        log_debug(f'Llama._decode_batch: decoding {batch_tokens}')

        if n_batch_tokens > self._n_batch:
            raise ValueError(
                f'Llama._decode_batch: n_batch_tokens cannot exceed n_batch '
                f'({n_batch_tokens} > {self._n_batch})'
            )
        
        if n_batch_tokens > 1: # prompt processing
            if logits_all:
                with self._lock:
                    self._stopwatch.start_pp()
                    batch_logits = _internals.decode_pp_with_logits(
                        self._ctx.ctx, self.pos, batch_tokens, n_batch_tokens, self._n_vocab
                    )
                    self._stopwatch.stop_pp()
            
            else:
                with self._lock:
                    self._stopwatch.start_pp()
                    _internals.decode_pp(self._ctx.ctx, self.pos, batch_tokens, n_batch_tokens)
                    self._stopwatch.stop_pp()
            
            self._stopwatch.increment_pp_tokens(n_batch_tokens)
        
        elif n_batch_tokens == 1: # text generation
            with self._lock:
                self._stopwatch.start_tg()
                batch_logits = _internals.decode_tg_with_logits(
                    self._ctx.ctx, self.pos, batch_tokens[0], self._n_vocab
                )
                self._stopwatch.stop_tg()
            
            self._stopwatch.increment_tg_tokens(1)
        
        else:
            raise RuntimeError(
                f'Llama._decode_batch: unexpected n_batch_tokens value {n_batch_tokens}'
            )
        
        # update the Llama position and context
        self.pos += n_batch_tokens
        self.context_tokens.extend(batch_tokens)
        
        return batch_logits

    def eval(
        self,
        input_tokens: list[int],
        logits_all: bool = False
    ) -> np.ndarray:
        """Evaluate the given tokens and update the model state.
        
        If `logits_all` is True, return the logits for all `input_tokens`. Otherwise, only
        return the logits for last token (which are the predictions for the next token)."""

        if self.vocab_only:
            raise RuntimeError('Llama.eval: this instance is vocab-only')

        self._stopwatch.reset()
        self._stopwatch.start_wall_time()

        n_input_tokens = len(input_tokens)

        if logits_all:
            if self._first_valid_pos(input_tokens) > 0:
                log(
                    f'Llama.eval: the KV cache will be cleared in order to compute the logits '
                    f'for all tokens in the input', 2
                )
            
            log_verbose(f'Llama.eval: {n_input_tokens} tokens to eval ...')

            self.reset()
            actual_input_tokens = input_tokens
            n_actual_input_tokens = len(input_tokens)
        else:
            actual_input_tokens = self._set_cache_tokens(input_tokens)

            n_actual_input_tokens = len(actual_input_tokens)
            n_cache_hit_tokens = n_input_tokens - n_actual_input_tokens

            log_verbose(
                f'Llama.eval: {n_cache_hit_tokens} tokens in cache, '
                f'{n_actual_input_tokens} tokens to eval ...'
            )

        if n_actual_input_tokens == 0:
            return self.get_logits()

        batches = split_tokens_into_batches(actual_input_tokens, self._n_batch)

        # decode each batch one-by-one
        if logits_all:
            all_logits = []
            for batch in _batches_with_progress_bar(batches):
                batch_logits = self._decode_batch(batch, logits_all=True)
                all_logits.append(batch_logits)
            final_logits = np.concatenate(all_logits, axis=0)
        else:
            for batch in _batches_with_progress_bar(batches):
                self._decode_batch(batch, logits_all=False)
            final_logits = self.get_logits()

        self._stopwatch.stop_wall_time()

        if get_verbose():
            self._stopwatch.print_stats()

        return final_logits

    def generate_single(
        self,
        input_tokens: list[int],
        sampler_preset: Optional[SamplerPreset] = None,
        return_logits: bool = False
    ) -> Union[int, np.ndarray]:
        """Generate a single token

        - input_tokens:
            The tokens to evaluate
        - sampler_preset:
            The `SamplerPreset` object to use for sampling. If not specified,
            use the model's default sampler parameters
        - return_logits:
            If True, return the logits for the generated token instead of the token ID."""
        
        if self.vocab_only:
            raise RuntimeError('Llama.generate_single: this instance is vocab-only')

        self._stopwatch.reset()
        self._stopwatch.start_wall_time()

        n_input_tokens = len(input_tokens)

        actual_input_tokens = self._set_cache_tokens(input_tokens)

        n_actual_input_tokens = len(actual_input_tokens)
        n_cache_hit_tokens = n_input_tokens - n_actual_input_tokens

        if sampler_preset is None:
            sampler_params = self._default_sampler_params
        else:
            sampler_params = self.sampler_params_from_preset(sampler_preset)

        if get_verbose():
            sampler_params.print_chain()
        
        log_verbose(
            f'Llama.generate_single: {n_cache_hit_tokens} tokens in cache, '
            f'{n_actual_input_tokens} tokens to eval ...'
        )

        batches = split_tokens_into_batches(actual_input_tokens, self._n_batch)

        # decode each batch one-by-one
        for batch in _batches_with_progress_bar(batches):
            self._decode_batch(batch, logits_all=False)
        
        self._stopwatch.stop_wall_time()

        if get_verbose():
            self._stopwatch.print_stats()

        if return_logits: # TODO: this is inefficient, it decodes the last token again. replace.
            return self.get_logits()

        return self.sample(sampler_params)

    def generate(
        self,
        input_tokens: list[int],
        n_predict: int,
        stop_tokens: Optional[list[int]] = None,
        sampler_preset: Optional[SamplerPreset] = None,
        return_logits: bool = False
    ) -> Union[list[int], np.ndarray]:
        """Generate new tokens and return them all at once

        - input_tokens:
            The tokens to evaluate
        - n_predict:
            The number of tokens to predict. If `n_predict < 0`, then the number of tokens
            predicted is only limited by the context length. If `n_predict == 0`, then no new
            tokens will be predicted, but the input_tokens will still be processed.
        - stop_tokens:
            A list of token IDs that will end the generation early. Note that the stop token
            will be included in the output. If this parameter is None, all built-in stop tokens
            for the model will be used. Pass an empty list `[]` to ignore all stop tokens.
        - sampler_preset:
            The `SamplerPreset` object to use for sampling. If not specified, use the model's
            default sampler parameters.
        - return_logits:
            If True, return the logits for the generated tokens instead of the token IDs. Note
            that this incurs a slight performance penalty."""
        
        if self.vocab_only:
            raise RuntimeError('Llama.generate: this instance is vocab-only')

        self._stopwatch.reset()
        self._stopwatch.start_wall_time()

        n_input_tokens = len(input_tokens)

        actual_input_tokens = self._set_cache_tokens(input_tokens)

        n_actual_input_tokens = len(actual_input_tokens)
        n_cache_hit_tokens = n_input_tokens - n_actual_input_tokens

        _stop_tokens = stop_tokens if stop_tokens is not None else self.eog_tokens

        if sampler_preset is None:
            sampler_params = self._default_sampler_params
        else:
            sampler_params = self.sampler_params_from_preset(sampler_preset)
        
        if get_verbose():
            sampler_params.print_chain()

        _n_predict = n_predict if n_predict >= 0 else self._n_ctx - self.pos

        log_verbose(
            f'Llama.generate: {n_cache_hit_tokens} tokens in cache, {n_actual_input_tokens} '
            f'tokens to eval ...'
        )

        batches = split_tokens_into_batches(actual_input_tokens, self._n_batch)

        log_debug('Llama.generate: start decoding input batches')

        # decode each input batch one-by-one
        for batch in _batches_with_progress_bar(batches):
            self._decode_batch(batch, logits_all=False)

        log_debug('Llama.generate: done decodinging input batches')
        
        if _n_predict == 0:
            self._stopwatch.stop_wall_time()
            if get_verbose():
                self._stopwatch.print_stats()
            return []
        
        predicted_tokens = []
        if return_logits:
            predicted_logits = []
        n_predicted = 0

        log_verbose(f'Llama.generate: predicting up to {_n_predict} new tokens ...')
        log_debug(f'Llama.generate: enter while loop')

        while n_predicted < _n_predict:
            # sample a token from the latest logits
            sampled_token = self.sample(sampler_params)

            # save the sampled token as a prediction
            predicted_tokens.append(sampled_token)
            n_predicted += 1

            # if it's a stop token, stop generating
            if sampled_token in _stop_tokens:
                if get_verbose():
                    tok_str = ez_decode(self.token_to_piece(sampled_token, True))
                    print()
                    log(f'inferred stop token {sampled_token} ({tok_str!r})')
                break
            
            # decode the sampled token to get the logits for the following token
            if return_logits:
                logits = self._decode_batch([sampled_token], True)
                predicted_logits.append(logits)
            else:
                self._decode_batch([sampled_token])
        
        # done generating, show stopwatch stats and return
        self._stopwatch.stop_wall_time()
        log_debug(f'Llama.generate: exited while loop')
        if get_verbose():
            self._stopwatch.print_stats()
        if return_logits:
            return np.stack(predicted_logits, axis=0)
        return predicted_tokens
    
    def stream(
        self,
        input_tokens: list[int],
        n_predict: int,
        stop_tokens: Optional[list[int]] = None,
        sampler_preset: Optional[SamplerPreset] = None,
        yield_logits: bool = False
    ) -> Iterable[Union[int, np.ndarray]]:
        """Return a Generator which yields tokens as they are generated

        - input_tokens:
            The tokens to evaluate
        - n_predict:
            The number of tokens to predict. If `n_predict < 0`, then the number of tokens
            predicted is only limited by the context length. If `n_predict == 0`, then no new
            tokens will be predicted, but the input_tokens will still be processed.
        - stop_tokens:
            A list of token IDs that will end the generation early. Note that
            the stop token will be included in the output. If this parameter is
            None, all built-in stop tokens for the model will be used. Pass an
            empty list `[]` to ignore all stop tokens.
        - sampler_preset:
            The `SamplerPreset` object to use for sampling. If not specified,
            use the model's default sampler parameters
        - yield_logits:
            If True, yield the logits for the generated tokens instead of the token IDs"""
        
        if self.vocab_only:
            raise RuntimeError('Llama.stream: this instance is vocab-only')

        self._stopwatch.reset()
        self._stopwatch.start_wall_time()

        n_input_tokens = len(input_tokens)

        actual_input_tokens = self._set_cache_tokens(input_tokens)

        n_actual_input_tokens = len(actual_input_tokens)
        n_cache_hit_tokens = n_input_tokens - n_actual_input_tokens

        _stop_tokens = stop_tokens if stop_tokens is not None else self.eog_tokens

        if sampler_preset is None:
            sampler_params = self._default_sampler_params
        else:
            sampler_params = self.sampler_params_from_preset(sampler_preset)
        
        if get_verbose():
            sampler_params.print_chain()

        _n_predict = n_predict if n_predict >= 0 else self._n_ctx - self.pos

        log_verbose(
            f'Llama.stream: {n_cache_hit_tokens} tokens in cache, {n_actual_input_tokens} '
            f'tokens to eval ...'
        )

        batches = split_tokens_into_batches(actual_input_tokens, self._n_batch)

        log_debug('Llama.stream: start decoding input batches')

        # decode each input batch one-by-one
        for batch in _batches_with_progress_bar(batches):
            self._decode_batch(batch, logits_all=False)
        
        log_debug('Llama.stream: done decoding input batches')
        
        if _n_predict == 0:
            self._stopwatch.stop_wall_time()
            if get_verbose():
                self._stopwatch.print_stats()
            return
        
        n_predicted = 0

        log_verbose(f'Llama.stream: predicting up to {_n_predict} new tokens ...')
        log_debug(f'Llama.stream: enter while loop')

        while n_predicted < _n_predict:
            # sample a token from the latest logits
            sampled_token = self.sample(sampler_params)

            is_stop_token = sampled_token in _stop_tokens

            if yield_logits:
                if not is_stop_token:
                    # decode the token, yield the logits for the next prediction
                    logits = self._decode_batch([sampled_token], logits_all=True)
                    yield logits
            else:
                yield sampled_token
            
            n_predicted += 1

            if is_stop_token:
                if get_verbose():
                    tok_str = ez_decode(self.token_to_piece(sampled_token, special=True)) 
                    print()
                    log(f'inferred stop token {sampled_token} ({tok_str!r})')
                break
            
            if not yield_logits:
                self._decode_batch([sampled_token], logits_all=False)
        
        # done generating, show stopwatch stats
        self._stopwatch.stop_wall_time()
        log_debug(f'Llama.stream: exited while loop')
        if get_verbose():
            self._stopwatch.print_stats()
    
    def stream_chars(
        self,
        input_tokens: list[int],
        n_predict: int,
        stop_tokens: Optional[list[int]] = None,
        sampler_preset: Optional[SamplerPreset] = None,
    ) -> Iterable[str]:
        """Return a Generator which yields single characters as they are generated

        - input_tokens:
            The tokens to evaluate
        - n_predict:
            The number of tokens to predict. If `n_predict < 0`, then the number of tokens
            predicted is only limited by the context length. If `n_predict == 0`, then no new
            tokens will be predicted, but the input_tokens will still be processed.
        - stop_tokens:
            A list of token IDs that will end the generation early. Note that
            the stop token will be included in the output. If this parameter is
            None, all built-in stop tokens for the model will be used. Pass an
            empty list `[]` to ignore all stop tokens.
        - sampler_preset:
            The `SamplerPreset` object to use for sampling. If not specified,
            use the model's default sampler parameters"""
        
        tok_gen = self.stream(
            input_tokens=input_tokens,
            n_predict=n_predict,
            stop_tokens=stop_tokens,
            sampler_preset=sampler_preset,
            yield_logits=False
        )
        
        detok_bytes_buffer = b''
        
        for tok in tok_gen:
            #
            # detok_bytes_buffer holds any incomplete UTF-8 characters until they
            # are completed by future tokens
            # 
            # for example, emojis are often split between two tokens, with one or
            # both of those tokens not being valid UTF-8 on its own
            #
            detok_bytes_buffer += self.token_to_piece(tok, special=False)
            try:
                detok_txt = detok_bytes_buffer.decode('utf-8', errors='strict')
            except UnicodeDecodeError:
                pass # try again on next token
            else:
                detok_bytes_buffer = b''
                yield detok_txt
        
        # yield any leftover bytes (though ideally there should be none)
        if detok_bytes_buffer != b'':
            leftover_txt = ez_decode(detok_bytes_buffer)
            yield leftover_txt
    
    def benchmark(
        self,
        n_tokens_pp: Optional[int] = None,
        n_tokens_tg: Optional[int] = None,
        n_runs: Optional[int] = None
    ) -> list[dict]:
        """Measure the prompt processing and text generation speed of this Llama."""

        if self.vocab_only:
            raise RuntimeError('Llama.benchmark: this instance is vocab-only')
        
        n_tokens_pp = n_tokens_pp if n_tokens_pp is not None else self.n_batch()
        n_tokens_tg = n_tokens_tg if n_tokens_tg is not None else 10
        n_runs      = n_runs      if n_runs      is not None else 3

        results = []
        total_pp_time_ns = 0
        total_tg_time_ns = 0

        for i in range(1, n_runs+1):

            log_verbose(f'benchmark: starting run {i}/{n_runs}:')

            log_verbose(f'benchmark: processing {n_tokens_pp} tokens ... please wait ...')   
            with suppress_output(disable=get_debug()):
                self.reset()
                self.eval(input_tokens=[0] * n_tokens_pp)
            pp_ns = self._stopwatch.get_elapsed_time_pp()
            total_pp_time_ns += pp_ns
            
            log_verbose(f'benchmark: generating {n_tokens_tg} tokens ... please wait ...')
            with suppress_output(disable=get_debug()):
                self.reset()
                self.generate(
                    input_tokens=[0],
                    n_predict=n_tokens_tg - 1,
                    stop_tokens=[],
                    sampler_preset=SamplerPreset(seed=42, top_k=1, temp=0.0)
                )
            tg_ns = self._stopwatch.get_elapsed_time_tg()
            total_tg_time_ns += tg_ns

            results.append({
                'n_tokens_pp' : n_tokens_pp,
                'n_tokens_tg' : n_tokens_tg,
                'pp_time_ns'  : pp_ns,
                'tg_time_ns'  : tg_ns
            })

        avg_pp_time_ns = total_pp_time_ns / n_runs
        avg_tg_time_ns = total_tg_time_ns / n_runs

        avg_pp_time_ms = avg_pp_time_ns / 1e6
        avg_tg_time_ms = avg_tg_time_ns / 1e6

        avg_pp_tok_per_sec = n_tokens_pp / (avg_pp_time_ns / 1e9)
        avg_tg_tok_per_sec = n_tokens_tg / (avg_tg_time_ns / 1e9)

        log_verbose(
            f'average pp speed for {n_tokens_pp:>7} tokens over {n_runs} runs: '
            f'{avg_pp_time_ms:>13.3f}ms ({avg_pp_tok_per_sec:10.2f} tok/s)', 4
        )
        log_verbose(
            f'average tg speed for {n_tokens_tg:>7} tokens over {n_runs} runs: '
            f'{avg_tg_time_ms:>13.3f}ms ({avg_tg_tok_per_sec:10.2f} tok/s)', 4
        )

        return results
    
    def sample_greedy(self) -> int:
        """Sample from the model's current logits greedily"""
        if self.vocab_only:
            raise RuntimeError('Llama.sample_greedy: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, 'self._ctx.ctx', 'Llama.sample_greedy')
        return _internals.sample_greedy(self._ctx.ctx)
    
    def sampler_params_from_preset(self, sampler_preset: SamplerPreset) -> SamplerParams:
        """Create and return a new `SamplerParams` object for this Llama using the provided
        `SamplerPreset`.

        @param sampler_preset: The `sampling.SamplerPreset` object which defines the sampler
        parameter values to use"""

        if self.vocab_only:
            raise RuntimeError('Llama.sampler_params_from_preset: this instance is vocab-only')
        
        return SamplerParams(
            llama                 = self,
            seed                  = sampler_preset.seed,
            top_k                 = sampler_preset.top_k,
            top_p                 = sampler_preset.top_p,
            min_p                 = sampler_preset.min_p,
            xtc_probability       = sampler_preset.xtc_probability,
            xtc_threshold         = sampler_preset.xtc_threshold,
            typical_p             = sampler_preset.typical_p,
            temp                  = sampler_preset.temp,
            dynatemp_delta        = sampler_preset.dynatemp_delta,
            dynatemp_exponent     = sampler_preset.dynatemp_exponent,
            penalty_last_n        = sampler_preset.penalty_last_n,
            penalty_repeat        = sampler_preset.penalty_repeat,
            penalty_freq          = sampler_preset.penalty_freq,
            penalty_present       = sampler_preset.penalty_present,
            dry_multiplier        = sampler_preset.dry_multiplier,
            dry_base              = sampler_preset.dry_base,
            dry_allowed_length    = sampler_preset.dry_allowed_length,
            dry_penalty_last_n    = sampler_preset.dry_penalty_last_n,
            mirostat              = sampler_preset.mirostat,
            top_n_sigma           = sampler_preset.top_n_sigma,
            mirostat_tau          = sampler_preset.mirostat_tau,
            mirostat_eta          = sampler_preset.mirostat_eta,
            dry_sequence_breakers = sampler_preset.dry_sequence_breakers,
            logit_bias            = sampler_preset.logit_bias
        )

    def sample(self, sampler_params: Optional[SamplerParams] = None) -> int:
        """Sample a token using the current context

        - params:
            The `sampling.SamplerParams` object which defines the sampling
            parameters to use. If this parameter is None, the default sampler
            paramater values will be used."""
        
        if self.vocab_only:
            raise RuntimeError('Llama.sample: this instance is vocab-only')
        
        sampler_params = sampler_params if sampler_params is not None else (
            self._default_sampler_params
        )
        null_ptr_check(self._ctx.ctx, 'self._ctx.ctx', 'Llama.sample')
        return lib.llama_sampler_sample(sampler_params.smpl, self._ctx.ctx, -1)
    
    def get_logits(self) -> np.ndarray:
        """Return the raw logits for the last token in the context, which are the predictions
        for the next token. The returned array has shape `(n_vocab,)`."""
        if self.vocab_only:
            raise RuntimeError('Llama.get_logits: this instance is vocab-only')
        null_ptr_check(self._ctx.ctx, 'self._ctx.ctx', 'Llama.get_logits')
        return _internals.get_logits(self._ctx.ctx, self._n_vocab)

    def get_scores(self, temp: Optional[float] = None) -> np.ndarray:
        """Compute the logits for the last token in the context, normalized with softmax.
        Optionally apply temperature `temp` if specified.

        Any floating-point value for temperature `temp` is valid, including 0.0
        and negative numbers.

        The returned array has shape `(n_vocab,)`."""
        logits = self.get_logits()
        return softmax(logits, T=temp if temp is not None else 1.0)
    
    def get_tokenization_mapping(self, tokens: list[int]) -> list[tuple[int, bytes]]:
        """Given some tokens, return a list of tuples where the first item in the
        tuple is the token ID and the second item is the corresponding UTF-8
        text bytes."""
        return list(
            zip(tokens, [self.token_to_piece(tok_id, special=True) for tok_id in tokens])
        )
    
    def print_tokenization_mapping(
        self,
        tokens: list[int],
        file: Optional[_SupportsWriteAndFlush] = None
    ) -> None:
        """Given some tokens, print a mapping of each token ID to the
        corresponding UTF-8 text bytes

        This is meant to be roughly equivalent to `llama.cpp/llama-tokenize`

        - tokens:
            The tokens to print a mapping for
        - file:
            The file or stream to which the mapping will be printed"""
        _file = sys.stderr if file is None else file
        token_mapping = self.get_tokenization_mapping(tokens)
        for tok_id, bytes in token_mapping:
            print(f"{tok_id:>7} -> {repr(bytes)} ({bytes.hex(':')})", file=_file)
            #print(f"{tok_id:>7} -> {str(txt)}", file=file)
        print(f"Total number of tokens: {len(token_mapping)}", file=_file, flush=True)
    
    def name(self) -> str:
        """Get the name of the model from the GGUF metadata. Fallback to using the filename
        if the name is not set in the metadata."""
        # '/path/to/my-model.gguf' --> 'my-model'
        model_file_basename = os.path.basename(self._model.path_model).removesuffix('.gguf')
        # 'my-model-00001-of-99999' --> 'my-model'
        model_file_basename = re.sub(r'-\d{5}-of-\d{5}$', '', model_file_basename)
        # use name from metadata if possible
        model_name = self.metadata.get('general.name', model_file_basename)
        return model_name
    
    def bpw(self) -> float:
        """Get the average bits per weight of the model"""
        return (self._model_size_bytes * 8) / self._n_params
    
    def save_state(self, file_path: str) -> None:
        """Save the current state of the context to a file"""
        if self.vocab_only:
            raise RuntimeError('Llama.save_state: this instance is vocab-only')
        
        null_ptr_check(self._ctx.ctx, 'self._ctx.ctx', 'Llama.save_state')

        state_size_bytes = lib.llama_state_get_size(self._ctx.ctx)
        state_size_mib = int(state_size_bytes / (1024 * 1024)) # approximate

        log(f'Llama.save_state: state size: {state_size_mib} MiB ({state_size_bytes} bytes)')
        log(f'Llama.save_state: saving to {file_path} ...')

        if os.path.exists(file_path):
            log(f'Llama.save_state: file exists, will be overwritten', 2)

        # save the llama state
        with suppress_output(disable=get_verbose()):
            success = lib.llama_state_save_file(self._ctx.ctx, file_path, self.context_tokens)

        if success:
            log(f'Llama.save_state: successfully saved state')
        else:
            raise RuntimeError(f'Llama.save_state: failed to save state')

    def load_state(self, file_path: str) -> None:
        """Load a previously saved context state from a file"""
        if self.vocab_only:
            raise RuntimeError('Llama.load_state: this instance is vocab-only')
        
        null_ptr_check(self._ctx.ctx, 'self._ctx.ctx', 'Llama.load_state')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Llama.load_state: file_path {file_path} does not exist')
        
        if os.path.isdir(file_path):
            raise IsADirectoryError(f'Llama.load_state: file_path {file_path} is a directory')
        
        # reset the current context before loading the new one
        self.reset()

        n_ctx = self.n_ctx()
        loaded_tokens_buf = (lib.llama_token * n_ctx)()
        n_loaded_tokens_p = ctypes.c_size_t(0)

        log(f'Llama.load_state: loading from {file_path} ...')

        # load the llama state
        with suppress_output(disable=get_verbose()):
            success = lib.llama_state_load_file(
                ctx=self._ctx.ctx,
                path_session=file_path,
                tokens_out=loaded_tokens_buf,
                n_token_capacity=n_ctx,
                n_token_count_out=ctypes.byref(n_loaded_tokens_p)
            )
        
        if success:
            n_loaded_tokens = n_loaded_tokens_p.value

            self.context_tokens = list(loaded_tokens_buf[:n_loaded_tokens])
            self.pos = n_loaded_tokens

            state_size_bytes = lib.llama_state_get_size(self._ctx.ctx)
            state_size_mib = int(state_size_bytes / (1024 * 1024)) # approximate

            log(f'Llama.load_state: state size: {state_size_mib} MiB ({state_size_bytes} bytes)')
            log(f'Llama.load_state: successfully loaded state ({n_loaded_tokens} tokens)')
        else:
            raise RuntimeError(f'Llama.load_state: failed to load state')

    def reset(self) -> None:
        """Reset the position of the model and clear the KV cache"""
        if not self.vocab_only:
            self.kv_cache_clear()
            self.pos = 0
            self.context_tokens = []
