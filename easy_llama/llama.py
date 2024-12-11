# llama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import ctypes
import sys
import os
import libllama as lib

from typing import NoReturn, Optional

from .constants import Colors

RESET  = Colors.RESET
GREEN  = Colors.GREEN
BLUE   = Colors.BLUE
GREY   = Colors.GREY
YELLOW = Colors.YELLOW
RED    = Colors.RED

C_NULL = None
C_NULLPTR = ctypes.c_void_p(C_NULL)

_SUPPORTED_KV_TYPES = [
    lib.GGMLType.GGML_TYPE_F32,   # lib only supports static types, not
    lib.GGMLType.GGML_TYPE_F16,   # k-types
    lib.GGMLType.GGML_TYPE_Q8_1,
    lib.GGMLType.GGML_TYPE_Q8_0,  # BF16 is also sometimes supported, but not
    lib.GGMLType.GGML_TYPE_Q5_1,  # always, and offers no benefit compared
    lib.GGMLType.GGML_TYPE_Q5_0,  # to F16, so is not listed here
    lib.GGMLType.GGML_TYPE_Q4_1,
    lib.GGMLType.GGML_TYPE_Q4_0
]

_DEFAULT_KV_TYPE = lib.GGMLType.GGML_TYPE_F16

_cpu_count = os.cpu_count()

class LlamaNullException(Exception):
    """Raised when a libllama functions returns C_NULL or C_NULLPTR"""

def null_exception_check(
    ptr: lib.ptr, func_name: str, loc_hint: str
) -> None | NoReturn:
    """
    Check to ensure some pointer is not None

    - loc_hint: Code location hint used in easy_llama
    """
    if ptr is C_NULL:
        raise LlamaNullException(
            f"{loc_hint}: {func_name} returned NULL"
        )

def nullptr_exception_check(
    ptr: lib.ptr, func_name: str, loc_hint: str
) -> None | NoReturn:
    """
    Check to ensure some pointer is not nullptr

    - loc_hint: Code location hint used in easy_llama
    """
    if ptr is C_NULLPTR:
        raise LlamaNullException(
            f"{loc_hint}: {func_name} returned C_NULLPTR"
        )

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
        null_exception_check(
            self.params, "llama_model_default_params", "_LlamaModel.__init__"
        )
        self.params.devices = (
                ctypes.c_void_p * (len(devices) + 1)
            )(*devices, None) if devices is not None else C_NULL
        self.params.n_gpu_layers = n_gpu_layers if n_gpu_layers >= 0 else lib.MAX_OFFLOAD_LAYERS
        self.params.split_mode = split_mode if split_mode is not None else lib.LlamaSplitMode.LLAMA_SPLIT_MODE_NONE
        self.params.main_gpu = main_gpu if main_gpu is not None else 0
        self.params.tensor_split = (
                ctypes.c_float * len(tensor_split)
            )(*tensor_split) if tensor_split is not None else C_NULL
        self.params.rpc_servers = (
                rpc_servers.encode('utf-8')
            ) if rpc_servers is not None else C_NULL
        self.params.progress_callback = (
                progress_callback
            ) if progress_callback is not None else lib.dummy_progress_callback()
        self.params.progress_callback_user_data = (
                progress_callback_user_data
            ) if progress_callback_user_data is not None else C_NULL
        self.params.kv_overrides = (
                lib.llama_model_kv_override * len(kv_overrides)
            )(*kv_overrides) if kv_overrides is not None else C_NULL
        self.params.vocab_only = vocab_only
        self.params.use_mmap = use_mmap if lib.llama_supports_mmap() else False
        self.params.use_mlock = use_mlock if lib.llama_supports_mlock() else False
        self.params.check_tensors = check_tensors
        self.model = lib.llama_load_model_from_file(path_model, self.params)
        null_exception_check(
            self.model, "llama_load_model_from_file", "_LlamaModel.__init__"
        )

    def free(self):
        if self.model is not None:
            lib.llama_free_model(self.model)

class _LlamaCtx:

    def __init__(
        self,
        model: '_LlamaModel',
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
        self.model = model
        self.params = lib.llama_context_default_params()
        null_exception_check(
            self.params, "llama_context_default_params", "_LlamaCtx.__init__"
        )
        self.params.n_ctx = n_ctx
        self.params.n_batch = n_batch
        self.params.n_ubatch = n_ubatch
        self.params.n_seq_max = n_seq_max
        if self.params.n_seq_max != 1:
            _print_warning(
                f'n_seq_max is not 1, this is not recommended'
            )
        self.params.n_threads = _get_optimal_n_threads() if n_threads <= 0 else n_threads
        self.params.n_threads_batch = (
                _get_optimal_n_threads_batch()
            ) if n_threads_batch <= 0 else n_threads_batch
        self.params.rope_scaling_type = rope_scaling_type if rope_scaling_type is not None else lib.LlamaRopeScalingType.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        self.params.pooling_type = pooling_type if pooling_type is not None else lib.LlamaPoolingType.LLAMA_POOLING_TYPE_UNSPECIFIED,
        self.params.attention_type = attention_type if attention_type is not None else lib.LlamaAttentionType.LLAMA_ATTENTION_TYPE_UNSPECIFIED
        self.params.rope_freq_base = rope_freq_base
        self.params.rope_freq_scale = rope_freq_scale
        self.params.yarn_ext_factor = yarn_ext_factor
        self.params.yarn_attn_factor = yarn_attn_factor
        self.params.yarn_beta_fast = yarn_beta_fast
        self.params.yarn_beta_slow = yarn_beta_slow
        self.params.yarn_orig_ctx = yarn_orig_ctx
        self.params.defrag_thold = defrag_thold
        self.params.cb_eval = cb_eval if cb_eval is not None else lib.dummy_eval_callback()
        self.params.cb_eval_user_data = cb_eval_user_data
        self.params.type_k = type_k if type_k is not None else _DEFAULT_KV_TYPE
        self.params.type_v = type_v if type_v is not None else _DEFAULT_KV_TYPE
        if type_k != type_v:
            _print_warning(
                f'type_k != type_v ({type_k} != {type_v}). this is rarely '
                f'supported'
            )
        if type_k not in _SUPPORTED_KV_TYPES:
            _print_warning(
                f'type_k is {type_k}, this is not supported. program may crash.'
            )
        if type_v not in _SUPPORTED_KV_TYPES:
            _print_warning(
                f'type_v is {type_k}, this is not supported. program may crash.'
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

        self.ctx = lib.llama_new_context_with_model(
            model.model, self.params
        )
        null_exception_check(
            self.ctx, "llama_new_context_with_model", "_LlamaCtx.__init__"
        )


    def free(self):
        if self.ctx is not None:
            lib.llama_free(self.ctx)
    
    def get_model(self) -> '_LlamaModel':
        if self.model is not None:
            return self.model
    
    def n_ctx(self) -> int:
        return self.params.n_ctx

#
# End of wrappers / Begin helper functions
#

# TODO: should remove these 3 print functions before release?

def _print_verbose(text: str) -> None:
    print(
        f"easy_llama:",
        text, file=sys.stderr, flush=True
    )

def _print_info(text: str) -> None:
    print(
        f"{RESET}easy_llama: {GREEN}INFO{RESET}:",
        text, file=sys.stderr, flush=True
    )

def _print_warning(text: str) -> None:
    print(
        f"{RESET}easy_llama: {YELLOW}WARNING{RESET}:",
        text, file=sys.stderr, flush=True
    )

def _init_backend_if_needed() -> None:
    if lib._BACKEND_INIT is not True:
        lib.llama_backend_init()

def _get_optimal_n_threads() -> int:
    global _cpu_count
    return max(_cpu_count//2, 1)

def _get_optimal_n_threads_batch() -> int:
    global _cpu_count
    return _cpu_count

def _get_random_seed() -> int:
    int.from_bytes(os.urandom(4), sys.byteorder)

#
# Llama
#

class Llama:
    pass

#
# End of functions / Begin test
#

if __name__ == '__main__':

    # Handy-dandy basic test of wrappers
    # Assumes model.gguf is available in the current working directory

    import os

    if not os.path.exists('./model.gguf'):
        raise FileNotFoundError('the file ./model.gguf was not found')
    
    this_module_name = os.path.splitext(os.path.basename(__file__))[0]

    def test_print(text: str) -> None:
        _print_verbose(f'{this_module_name} test: {text}')

    print("-" * 80)

    test_print("constructing _LlamaModel")
    _model = _LlamaModel('./model.gguf')

    test_print("constructing _LlamaCtx")
    _ctx = _LlamaCtx(_model)

    test_print("freeing context via wrapper")
    _ctx.free()

    test_print("freeing model via wrapper")
    _model.free()

    test_print("deleting _LlamaCtx")
    del _ctx

    test_print("deleting _LlamaModel")
    del _model

    print("-" * 80)
