# wrappers.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import ctypes
import sys
import os
import libllama as lib

from typing import NoReturn

C_NULL = None
C_NULLPTR = ctypes.c_void_p(C_NULL)

_cpu_count = os.cpu_count()

class LibllamaNullException(Exception):
    """Raised when a libllama functions returns a null pointer"""

def null_exception_check(
    ptr: lib.ptr, func_name: str, loc_hint: str
) -> None | NoReturn:
    """
    Check to ensure some pointer is not None

    - loc_hint: Code location hint used in easy_llama
    """
    if ptr is None:
        raise LibllamaNullException(
            f"{loc_hint}: {func_name} returned None"
        )

class ModelWrapper:

    def __init__(
        self,
        path_model: str,
        devices = None,
        n_gpu_layers = 0, # Less than 0 for MAX_OFFLOAD_LAYERS
        split_mode = lib.LlamaSplitMode.LLAMA_SPLIT_MODE_NONE,
        main_gpu = 0,
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
            self.params, "llama_model_default_params", "ModelWrapper.__init__"
        )
        self.params.devices = (
                ctypes.c_void_p * (len(devices) + 1)
            )(*devices, None) if devices is not None else C_NULL
        self.params.n_gpu_layers = n_gpu_layers if n_gpu_layers >= 0 else lib.MAX_OFFLOAD_LAYERS
        self.params.split_mode = split_mode
        self.params.main_gpu = main_gpu
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
            self.model, "llama_load_model_from_file", "ModelWrapper.__init__"
        )

    def free(self):
        if self.model is not None:
            lib.llama_free_model(self.model)
            self.model = None

class CtxWrapper:

    dummy_eval_callback = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
    )()

    def __init__(
        self,
        model: 'ModelWrapper',
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        #n_seq_max = 1, # AKA n_parallel, values >1 unsupported
        n_threads = 0,
        n_threads_batch = 0,
        rope_scaling_type = lib.LlamaRopeScalingType.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        pooling_type = lib.LlamaPoolingType.LLAMA_POOLING_TYPE_UNSPECIFIED,
        attention_type = lib.LlamaAttentionType.LLAMA_ATTENTION_TYPE_CAUSAL,
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
        type_k = lib.GGMLType.GGML_TYPE_F16,
        type_v = lib.GGMLType.GGML_TYPE_F16,
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
            self.params, "llama_context_default_params", "CtxWrapper.__init__"
        )
        self.params.n_ctx = n_ctx
        self.params.n_batch = n_batch
        self.params.n_ubatch = n_ubatch
        self.params.n_seq_max = 1
        self.params.n_threads = _get_optimal_n_threads() if n_threads <= 0 else n_threads
        self.params.n_threads_batch = (
                _get_optimal_n_threads_batch()
            ) if n_threads_batch <= 0 else n_threads_batch
        self.params.rope_scaling_type = rope_scaling_type
        self.params.pooling_type = pooling_type
        self.params.attention_type = attention_type
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
        self.params.type_k = type_k
        self.params.type_v = type_v
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
            self.ctx, "llama_new_context_with_model", "CtxWrapper.__init__"
        )
        
    def free(self):
        if self.ctx is not None:
            lib.llama_free(self.ctx)
            self.ctx = None
    
    def get_model(self) -> 'ModelWrapper':
        if self.model is not None:
            return self.model
    
    def n_ctx(self) -> int:
        return self.params.n_ctx

#
# End of wrappers / Begin helper functions
#

def print_verbose(text: str) -> None:  # TODO: remove me when packaging
    print(
        f"easy_llama:",
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

#
# End of functions / Begin test
#

# Example usage
if __name__ == '__main__':
    import os

    if not os.path.exists('./model.gguf'):
        raise FileNotFoundError('the file ./model.gguf was not found')

    def test_print(text: str) -> None:
        print(f'easy_llama:', text, file=sys.stderr, flush=True)

    print("-" * 80)

    test_print("constructing model wrapper (loading model)")
    model_wrapper = ModelWrapper('./model.gguf', n_gpu_layers=-1)

    test_print("constructing context wrapper (loading context)")
    ctx_wrapper = CtxWrapper(model_wrapper, n_ctx=131072, flash_attn=True)

    test_print("freeing context via wrapper")
    ctx_wrapper.free()

    test_print("freeing model via wrapper")
    model_wrapper.free()

    test_print("deleting context wrapper")
    del ctx_wrapper

    test_print("deleting model wrapper")
    del model_wrapper

    print("-" * 80)
