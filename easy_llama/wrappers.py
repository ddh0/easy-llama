# wrappers.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from .libllama import *
import ctypes
import sys

class ModelWrapper:

    def __init__(
        self,
        devices = None,
        n_gpu_layers = 0,
        split_mode = 0,
        main_gpu = 0,
        tensor_split = None,
        rpc_servers = None,
        progress_callback = None,
        progress_callback_user_data = None,
        kv_overrides = None,
        vocab_only = False,
        use_mmap = False,
        use_mlock = False,
        check_tensors = False
    ):
        self.params = llama_model_params()
        self.params.devices = (ctypes.c_void_p * (len(devices) + 1))(*devices, None) if devices else None
        self.params.n_gpu_layers = n_gpu_layers
        self.params.split_mode = split_mode
        self.params.main_gpu = main_gpu
        self.params.tensor_split = (ctypes.c_float * len(tensor_split))(*tensor_split) if tensor_split else None
        self.params.rpc_servers = rpc_servers.encode('utf-8') if rpc_servers else None
        self.params.progress_callback = progress_callback
        self.params.progress_callback_user_data = progress_callback_user_data
        self.params.kv_overrides = (llama_model_kv_override * len(kv_overrides))(*kv_overrides) if kv_overrides else None
        self.params.vocab_only = vocab_only
        self.params.use_mmap = use_mmap
        self.params.use_mlock = use_mlock
        self.params.check_tensors = check_tensors
        self.model = None

    def load_model(self, path_model):
        self.model = llama_load_model_from_file(path_model, self.params)

    def free_model(self):
        if self.model:
            llama_free_model(self.model)
            self.model = None

class ContextWrapper:

    def __init__(
        self,
        n_ctx = 0,
        n_batch = 1,
        n_ubatch = 1,
        n_seq_max = 512,
        n_threads = 1,
        n_threads_batch = 1,
        rope_scaling_type = 0,
        pooling_type = 0,
        attention_type = 0,
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
        type_k = 0,
        type_v = 0,
        logits_all = False,
        embeddings = False,
        offload_kqv = False,
        flash_attn = False,
        no_perf = False,
        abort_callback = None,
        abort_callback_data = None
    ):
        self.params = llama_context_params()
        self.params.n_ctx = n_ctx
        self.params.n_batch = n_batch
        self.params.n_ubatch = n_ubatch
        self.params.n_seq_max = n_seq_max
        self.params.n_threads = n_threads
        self.params.n_threads_batch = n_threads_batch
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
        self.params.cb_eval = cb_eval
        self.params.cb_eval_user_data = cb_eval_user_data
        self.params.type_k = type_k
        self.params.type_v = type_v
        self.params.logits_all = logits_all
        self.params.embeddings = embeddings
        self.params.offload_kqv = offload_kqv
        self.params.flash_attn = flash_attn
        self.params.no_perf = no_perf
        self.params.abort_callback = abort_callback
        self.params.abort_callback_data = abort_callback_data
        self.context = None

    def create_context(self, model_wrapper):
        self.context = llama_new_context_with_model(model_wrapper.model, self.params)

    def free_context(self):
        if self.context:
            llama_free(self.context)
            self.context = None

# Example usage
if __name__ == '__main__':
    import os

    if not os.path.exists('./model.gguf'):
        raise FileNotFoundError('the file ./model.gguf was not found')

    def test_print(text: str) -> None:
        print(f'easy_llama:', text, file=sys.stderr, flush=True)

    test_print(f"begin basic libllama test")

    print("-" * 80)

    test_print(f"calling llama_backend_init ...")
    llama_backend_init()

    model_wrapper = ModelWrapper()
    test_print(f"calling llama_load_model_from_file ...")
    model_wrapper.load_model('./model.gguf')

    context_wrapper = ContextWrapper()
    test_print(f"calling llama_new_context_with_model ...")
    context_wrapper.create_context(model_wrapper)

    print(f"{context_wrapper.model is model_wrapper.context=}")

    test_print(f"calling llama_free(ctx) ...")
    context_wrapper.free_context()

    test_print(f"calling llama_free_model ...")
    model_wrapper.free_model()

    print("-" * 80)

    test_print("basic libllama test complete")
