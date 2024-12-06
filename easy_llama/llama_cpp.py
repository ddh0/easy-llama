# llama_cpp.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import ctypes
from typing import Optional

libllama = ctypes.CDLL('/Users/dylan/Documents/AI/easy-llama/easy_llama/libllama.dylib')

class LlamaCPPModel:
    def __init__(self, model_path: str, params: Optional[dict] = None):
        """
        Initialize the Llama model with the given model path and parameters.

        Args:
            model_path (str): The path to the model file.
            params (Optional[dict]): A dictionary of parameters for the model.
        """
        self.model_path = model_path
        self.params = params if params else self.default_model_params()
        self.model = self.load_model(model_path, self.params)
        self.context = self.new_context(self.model, self.default_context_params())

    @staticmethod
    def default_model_params() -> dict:
        """
        Get the default parameters for the Llama model.

        Returns:
            dict: Default model parameters.
        """
        return {
            "n_gpu_layers": 0,
            "split_mode": 0,
            "main_gpu": 0,
            "tensor_split": None,
            "rpc_servers": None,
            "progress_callback": None,
            "progress_callback_user_data": None,
            "kv_overrides": None,
            "vocab_only": False,
            "use_mmap": False,
            "use_mlock": False,
            "check_tensors": False,
        }

    @staticmethod
    def default_context_params() -> dict:
        """
        Get the default parameters for the Llama context.

        Returns:
            dict: Default context parameters.
        """
        return {
            "n_ctx": 0,
            "n_batch": 0,
            "n_ubatch": 0,
            "n_seq_max": 0,
            "n_threads": 0,
            "n_threads_batch": 0,
            "rope_scaling_type": 0,
            "pooling_type": 0,
            "attention_type": 0,
            "rope_freq_base": 0.0,
            "rope_freq_scale": 0.0,
            "yarn_ext_factor": 0.0,
            "yarn_attn_factor": 0.0,
            "yarn_beta_fast": 0.0,
            "yarn_beta_slow": 0.0,
            "yarn_orig_ctx": 0,
            "defrag_thold": 0.0,
            "cb_eval": None,
            "cb_eval_user_data": None,
            "type_k": 0,
            "type_v": 0,
            "logits_all": False,
            "embeddings": False,
            "offload_kqv": False,
            "flash_attn": False,
            "no_perf": False,
            "abort_callback": None,
            "abort_callback_data": None,
        }

    @staticmethod
    def load_model(model_path: str, params: dict) -> ctypes.c_void_p:
        """
        Load the Llama model from the given file path with the specified parameters.

        Args:
            model_path (str): The path to the model file.
            params (dict): A dictionary of parameters for the model.

        Returns:
            ctypes.c_void_p: A pointer to the loaded model.
        """
        model_params = LlamaCPPModel.create_model_params(params)
        return libllama.llama_load_model_from_file(model_path.encode(), ctypes.byref(model_params))

    @staticmethod
    def create_model_params(params: dict) -> ctypes.c_void_p:
        """
        Create a ctypes structure for model parameters.

        Args:
            params (dict): A dictionary of parameters for the model.

        Returns:
            ctypes.c_void_p: A pointer to the model parameters structure.
        """
        class ModelParams(ctypes.Structure):
            _fields_ = [
                ("n_gpu_layers", ctypes.c_int32),
                ("split_mode", ctypes.c_int),
                ("main_gpu", ctypes.c_int32),
                ("tensor_split", ctypes.POINTER(ctypes.c_float)),
                ("rpc_servers", ctypes.c_char_p),
                ("progress_callback", ctypes.c_void_p),
                ("progress_callback_user_data", ctypes.c_void_p),
                ("kv_overrides", ctypes.c_void_p),
                ("vocab_only", ctypes.c_bool),
                ("use_mmap", ctypes.c_bool),
                ("use_mlock", ctypes.c_bool),
                ("check_tensors", ctypes.c_bool),
            ]

        return ModelParams(
            n_gpu_layers=params.get("n_gpu_layers", 0),
            split_mode=params.get("split_mode", 0),
            main_gpu=params.get("main_gpu", 0),
            tensor_split=params.get("tensor_split", None),
            rpc_servers=params.get("rpc_servers", None),
            progress_callback=params.get("progress_callback", None),
            progress_callback_user_data=params.get("progress_callback_user_data", None),
            kv_overrides=params.get("kv_overrides", None),
            vocab_only=params.get("vocab_only", False),
            use_mmap=params.get("use_mmap", False),
            use_mlock=params.get("use_mlock", False),
            check_tensors=params.get("check_tensors", False),
        )

    @staticmethod
    def new_context(model: ctypes.c_void_p, params: dict) -> ctypes.c_void_p:
        """
        Create a new context with the given model and parameters.

        Args:
            model (ctypes.c_void_p): A pointer to the model.
            params (dict): A dictionary of parameters for the context.

        Returns:
            ctypes.c_void_p: A pointer to the new context.
        """
        context_params = LlamaCPPModel.create_context_params(params)
        return libllama.llama_new_context_with_model(model, ctypes.byref(context_params))

    @staticmethod
    def create_context_params(params: dict) -> ctypes.c_void_p:
        """
        Create a ctypes structure for context parameters.

        Args:
            params (dict): A dictionary of parameters for the context.

        Returns:
            ctypes.c_void_p: A pointer to the context parameters structure.
        """
        class ContextParams(ctypes.Structure):
            _fields_ = [
                ("n_ctx", ctypes.c_uint32),
                ("n_batch", ctypes.c_uint32),
                ("n_ubatch", ctypes.c_uint32),
                ("n_seq_max", ctypes.c_uint32),
                ("n_threads", ctypes.c_int32),
                ("n_threads_batch", ctypes.c_int32),
                ("rope_scaling_type", ctypes.c_int),
                ("pooling_type", ctypes.c_int),
                ("attention_type", ctypes.c_int),
                ("rope_freq_base", ctypes.c_float),
                ("rope_freq_scale", ctypes.c_float),
                ("yarn_ext_factor", ctypes.c_float),
                ("yarn_attn_factor", ctypes.c_float),
                ("yarn_beta_fast", ctypes.c_float),
                ("yarn_beta_slow", ctypes.c_float),
                ("yarn_orig_ctx", ctypes.c_uint32),
                ("defrag_thold", ctypes.c_float),
                ("cb_eval", ctypes.c_void_p),
                ("cb_eval_user_data", ctypes.c_void_p),
                ("type_k", ctypes.c_int),
                ("type_v", ctypes.c_int),
                ("logits_all", ctypes.c_bool),
                ("embeddings", ctypes.c_bool),
                ("offload_kqv", ctypes.c_bool),
                ("flash_attn", ctypes.c_bool),
                ("no_perf", ctypes.c_bool),
                ("abort_callback", ctypes.c_void_p),
                ("abort_callback_data", ctypes.c_void_p),
            ]

        return ContextParams(
            n_ctx=params.get("n_ctx", 0),
            n_batch=params.get("n_batch", 0),
            n_ubatch=params.get("n_ubatch", 0),
            n_seq_max=params.get("n_seq_max", 0),
            n_threads=params.get("n_threads", 0),
            n_threads_batch=params.get("n_threads_batch", 0),
            rope_scaling_type=params.get("rope_scaling_type", 0),
            pooling_type=params.get("pooling_type", 0),
            attention_type=params.get("attention_type", 0),
            rope_freq_base=params.get("rope_freq_base", 0.0),
            rope_freq_scale=params.get("rope_freq_scale", 0.0),
            yarn_ext_factor=params.get("yarn_ext_factor", 0.0),
            yarn_attn_factor=params.get("yarn_attn_factor", 0.0),
            yarn_beta_fast=params.get("yarn_beta_fast", 0.0),
            yarn_beta_slow=params.get("yarn_beta_slow", 0.0),
            yarn_orig_ctx=params.get("yarn_orig_ctx", 0),
            defrag_thold=params.get("defrag_thold", 0.0),
            cb_eval=params.get("cb_eval", None),
            cb_eval_user_data=params.get("cb_eval_user_data", None),
            type_k=params.get("type_k", 0),
            type_v=params.get("type_v", 0),
            logits_all=params.get("logits_all", False),
            embeddings=params.get("embeddings", False),
            offload_kqv=params.get("offload_kqv", False),
            flash_attn=params.get("flash_attn", False),
            no_perf=params.get("no_perf", False),
            abort_callback=params.get("abort_callback", None),
            abort_callback_data=params.get("abort_callback_data", None),
        )

    @staticmethod
    def free_model(model: ctypes.c_void_p):
        """
        Free the memory allocated for the model.

        Args:
            model (ctypes.c_void_p): A pointer to the model.
        """
        libllama.llama_free_model(model)

    @staticmethod
    def free_context(context: ctypes.c_void_p):
        """
        Free the memory allocated for the context.

        Args:
            context (ctypes.c_void_p): A pointer to the context.
        """
        libllama.llama_free(context)

    @staticmethod
    def tokenize(model: ctypes.c_void_p, text: str, text_len: int, tokens: list[int], n_tokens_max: int, add_special: bool, parse_special: bool) -> int:
        """
        Tokenize the given text into tokens.

        Args:
            model (ctypes.c_void_p): A pointer to the model.
            text (str): The text to tokenize.
            text_len (int): The length of the text.
            tokens (list[int]): A list to store the resulting tokens.
            n_tokens_max (int): The maximum number of tokens to return.
            add_special (bool): Allow adding special tokens if the model is configured to do so.
            parse_special (bool): Allow parsing special tokens.

        Returns:
            int: The number of tokens on success, no more than n_tokens_max. Returns a negative number on failure.
        """
        tokens_array = (ctypes.c_int32 * n_tokens_max)(*tokens)
        result = libllama.llama_tokenize(model, text.encode(), text_len, tokens_array, n_tokens_max, add_special, parse_special)
        tokens[:result] = tokens_array[:result]
        return result

    @staticmethod
    def detokenize(model: ctypes.c_void_p, tokens: list[int], n_tokens: int, text: bytes, text_len_max: int, remove_special: bool, unparse_special: bool) -> int:
        """
        Detokenize the given tokens into text.

        Args:
            model (ctypes.c_void_p): A pointer to the model.
            tokens (list[int]): The tokens to detokenize.
            n_tokens (int): The number of tokens.
            text (bytes): The buffer to write the text to.
            text_len_max (int): The length of the buffer.
            remove_special (bool): Allow to remove BOS and EOS tokens if model is configured to do so.
            unparse_special (bool): If true, special tokens are rendered in the output.

        Returns:
            int: The number of characters on success, no more than text_len_max. Returns a negative number on failure.
        """
        tokens_array = (ctypes.c_int32 * n_tokens)(*tokens)
        result = libllama.llama_detokenize(model, tokens_array, n_tokens, text, text_len_max, remove_special, unparse_special)
        return result

    @staticmethod
    def encode(context: ctypes.c_void_p, batch: ctypes.c_void_p) -> int:
        """
        Encode the given batch of tokens.

        Args:
            context (ctypes.c_void_p): A pointer to the context.
            batch (ctypes.c_void_p): A pointer to the batch of tokens.

        Returns:
            int: 0 on success, negative on error.
        """
        return libllama.llama_encode(context, batch)

    @staticmethod
    def decode(context: ctypes.c_void_p, batch: ctypes.c_void_p) -> int:
        """
        Decode the given batch of tokens.

        Args:
            context (ctypes.c_void_p): A pointer to the context.
            batch (ctypes.c_void_p): A pointer to the batch of tokens.

        Returns:
            int: 0 on success, positive on warning, negative on error.
        """
        return libllama.llama_decode(context, batch)

    @staticmethod
    def batch_init(n_tokens: int, embd: int, n_seq_max: int) -> ctypes.c_void_p:
        """
        Initialize a batch of tokens.

        Args:
            n_tokens (int): The number of tokens.
            embd (int): The embedding size.
            n_seq_max (int): The maximum number of sequences.

        Returns:
            ctypes.c_void_p: A pointer to the initialized batch.
        """
        return libllama.llama_batch_init(n_tokens, embd, n_seq_max)

    @staticmethod
    def batch_free(batch: ctypes.c_void_p):
        """
        Free the memory allocated for the batch.

        Args:
            batch (ctypes.c_void_p): A pointer to the batch.
        """
        libllama.llama_batch_free(batch)

    @staticmethod
    def get_logits(context: ctypes.c_void_p) -> list[float]:
        """
        Get the logits from the last call to llama_decode.

        Args:
            context (ctypes.c_void_p): A pointer to the context.

        Returns:
            list[float]: The logits.
        """
        logits_ptr = libllama.llama_get_logits(context)
        logits = [logits_ptr[i] for i in range(LlamaCPPModel.n_vocab(context.model))]
        return logits

    @staticmethod
    def get_embeddings(context: ctypes.c_void_p) -> list[float]:
        """
        Get the embeddings from the last call to llama_decode.

        Args:
            context (ctypes.c_void_p): A pointer to the context.

        Returns:
            list[float]: The embeddings.
        """
        embeddings_ptr = libllama.llama_get_embeddings(context)
        embeddings = [embeddings_ptr[i] for i in range(LlamaCPPModel.n_embd(context.model))]
        return embeddings

    @staticmethod
    def n_vocab(model: ctypes.c_void_p) -> int:
        """
        Get the vocabulary size of the model.

        Args:
            model (ctypes.c_void_p): A pointer to the model.

        Returns:
            int: The vocabulary size.
        """
        return libllama.llama_n_vocab(model)

    @staticmethod
    def n_embd(model: ctypes.c_void_p) -> int:
        """
        Get the embedding size of the model.

        Args:
            model (ctypes.c_void_p): A pointer to the model.

        Returns:
            int: The embedding size.
        """
        return libllama.llama_n_embd(model)
