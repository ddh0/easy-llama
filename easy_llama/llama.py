# llama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import ctypes
from typing import Optional
from .llama_cpp import LlamaCPPModel, libllama

class Llama:

    def __init__(self, model_path: str, params: Optional[dict] = None):
        """
        Initialize the Llama model with the given model path and parameters.

        Args:
            model_path (str): The path to the model file.
            params (Optional[dict]): A dictionary of parameters for the model.
        """
        self.model_path = model_path
        self.params = params if params else LlamaCPPModel.default_model_params()
        self.model = LlamaCPPModel.load_model(model_path, self.params)
        self.context = LlamaCPPModel.new_context(self.model, LlamaCPPModel.default_context_params())

    def __del__(self):
        """
        Destructor to free the model and context.
        """
        LlamaCPPModel.free_model(self.model)
        LlamaCPPModel.free_context(self.context)

    def vocab_type(self) -> int:
        """
        Get the vocabulary type of the model.

        Returns:
            int: The vocabulary type.
        """
        # Assuming there's a function in the library to get the vocab type
        return libllama.llama_vocab_type(self.model)

    def n_vocab(self) -> int:
        """
        Get the vocabulary size of the model.

        Returns:
            int: The vocabulary size.
        """
        return LlamaCPPModel.n_vocab(self.model)

    def n_ctx_train(self) -> int:
        """
        Get the context size used during training.

        Returns:
            int: The context size.
        """
        # Assuming there's a function in the library to get the context size
        return libllama.llama_n_ctx_train(self.model)

    def n_embd(self) -> int:
        """
        Get the embedding size of the model.

        Returns:
            int: The embedding size.
        """
        return LlamaCPPModel.n_embd(self.model)

    def rope_freq_scale_train(self) -> float:
        """
        Get the RoPE frequency scale used during training.

        Returns:
            float: The RoPE frequency scale.
        """
        # Assuming there's a function in the library to get the RoPE frequency scale
        return libllama.llama_rope_freq_scale_train(self.model)

    def desc(self) -> str:
        """
        Get a description of the model.

        Returns:
            str: The model description.
        """
        # Assuming there's a function in the library to get the model description
        return libllama.llama_desc(self.model).decode('utf-8')

    def size(self) -> int:
        """
        Get the size of the model in bytes.

        Returns:
            int: The model size.
        """
        # Assuming there's a function in the library to get the model size
        return libllama.llama_size(self.model)

    def n_params(self) -> int:
        """
        Get the number of parameters in the model.

        Returns:
            int: The number of parameters.
        """
        # Assuming there's a function in the library to get the number of parameters
        return libllama.llama_n_params(self.model)

    def token_bos(self) -> int:
        """
        Get the BOS (Beginning of Sentence) token.

        Returns:
            int: The BOS token.
        """
        # Assuming there's a function in the library to get the BOS token
        return libllama.llama_token_bos(self.model)

    def token_eos(self) -> int:
        """
        Get the EOS (End of Sentence) token.

        Returns:
            int: The EOS token.
        """
        # Assuming there's a function in the library to get the EOS token
        return libllama.llama_token_eos(self.model)

    def token_cls(self) -> int:
        """
        Get the CLS (Classification) token.

        Returns:
            int: The CLS token.
        """
        # Assuming there's a function in the library to get the CLS token
        return libllama.llama_token_cls(self.model)

    def token_sep(self) -> int:
        """
        Get the SEP (Separator) token.

        Returns:
            int: The SEP token.
        """
        # Assuming there's a function in the library to get the SEP token
        return libllama.llama_token_sep(self.model)

    def token_nl(self) -> int:
        """
        Get the NL (New Line) token.

        Returns:
            int: The NL token.
        """
        # Assuming there's a function in the library to get the NL token
        return libllama.llama_token_nl(self.model)

    def token_prefix(self) -> int:
        """
        Get the PREFIX token.

        Returns:
            int: The PREFIX token.
        """
        # Assuming there's a function in the library to get the PREFIX token
        return libllama.llama_token_prefix(self.model)

    def token_middle(self) -> int:
        """
        Get the MIDDLE token.

        Returns:
            int: The MIDDLE token.
        """
        # Assuming there's a function in the library to get the MIDDLE token
        return libllama.llama_token_middle(self.model)

    def token_suffix(self) -> int:
        """
        Get the SUFFIX token.

        Returns:
            int: The SUFFIX token.
        """
        # Assuming there's a function in the library to get the SUFFIX token
        return libllama.llama_token_suffix(self.model)

    def token_eot(self) -> int:
        """
        Get the EOT (End of Text) token.

        Returns:
            int: The EOT token.
        """
        # Assuming there's a function in the library to get the EOT token
        return libllama.llama_token_eot(self.model)

    def add_bos_token(self) -> bool:
        """
        Check if the model adds the BOS token.

        Returns:
            bool: True if the model adds the BOS token, False otherwise.
        """
        # Assuming there's a function in the library to check if the model adds the BOS token
        return libllama.llama_add_bos_token(self.model)

    def add_eos_token(self) -> bool:
        """
        Check if the model adds the EOS token.

        Returns:
            bool: True if the model adds the EOS token, False otherwise.
        """
        # Assuming there's a function in the library to check if the model adds the EOS token
        return libllama.llama_add_eos_token(self.model)

    def tokenize(self, text: bytes, add_bos: bool, special: bool):
        """
        Tokenize the given text into tokens.

        Args:
            text (bytes): The text to tokenize (utf-8 encoded).
            add_bos (bool): Whether to add the BOS token.
            special (bool): Whether to recognize special tokens.

        Returns:
            List[int]: The list of tokens.
        """
        tokens = []
        n_tokens_max = 1024  # Assuming a reasonable maximum number of tokens
        result = LlamaCPPModel.tokenize(self.model, text.decode('utf-8'), len(text), tokens, n_tokens_max, add_bos, special)
        return tokens[:result]

    def token_to_piece(self, token: int, special: bool = False) -> bytes:
        """
        Convert a token to its corresponding text piece.

        Args:
            token (int): The token to convert.
            special (bool): Whether to recognize special tokens.

        Returns:
            bytes: The text piece corresponding to the token (utf-8 encoded).
        """
        text_buffer = ctypes.create_string_buffer(1024)  # Assuming a reasonable buffer size
        result = LlamaCPPModel.detokenize(self.model, [token], 1, text_buffer, 1024, special, special)
        return text_buffer.value[:result]

    def detokenize(self, tokens: list[int]) -> bytes:
        """
        Detokenize the given tokens into text.

        Args:
            tokens (List[int]): The tokens to detokenize.

        Returns:
            bytes: The detokenized text (utf-8 encoded).
        """
        text_buffer = ctypes.create_string_buffer(1024)  # Assuming a reasonable buffer size
        result = LlamaCPPModel.detokenize(self.model, tokens, len(tokens), text_buffer, 1024, True, True)
        return text_buffer.value[:result]
