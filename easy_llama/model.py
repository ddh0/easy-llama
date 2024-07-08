# model.py
# https://github.com/ddh0/easy-llama/
from ._version import __version__, __llama_cpp_version__

"""Submodule containing the Model class to work with language models"""

import os
import sys
import heapq
import numpy as np

from .utils import (
    _SupportsWriteAndFlush,
    QuickGGUFReader,
    print_warning,
    print_verbose,
    _print_debug,
    assert_type,
    NoneType,
    softmax
)

from .samplers import SamplerSettings, DefaultSampling
from llama_cpp import Llama, StoppingCriteriaList
from typing    import Generator, Optional, Union


class ModelUnloadedException(Exception):
    """Exception raised when trying to use a model that has been unloaded"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class Model:
    """
    A high-level abstraction of a llama model

    This is just a brief overview of ez.Model.
    To see a full description of each method and its parameters,
    call help(Model), or see the relevant docstring.

    The following methods are available:
    - `.generate()` - Generate text
    - `.get_length()` - Get the length of a given text in tokens
    - `.ingest()` - Ingest text into the model's cache
    - `.next_candidates()` - Get a list of the most likely next tokens (WIP)
    - `.stream()` - Return a Generator that can stream text as it is generated
    - `.stream_print()` - Print text as it is generated
    - `.trim()` - Trim a given text to the model's context length
    - `.unload()` - Unload the model from memory

    The following attributes are available:
    - `.bos_token` - The model's beginning-of-stream token ID
    - `.context_length` - The model's loaded context length
    - `.flash_attn` - Whether the model was loaded with `flash_attn=True`
    - `.eos_token` - The model's end-of-stream token ID
    - `.llama` - The underlying `llama_cpp.Llama` instance
    - `.metadata` - The GGUF metadata of the model
    - `.n_ctx_train` - The native context length of the model
    - `.rope_freq_base` - The model's loaded RoPE frequency base
    - `.rope_freq_base_train` - The model's native RoPE frequency base
    - `.vocab` - A list of all the tokens in the model's vocabulary
    - `.verbose` - Whether the model was loaded with `verbose=True`
    """

    def __init__(
        self,
        model_path: str,
        context_length: Optional[int] = None,
        n_gpu_layers: int = 0,
        offload_kqv: bool = True,
        flash_attn: bool = False,
        verbose: bool = False
    ):
        """
        Given the path to a GGUF file, construct a Model instance.

        The model must be in GGUF format.

        The following parameters are optional:
        - context_length: The context length at which to load the model, in tokens
        - n_gpu_layers: The number of layers to be offloaded to the GPU
        - offload_kqv: Whether the KQV cache (context) should be offloaded
        - flash_attn: Whether to use Flash Attention
        - verbose: Whether to print additional backend information
        """

        assert_type(verbose, bool, 'verbose', 'Model')
        if verbose:
            print_verbose(f"easy_llama package version: {__version__}")
            print_verbose(f"llama_cpp package version: {__llama_cpp_version__}")

        assert_type(model_path, str, 'model_path', 'Model')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model: the given model_path {model_path!r} does not exist"
            )
        if os.path.isdir(model_path):
            raise IsADirectoryError(
                f"Model: the given model_path {model_path!r} is a directory, "
                "not a GGUF file"
            )
        assert_type(context_length, (int, NoneType), 'context_length', 'Model')
        assert_type(n_gpu_layers, int, 'n_gpu_layers', 'Model')
        assert_type(offload_kqv, bool, 'offload_kqv', 'Model')
        assert_type(flash_attn, bool, 'flash_attn', 'Model')
        
        # save __init__ parameters for __repr__
        self._model_path = model_path
        self._context_length = context_length
        self._n_gpu_layers = n_gpu_layers
        self._offload_kqv = offload_kqv
        self._flash_attn = flash_attn
        self._verbose = self.verbose = verbose

        # if context_length <= 0, use n_ctx_train
        if isinstance(context_length, int) and context_length <= 0:
            context_length = None

        if sys.byteorder == 'big':
            print_warning(
                "host is big-endian, please ensure your GGUF file is also "
                "big-endian"
            )
        elif sys.byteorder == 'little':
            if verbose:
                print_verbose(
                    "host is little-endian"
                )
        else:
            print_warning(
                f"unexpected value for sys.byteorder: {sys.byteorder!r}, "
                "expected 'little' for little-endian or 'big' for big-endian"
            )
        
        self._model_file_size_bytes = os.stat(model_path).st_size
        self.metadata = QuickGGUFReader.load_metadata(model_path)

        n_ctx_train = None
        rope_freq_base_train = None
        n_layer = None
        ctx_scale = None
        ctx_quality_hint = None

        for key in self.metadata.keys():
            if key.endswith('.context_length'):
                n_ctx_train = int(self.metadata[key])
            elif key.endswith('.rope.freq_base'):
                rope_freq_base_train = float(self.metadata[key])
            elif key.endswith('.block_count'):
                n_layer = int(self.metadata[key])

        if n_ctx_train is None:
            raise KeyError(
                "GGUF file does not specify a context length"
            )
        
        rope_freq_base = Model._calculate_rope_freq_base(
            n_ctx_train,
            context_length or n_ctx_train,
            rope_freq_base_train
        )

        if rope_freq_base_train is None or context_length is None or context_length <= n_ctx_train:
            # no need to do context scaling, load model normally

            if context_length is None:
                self.context_length = n_ctx_train
                ctx_scale = 1.0
            else:
                self.context_length = context_length
                ctx_scale = context_length/n_ctx_train
            ctx_quality_hint = 'native'

        elif context_length > n_ctx_train:
            # multiply rope_freq_base according to requested context length
            # because context length > n_ctx_train and rope freq base is known

            self.context_length = context_length
            ctx_scale = context_length/n_ctx_train

            if 1 < ctx_scale < 1.5:
                ctx_quality_hint = 'great'
            elif 1.5 <= ctx_scale < 2:
                ctx_quality_hint = 'good'
            elif 2 <= ctx_scale < 4:
                ctx_quality_hint = 'fair'
            elif 4 <= ctx_scale < 8:
                ctx_quality_hint = 'poor'
            else: # x8 or more
                ctx_quality_hint = 'abysmal'
            
            if ctx_scale >= 2: # anything below 'good'
                print_warning(
                    f"context scale is x{ctx_scale} ({ctx_quality_hint})"
                )

        cpu_count = int(os.cpu_count()) # only read once

        if n_layer is not None and (n_gpu_layers >= n_layer or n_gpu_layers < 0):
            # if model is fully offloaded
            n_batch = 1024
        else:
            # if model is not fully offloaded
            n_batch = 512
        
        # these values for n_threads and n_threads_batch are known to
        # be optimal unless the host system has an unequal amount of
        # P-cores and E-cores
        n_threads = max(cpu_count//2, 1)
        n_threads_batch = cpu_count

        if flash_attn and n_gpu_layers == 0:
            print_warning(
                "disabling flash_attn because n_gpu_layers == 0"
            )
            flash_attn = False
        
        # guard against models with no rope_freq_base
        if rope_freq_base is None:
            rope_freq_base = 0

        self.llama: Llama = Llama(
            model_path=model_path,
            n_ctx=self.context_length,
            n_gpu_layers=n_gpu_layers,         # KV cache quantization is
            use_mmap=True,                     # controlled by the `type_k`
            use_mlock=False,                   # and `type_v` parameters.
            logits_all=False,                  # not all combinations are
            n_batch=n_batch,                   # supported.
            n_threads=n_threads,               #
            n_threads_batch=n_threads_batch,   # use `1` for f16 (default)
            rope_freq_base=rope_freq_base,     # use `8` for q8_0
            mul_mat_q=True,                    # use `7` for q5_1
            offload_kqv=offload_kqv,           # use `6` for q5_0
            flash_attn=flash_attn,             # use `3` for q4_1
            #type_k=8,                         # use `2` for q4_0
            #type_v=8,
            verbose=verbose
        )

        try:
            self.vocab: list[str] = self.metadata['tokenizer.ggml.tokens']
        except (KeyError, TypeError, ValueError):
            self.vocab = None
            print_warning(
                "could not set Model.vocab, defaulting to None"
            )
        try:
            self.bos_token = int(self.metadata['tokenizer.ggml.bos_token_id'])
        except (KeyError, TypeError, ValueError):
            self.bos_token = int(self.llama.token_bos())
            if self.bos_token < 0:
                self.bos_token = None
                print_warning(
                    "could not set Model.bos_token, defaulting to None"
                )
        try:
            self.eos_token = int(self.metadata['tokenizer.ggml.eos_token_id'])
        except (KeyError, TypeError, ValueError):
            self.eos_token = int(self.llama.token_eos())
            if self.eos_token < 0:
                self.eos_token = None
                print_warning(
                    "could not set Model.eos_token, defaulting to None"
                )

        # These special tokens are optional
        # if a negative value is returned as a token, it is not defined
        self.eot_token = int(self.llama._model.token_eot())
        if self.eot_token < 0:
            self.eot_token = None
            if verbose:
                print_verbose(
                    "could not set Model.eot_token, defaulting to None"
                )
        self.nl_token  = int(self.llama._model.token_nl())
        if self.nl_token < 0:
            self.nl_token = None
            if verbose:
                print_verbose(
                    "could not set Model.nl_token, defaulting to None"
                )
        self.prefix_token = int(self.llama._model.token_prefix())
        if self.prefix_token < 0:
            self.prefix_token = None
            if verbose:
                print_verbose(
                    "could not set Model.prefix_token, defaulting to None"
                )
        self.middle_token = int(self.llama._model.token_middle())
        if self.middle_token < 0:
            self.middle_token = None
            if verbose:
                print_verbose(
                    "could not set Model.middle_token, defaulting to None"
                )
        self.suffix_token = int(self.llama._model.token_suffix())
        if self.suffix_token < 0:
            self.suffix_token = None
            if verbose:
                print_verbose(
                    "could not set Model.suffix_token, defaulting to None"
                )
        self.cls_token = int(self.llama._model.token_cls())
        if self.cls_token < 0:
            self.cls_token = None
            if verbose:
                print_verbose(
                    "could not set Model.cls_token, defaulting to None"
                )
        self.sep_token = int(self.llama._model.token_sep())
        if self.sep_token < 0:
            self.sep_token = None
            if verbose:
                print_verbose(
                    "could not set Model.sep_token, defaulting to None"
                )
        
        # expose these values because they may be useful / informative
        self.filename: str = os.path.basename(model_path)
        self.n_ctx_train: int = n_ctx_train
        self.rope_freq_base_train: float = rope_freq_base_train
        self.rope_freq_base: float = rope_freq_base
        self.flash_attn: bool = flash_attn
        self.n_vocab: int = len(self.vocab)
        self.n_layer: int = n_layer
        self.n_gpu_layers: int = n_gpu_layers 
        self.ctx_scale: float = ctx_scale

        if verbose:
            print_verbose("new Model instance with the following attributes:")
            print_verbose(f"filename             == {self.filename}")
            print_verbose(f"n_gpu_layers         == {n_gpu_layers}")
            if n_layer is not None:
                print_verbose(f"n_layer              == {self.n_layer}")
            print_verbose(f"offload_kqv          == {offload_kqv}")
            print_verbose(f"flash_attn           == {flash_attn}")
            print_verbose(f"n_batch              == {n_batch}")
            print_verbose(f"n_threads            == {n_threads}")
            print_verbose(f"n_threads_batch      == {n_threads_batch}")
            print_verbose(f"n_ctx_train          == {n_ctx_train}")
            print_verbose(f"context_length       == {self.context_length}")
            print_verbose(f"rope_freq_base_train == {rope_freq_base_train}")
            print_verbose(f"rope_freq_base       == {rope_freq_base}")
            print_verbose(f"ctx_scale            == {ctx_scale} ({ctx_quality_hint})")
            if self.n_vocab is not None:
                print_verbose(f"n_vocab              == {self.n_vocab}")
            if self.bos_token is not None:
                print_verbose(f"self.bos_token       == {self.bos_token}")
            if self.eos_token is not None:
                print_verbose(f"self.eos_token       == {self.eos_token}")
            if self.eot_token is not None:
                print_verbose(f"self.eot_token       == {self.eot_token}")
            if self.nl_token is not None:
                print_verbose(f"self.nl_token        == {self.nl_token}")
            if self.prefix_token is not None:
                print_verbose(f"self.prefix_token    == {self.prefix_token}")
            if self.middle_token is not None:
                print_verbose(f"self.middle_token    == {self.middle_token}")
            if self.suffix_token is not None:
                print_verbose(f"self.suffix_token    == {self.suffix_token}")
            if self.cls_token is not None:
                print_verbose(f"self.cls_token       == {self.cls_token}")
            if self.sep_token is not None:
                print_verbose(f"self.sep_token       == {self.sep_token}")
    
    @staticmethod
    def _calculate_rope_freq_base(
            n_ctx_train: int,
            n_ctx_load: int,
            rope_freq_base_train: Union[float, None]
        ) -> float:
        """
        Returns the rope_freq_base value at which model should be loaded
        """
        assert_type(n_ctx_train, int, 'n_ctx_train', '_calculate_rope_freq_base')
        assert_type(n_ctx_load, int, 'n_ctx_load', '_calculate_rope_freq_base')
        assert_type(rope_freq_base_train, (float, NoneType),
                    'rope_freq_base_train', '_calculate_rope_freq_base')

        if n_ctx_load <= n_ctx_train:
            if rope_freq_base_train is None:
                return 0.0
            else:
                return rope_freq_base_train
        
        if rope_freq_base_train in [0.0, None]:
            raise ValueError(
                'unable to load model with greater than native '
                f'context length ({n_ctx_load} > {n_ctx_train}) '
                'because model does not specify rope_freq_base. '
                f'try again with context_length <= {n_ctx_train}'
            )
        
        return ((n_ctx_load/n_ctx_train)**(2**(1/4)))*rope_freq_base_train

        # traditional formula:
        #   return ctx_scale*rope_freq_base_train
        # experimental formula A:
        #   return (ctx_scale**2)*rope_freq_base_train
        # experimental formula B:
        #   return (ctx_scale**(2**(1/4)))*rope_freq_base_train

    
    def __repr__(self) -> str:
        return (
            f"Model({self._model_path!r}, "
            f"context_length={self._context_length}, "
            f"n_gpu_layers={self._n_gpu_layers}, "
            f"offload_kqv={self._offload_kqv}, "
            f"flash_attn={self._flash_attn}, "
            f"verbose={self._verbose})"
        )
    
    def __sizeof__(self) -> int:
        return self._model_file_size_bytes
    
    def __del__(self):
        self.unload()
    
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.unload()
    
    def __call__(
        self,
        prompt: Union[str, list[int]],
        stops: list[Union[str, int]] = list(),
        sampler: SamplerSettings = DefaultSampling
    ) -> str:
        """
        `Model(...)` is a shorthand for `Model.generate(...)`
        """
        return self.generate(prompt, stops, sampler)
    
    def _print_debug(self) -> None:
        try:
            assert_model_is_loaded(self)
        except ModelUnloadedException as failed_assertion:
            exc = ModelUnloadedException(
                "_print_debug: Cannot debug a model that is not fully loaded"
            )
            raise exc from failed_assertion
        print(
            "Model: ---------------------------------------------------------",
            file=sys.stderr
        )
        _print_debug(self)
        print(
            "Llama: ---------------------------------------------------------",
            file=sys.stderr
        )
        _print_debug(self.llama)
        print(
            "_LlamaModel: ---------------------------------------------------",
            file=sys.stderr
        )
        _print_debug(self.llama._model)
        print(
            "llama_model_p: -------------------------------------------------",
            file=sys.stderr
        )
        _print_debug(self.llama._model.model)

    def unload(self):
        """
        Unload the model from memory
        """
        if not hasattr(self, 'llama'):
            if self.verbose:
                print_verbose('model already unloaded')
            return
        
        if self.verbose:
            print_verbose('unloading model...')

        self.llama.close()

        while hasattr(self, 'llama'):
            delattr(self, 'llama')

        if self.verbose:
            print_verbose('model unloaded')
    
    def reload(
        self,
        context_length: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        offload_kqv: Optional[bool] = None,
        flash_attn: Optional[bool] = None,
        verbose: Optional[bool] = None
    ):
        """
        Reload the model

        Any parameters unspecified will be unchanged
        """
        self.unload()
        self.__init__(
            model_path=self._model_path,
            context_length=self._context_length if context_length is None else context_length,
            n_gpu_layers=self._n_gpu_layers if n_gpu_layers is None else n_gpu_layers,
            offload_kqv=self._offload_kqv if offload_kqv is None else offload_kqv,
            flash_attn=self._flash_attn if flash_attn is None else flash_attn,
            verbose=self._verbose if verbose is None else verbose,
        )
        assert_model_is_loaded(self)

    def get_length(self, text: str) -> int:
        """
        Return the length of the given text in tokens according to this model,
        including the appended BOS token.
        """
        assert_model_is_loaded(self)
        return len(
            self.llama.tokenize(
                text.encode(
                    "utf-8",
                    errors="ignore"
                )
            )
        )

    def generate(
        self,
        prompt: Union[str, list[int]],
        stops: list[Union[str, int]] = list(),
        sampler: SamplerSettings = DefaultSampling
    ) -> str:
        """
        Given a prompt, return a generated string.

        prompt: The text from which to generate

        The following parameters are optional:
        - stops: A list of strings and/or token IDs at which to end the generation early
        - sampler: The SamplerSettings object used to control text generation
        """
        assert_type(prompt, (str, list), 'prompt', 'generate')
        if isinstance(prompt, list):
            for tok in prompt:
                assert_type(
                    tok,
                    int,
                    'some item in the list of prompt tokens',
                    'generate',
                )
        assert_type(stops, list, 'stops', 'generate')
        for item in stops:
            assert_type(
                item,
                (str, int),
                "some item in parameter 'stops'",
                'generate'
            )

        if self.verbose:
            print_verbose(f'using the following sampler settings for Model.generate:')
            print_verbose(f'max_len_tokens    == {sampler.max_len_tokens}')
            print_verbose(f'temp              == {sampler.temp}')
            print_verbose(f'top_p             == {sampler.top_p}')
            print_verbose(f'min_p             == {sampler.min_p}')
            print_verbose(f'frequency_penalty == {sampler.frequency_penalty}')
            print_verbose(f'presence_penalty  == {sampler.presence_penalty}')
            print_verbose(f'repeat_penalty    == {sampler.repeat_penalty}')
            print_verbose(f'top_k             == {sampler.top_k}')

        stop_strs: list[str] = [stop for stop in stops if isinstance(stop, str)]
        stop_token_ids: list[int] = [tok_id for tok_id in stops if isinstance(tok_id, int)]
        stopping_criteria = None
        if stop_token_ids != list():
            def stop_on_token_ids(tokens, *args, **kwargs):
                return tokens[-1] in stop_token_ids
            stopping_criteria = StoppingCriteriaList([stop_on_token_ids])

        assert_model_is_loaded(self)
        return self.llama.create_completion(
            prompt,
            max_tokens=sampler.max_len_tokens,
            temperature=sampler.temp,
            top_p=sampler.top_p,
            min_p=sampler.min_p,
            frequency_penalty=sampler.frequency_penalty,
            presence_penalty=sampler.presence_penalty,
            repeat_penalty=sampler.repeat_penalty,
            top_k=sampler.top_k,
            stop=stop_strs,
            stopping_criteria=stopping_criteria
        )['choices'][0]['text']
    

    def stream(
        self,
        prompt: Union[str, list[int]],
        stops: list[Union[str, int]] = list(),
        sampler: SamplerSettings = DefaultSampling
    ) -> Generator:

        """
        Given a prompt, return a Generator that yields dicts containing tokens.

        To get the token string itself, subscript the dict with:

        `['choices'][0]['text']`

        prompt: The text from which to generate

        The following parameters are optional:
        - stops: A list of strings and/or token IDs at which to end the generation early
        - sampler: The SamplerSettings object used to control text generation
        """

        assert_type(prompt, (str, list), 'prompt', 'stream')
        if isinstance(prompt, list):
            for tok in prompt:
                assert_type(
                    tok,
                    int,
                    'some item in the list of prompt tokens',
                    'stream'
                )
        assert_type(stops, list, 'stops', 'stream')
        for item in stops:
            assert_type(
                item,
                (str, int),
                "some item in parameter 'stops'",
                'stream'
            )

        if self.verbose:
            print_verbose(f'using the following sampler settings for Model.stream:')
            print_verbose(f'max_len_tokens    == {sampler.max_len_tokens}')
            print_verbose(f'temp              == {sampler.temp}')
            print_verbose(f'top_p             == {sampler.top_p}')
            print_verbose(f'min_p             == {sampler.min_p}')
            print_verbose(f'frequency_penalty == {sampler.frequency_penalty}')
            print_verbose(f'presence_penalty  == {sampler.presence_penalty}')
            print_verbose(f'repeat_penalty    == {sampler.repeat_penalty}')
            print_verbose(f'top_k             == {sampler.top_k}')
        
        stop_strs: list[str] = [stop for stop in stops if isinstance(stop, str)]
        stop_token_ids: list[int] = [tok_id for tok_id in stops if isinstance(tok_id, int)]
        stopping_criteria = None
        if stop_token_ids != list():
            def stop_on_token_ids(tokens, *args, **kwargs):
                return tokens[-1] in stop_token_ids
            stopping_criteria = StoppingCriteriaList([stop_on_token_ids])
        
        assert_model_is_loaded(self)
        return self.llama.create_completion(
            prompt,
            max_tokens=sampler.max_len_tokens,
            temperature=sampler.temp,
            top_p=sampler.top_p,
            min_p=sampler.min_p,
            frequency_penalty=sampler.frequency_penalty,
            presence_penalty=sampler.presence_penalty,
            repeat_penalty=sampler.repeat_penalty,
            top_k=sampler.top_k,
            stream=True,
            stop=stop_strs,
            stopping_criteria=stopping_criteria
        )


    def stream_print(
        self,
        prompt: Union[str, list[int]],
        stops: list[Union[str, int]] = list(),
        sampler: SamplerSettings = DefaultSampling,
        end: str = "\n",
        file: _SupportsWriteAndFlush = sys.stdout,
        flush: bool = True
    ) -> str:
        """
        Given a prompt, stream text to a file as it is generated, and return
        the generated string. The returned string does not include the `end`
        parameter.

        prompt: The text from which to generate

        The following parameters are optional:
        - stops: A list of strings and/or token IDs at which to end the generation early
        - sampler: The SamplerSettings object used to control text generation
        - end: A string to print after the generated text
        - file: The file where text should be printed
        - flush: Whether to flush the stream after each token
        """
        
        token_generator = self.stream(
            prompt=prompt,
            stops=stops,
            sampler=sampler
        )

        res = ''
        for i in token_generator:
            tok = i['choices'][0]['text']
            print(tok, end='', file=file, flush=flush)
            res += tok

        # print `end`, and always flush stream after generation is done
        print(end, end='', file=file, flush=True)

        return res


    def ingest(self, text: Union[str, list[int]]) -> None:
        """
        Ingest the given text into the model's cache
        """

        assert_model_is_loaded(self)
        self.llama.create_completion(
            text,
            max_tokens=1,
            temperature=0.0
        )
    

    def candidates(
        self,
        prompt: str,
        k: int,
        temp: Optional[float] = None
    ) -> list[tuple[str, np.floating]]:
        """
        Given prompt `str` and k `int`, return a sorted list of the
        top k candidates for most likely next token, along with their
        normalized probabilities.

        Optionally apply temperature `temp` to the probabilities.
        """

        # TODO: this functions uses 7GB of memory
        # maybe switch from lists to numpy arrays?

        assert_type(prompt, str, 'prompt', 'candidates')
        assert_type(k, int, 'k', 'candidates')
        assert_type(temp, (float, NoneType), 'temp', 'candidates')
        if not 1 <= k <= len(self.vocab):
            raise ValueError(
                f"candidates: k should be between 1 and {len(self.vocab)} inclusive"
            )

        assert_model_is_loaded(self)
        prompt_tokens = self.llama.tokenize(prompt.encode('utf-8', errors='ignore'))
        self.llama.reset() # reset model state
        self.llama.eval(prompt_tokens)
        scores = self.llama.scores[len(prompt_tokens) - 1]

        # len(self.llama.scores) == self.context_length
        # len(self.llama.scores[i]) == len(self.vocab)
        
        # normalize scores with softmax
        # must normalize over all logits, not just top k
        if self.verbose:
            print_verbose(f'calculating softmax over {len(scores)} values')
        normalized_scores: list[np.floating] = list(
            softmax(z=scores, T=temp, dtype=None)
        )

        # construct the final list
        i = 0
        token_probs_list: list[tuple[str, np.floating]] = list()
        for tok_str in self.vocab:
            token_probs_list.append((tok_str, normalized_scores[i]))
            i += 1

        # return token_probs_list, sorted by probability, only top k
        return heapq.nlargest(k, token_probs_list, key=lambda x:x[1])


    def print_candidates(
        self,
        prompt: str,
        k: int,
        temp: Optional[float] = None,
        file: _SupportsWriteAndFlush = sys.stdout,
    ) -> None:
        """
        Like `Model.candidates()`, but print the values instead
        of returning them
        """

        for _tuple in self.candidates(prompt, k, temp):
            print(
                f"token {_tuple[0]!r} has probability {_tuple[1]}",
                file=file,
            )


def assert_model_is_loaded(model: Model) -> None:
    """
    Ensure the model is fully constructed, such that
    `model.llama._model.model is not None` is guaranteed to be `True`

    Raise ModelUnloadedException otherwise
    """
    try:
        if model.llama._model.model is not None:
            return
    except AttributeError:
        pass

    if model is None:
        exc = ModelUnloadedException(
            "model is None"
        )
    elif not hasattr(model, 'llama'):
        exc = ModelUnloadedException(
            "model has no attribute 'llama'"
        )
    elif not hasattr(model.llama, '_model'):
        exc = ModelUnloadedException(
            "llama_cpp.Llama instance has no attribute '_model'"
        )
    elif not hasattr(model.llama._model, 'model'):
        exc = ModelUnloadedException(
            "llama_cpp._internals._LlamaModel instance has no attribute 'model'"
        )
    elif model.llama._model.model is None:
        exc = ModelUnloadedException(
            "llama_cpp._internals._LlamaModel.model is None"
        )
    else:
        exc = ModelUnloadedException(     # likely unreachable
            "model is not loaded"
        )
    exc.add_note('Are you trying to use a model that has been unloaded?')
    raise exc
