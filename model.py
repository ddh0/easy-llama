# model.py
# Python 3.11.6

"""Submodule containing the Model class to work with language models"""

import llama_cpp
import globals
import sys
import os

from samplers import SamplerSettings, DefaultSampling
from utils import print_warning, verify_backend
from typing import Generator, Optional, TextIO
from gguf_reader import GGUFReader

# for typing of Model.stream_print() parameter `file`
class _SupportsWriteAndFlush(TextIO):
    pass

class Model(object):
    """
    A high-level abstraction of a llama model

    This is just a brief overview of easy_llama.Model.
    To see a full description of each method and its parameters,
    call help(Model), or see the relevant docstring.

    The following methods are available:
    - .generate(): given a prompt, return a generated string
    - .stream(): like .generate() but returns a generator that yields
    dicts containing tokens. Subscript the dict with
    `['choices'][0]['text']` to get the token string
    - .get_length(): return the length of a given string in tokens
    - .trim(): trim a given string to this model's context length
    - .next_candidates(): return a list of candidates for the most likely
    next token

    The following attributes are available:
    - .metadata: the GGUF metadata read from the model file, dict
    - .context_length: the context length of the model in tokens, int
    - .llama: the raw llama_cpp.Llama instance
    """

    def __init__(
            self,
            model_path: str,
            context_length: Optional[int] = None
        ):
        """
        Given the path to a GGUF file, construct a Model instance

        The model must be in GGUF format.

        easy_llama will automatically determine the model's trained
        context length from the GGUF metadata. Optionally, you can
        specifiy another context length in the `context_length`
        parameter.
        """

        assert isinstance(model_path, str), \
            f"model_path should be a string, not {type(model_path)}"
        assert not os.path.isdir(model_path), \
            f"the given model_path '{model_path}' is a directory, not a file"
        assert os.path.exists(model_path), \
            f"the given model_path '{model_path}' does not exist"
        assert isinstance(context_length, (int, type(None))), \
            f"context_length should be int or None, not {type(context_length)}"

        self.metadata = GGUFReader.load_metadata(self, model_path)

        if 'llama.context_length' in self.metadata:
            n_ctx_train = self.metadata['llama.context_length']
        elif 'stablelm.context_length' in self.metadata:
            n_ctx_train = self.metadata['stablelm.context_length']
        elif 'phi2.context_length' in self.metadata:
            n_ctx_train = self.metadata['phi2.context_length']
        else:
            raise KeyError(
                "GGUF file does not specify a context length"
            )

        if 'llama.rope.freq_base' in self.metadata:
            rope_freq_base_train = self.metadata['llama.rope.freq_base']
        elif 'stablelm.rope.freq_base' in self.metadata:
            rope_freq_base_train = self.metadata['stablelm.rope.freq_base']
        elif 'phi2.rope.freq_base' in self.metadata:
            rope_freq_base_train = self.metadata['phi2.rope.freq_base']
        else:
            rope_freq_base_train = None

        if rope_freq_base_train is None and context_length is not None:
            if context_length > n_ctx_train:
                raise ValueError(
                    'unable to load model with greater than native ' + \
                    f'context length ({context_length} > {n_ctx_train}) ' + \
                    'because model does not specify a RoPE frequency ' + \
                    f'base. try again with `context_length={n_ctx_train}`'
                )

        if rope_freq_base_train is None or context_length is None or \
            context_length <= n_ctx_train:
            # no need to do context scaling, load model normally

            if context_length is None:
                self.context_length = n_ctx_train
            else:
                self.context_length = context_length
            rope_scaling_type = llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED
            rope_freq_base = 0

        elif context_length > n_ctx_train:
            # multiply rope_freq_base according to requested context length
            # because context length > n_ctx_train and rope freq base is known

            rope_scaling_type = llama_cpp.LLAMA_ROPE_SCALING_LINEAR
            rope_freq_base = (context_length/n_ctx_train)*rope_freq_base_train
            self.context_length = context_length
            
            print_warning(
                'chosen context length is ' + \
                'greater than native context length ' + \
                f'({context_length} > {n_ctx_train}), ' + \
                'freq_base has been changed from ' + \
                f'{rope_freq_base_train} to {rope_freq_base}'
            )

        n_batch = os.cpu_count() * 16
        n_threads = max(os.cpu_count()//2, 1)
        n_threads_batch = os.cpu_count()

        # Set parameters to valid values based on backend
        globals.BACKEND, globals.NUM_GPU_LAYERS, mul_mat_q, \
        mmap, mlock = verify_backend(
            backend=globals.BACKEND,
            num_gpu_layers=globals.NUM_GPU_LAYERS
        )

        self.llama: llama_cpp.Llama = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=self.context_length,
            n_gpu_layers=globals.NUM_GPU_LAYERS,
            use_mmap=mmap,
            use_mlock=mlock,
            logits_all=False,
            n_batch=n_batch,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            rope_scaling_type=rope_scaling_type,
            rope_freq_base=rope_freq_base,
            mul_mat_q=mul_mat_q,
            verbose=globals.VERBOSE,
        )
        
        if globals.VERBOSE:
            print("-------------------- easy_llama.Model ----------------------")
            print(f"{model_path}")
            print(f"global: BACKEND              == {globals.BACKEND}")
            print(f"global: NUM_GPU_LAYERS       == {globals.NUM_GPU_LAYERS}")
            print(f" param: MUL_MAT_Q            == {mul_mat_q}")
            print(f" param: MMAP                 == {mmap}")
            print(f" param: MLOCK                == {mlock}")
            print(f" param: n_batch              == {n_batch}")
            print(f" param: n_threads            == {n_threads}")
            print(f" param: n_threads_batch      == {n_threads_batch}")
            print(f"  gguf: n_ctx_train          == {n_ctx_train}")
            print(f" param: self.context_length  == {self.context_length}")
            print(f"  gguf: rope_freq_base_train == {rope_freq_base_train}")
            print(f" param: rope_freq_base       == {rope_freq_base}")
    
    def __enter__(self):
        return self

    def __exit__(self, *_):
        # this unloads the model from memory, which is the important part
        # (unless an unexpected reference is made to Model.llama)
        # however, Model object might still exist outside of a `with` block
        # unsure if/how to fix that
        self.llama = None
        self = None
        del self
        return None
    
    def __call__(
            self,
            prompt: str,
            stops: Optional[list[str]] = None,
            sampler: SamplerSettings = DefaultSampling
        ) -> str:
        """
        `Model('some text')` is a shorthand for `Model.generate('some text')`
        """
        return self.generate(prompt, stops, sampler)

    def trim(self,
             text: str,
             overwrite: Optional[str] = None
        ) -> str:

        """
        Trim the given text to the context length of this model,
        leaving room for two extra tokens.

        Optionally overwrite the oldest tokens with the text given in the
        'overwrite'parameter, which is useful for keeping the system prompt
        in context.

        Does nothing if the text is equal to or shorter than
        (context_length - 2).
        """
        trim_length = self.context_length - 2
        tokens_list = self.llama.tokenize(
            text.encode("utf-8", errors="ignore")
        )

        if len(tokens_list) <= trim_length:
            # TODO: ensure overwrite
            return text

        if len(tokens_list) > trim_length and overwrite is None:
            # Cut to context length
            tokens_list = tokens_list[-trim_length:]
            return self.llama.detokenize(tokens_list).decode(
                "utf-8",
                errors="ignore"
            )

        if len(tokens_list) > self.context_length and overwrite is not None:
            # Cut to context length and overwrite the oldest tokens with
            # overwrite
            tokens_list = tokens_list[-trim_length:]
            overwrite_tokens = self.llama.tokenize(overwrite.encode(
                "utf-8",
                errors="ignore"
                )
            )
            tokens_list[0 : len(overwrite_tokens)] = overwrite_tokens
            return self.llama.detokenize(tokens_list).decode(
                "utf-8",
                errors="ignore"
            )

    def get_length(self, text: str) -> int:
        """
        Return the length of the given text in tokens,
        according to this model.
        """
        return len(self.llama.tokenize(
            text.encode(
                "utf-8",
                errors="ignore"
                )
            ))

    def generate(
            self,
            prompt: str,
            stops: Optional[list[str]] = None,
            sampler: SamplerSettings = DefaultSampling
            ) -> str:
        """
        Given a prompt, return a generated string.

        The following parameter is optional:

        stops: list[str] | None: a list of strings at which to end the
        generation early
        """

        assert isinstance(prompt, str), \
            f"prompt should be string, not {type(prompt)}"
        if isinstance(stops, list):
            for stopping_string in stops:
                assert isinstance(stopping_string, str), \
                    f"item {stopping_string} in stops list is not a string"
        else:
            assert (stops is None), \
                f"stops should be list[str] or None, not {type(stops)}"

        if globals.VERBOSE:
            print(f'easy_llama: using the following sampler settings for generation')
            print(f'easy_llama: sampler.max_len_tokens   == {sampler.max_len_tokens}')
            print(f'easy_llama: sampler.temp             == {sampler.temp}')
            print(f'easy_llama: sampler.top_p            == {sampler.top_p}')
            print(f'easy_llama: sampler.min_p            == {sampler.min_p}')
            print(f'easy_llama: sampler.presence_penalty == {sampler.presence_penalty}')
            print(f'easy_llama: sampler.repeat_penalty   == {sampler.repeat_penalty}')
            print(f'easy_llama: sampler.top_k            == {sampler.top_k}')
            print()
        
        return self.llama.create_completion(
            prompt,
            max_tokens=sampler.max_len_tokens,
            temperature=sampler.temp,
            top_p=sampler.top_p,
            min_p=sampler.min_p,
            presence_penalty=sampler.presence_penalty,
            repeat_penalty=sampler.repeat_penalty,
            top_k=sampler.top_k,
            stop=stops
        )['choices'][0]['text']
    

    def stream(
            self,
            prompt: str,
            stops: Optional[list[str]] = None,
            sampler: SamplerSettings = DefaultSampling
        ) -> Generator:

        """
        Given a prompt, return a generator that yields dicts containing tokens.

        To get the token string itself, subscript the dict with:

        `['choices'][0]['text']`

        The following parameter is optional:

        stops: list[str] | None: a list of strings at which to end the
        generation early

        See also `Model.stream_print()`
        """

        assert isinstance(prompt, str), \
            f"prompt should be string, not {type(prompt)}"
        if isinstance(stops, list):
            for stopping_string in stops:
                assert isinstance(stopping_string, str), \
                    f"item {stopping_string} in stops list is not a string"
        else:
            assert stops is None, \
                f"stops should be list[str] or None, not {type(stops)}"

        if globals.VERBOSE:
            print(f'easy_llama: using the following sampler settings for generation')
            print(f'easy_llama: sampler.max_len_tokens   == {sampler.max_len_tokens}')
            print(f'easy_llama: sampler.temp             == {sampler.temp}')
            print(f'easy_llama: sampler.top_p            == {sampler.top_p}')
            print(f'easy_llama: sampler.min_p            == {sampler.min_p}')
            print(f'easy_llama: sampler.presence_penalty == {sampler.presence_penalty}')
            print(f'easy_llama: sampler.repeat_penalty   == {sampler.repeat_penalty}')
            print(f'easy_llama: sampler.top_k            == {sampler.top_k}')
            print()

        return self.llama.create_completion(
            prompt,
            max_tokens=sampler.max_len_tokens,
            temperature=sampler.temp,
            top_p=sampler.top_p,
            min_p=sampler.min_p,
            presence_penalty=sampler.presence_penalty,
            repeat_penalty=sampler.repeat_penalty,
            top_k=sampler.top_k,
            stream=True,
            stop=stops
        )


    def stream_print(
            self,
            prompt: str,
            stops: Optional[list[str]] = None,
            sampler: SamplerSettings = DefaultSampling,
            end: str = "\n",
            file: _SupportsWriteAndFlush = sys.stdout,
            flush: bool = True
    ) -> None:
        """
        `Model.stream_print(...)` is a shorthand for:
        ```
        s = Model.stream(prompt, stops=stops, sampler=sampler)
        for i in s:
            tok = i['choices'][0]['text']
            print(tok, end='', file=file, flush=flush)
        print(end, end='', file=file, flush=True)
        ```
        Once finished, returns the complete generated string. The returned
        string does not include the `end` parameter.
        """
        
        tok_gen = self.stream(
            prompt=prompt,
            stops=stops,
            sampler=sampler
        )

        res = ''
        for i in tok_gen:
            tok = i['choices'][0]['text']
            print(tok, end='', file=file, flush=flush)
            res += tok

        # always flush stream after generation is done
        print(end, end='', file=file, flush=True)

        return res


    def ingest(self, text: str) -> None:
        """
        Ingest the given text into the model's cache by calling a
        single-token completion and discarding the result
        """

        self.llama.create_completion(
            text,
            max_tokens=1,
        )
    

    def next_candidates(
            self,
            prompt: str,
            k: int
        ) -> list[str]:
        """
        Given prompt (str) and k (int), return a sorted list of the
        top k candidates for most likely next token
        """

        # TODO
        # Llama.logits_to_logprobs()[tok_id]
        # Llama.eval(tokens_list_ints)
        pass