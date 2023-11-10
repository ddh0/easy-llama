# easy_llama.py
# Python 3.11.6

"""
Natural text generation in Python, made easy

https://github.com/ddh0/easy-llama/
"""

# TODO: Thread.add_message as shorthand for
#       T.messages.append(T.create_message) ?
# TODO: function to transfer a list of messages between disk / models,
#       handle token count
# TODO: Model.next_candidates() -> list[str]

import os
import sys
import time
import struct
from typing import Generator
from enum import IntEnum

# this will be set per pip package
# 'metal' | 'cuda' | 'rocm' | 'cpu'
# NOTE TO END USER: DO NOT TOUCH!
_backend = None

NUM_GPU_LAYERS = 1
MUL_MAT_Q = True

# Alpha value used in contrastive search
# Between 0.0 and 1.0 inclusive
# Lower values lead to more predictable outputs
# Higher values lead to more varied or creative outputs
# If you're not sure, leave it at the default of 0.5
ALPHA: float = 0.5

# Number of tokens to sample from
# If you're not sure, leave it at the default of 4
TOP_K: int = 4

# Penalize repetion of tokens already in context
# If you're not sure, leave it at the default of 1.1
REPEAT_PENALTY: float = 1.1

# Max length of each generation in tokens
MAX_LEN_TOKENS: int = None

# Do not suppress llama.cpp output
VERBOSE: bool = False

# Leave at -1 for random seed
SEED: int = -1

# Warnings are infrequent and contain helpful information
SUPPRESS_WARNINGS: bool = False


class _GGUFReader:
    """
    Peek at file header for GGUF metadata

    Raise ValueError if file is not GGUF or is outdated

    Credit to oobabooga for the parts of the code in this class

    Format spec: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
    """

    class GGUFValueType(IntEnum):
        UINT8 = 0
        INT8 = 1
        UINT16 = 2
        INT16 = 3
        UINT32 = 4
        INT32 = 5
        FLOAT32 = 6
        BOOL = 7
        STRING = 8
        ARRAY = 9
        UINT64 = 10
        INT64 = 11
        FLOAT64 = 12

    _simple_value_packing = {
        GGUFValueType.UINT8: "<B",
        GGUFValueType.INT8: "<b",
        GGUFValueType.UINT16: "<H",
        GGUFValueType.INT16: "<h",
        GGUFValueType.UINT32: "<I",
        GGUFValueType.INT32: "<i",
        GGUFValueType.FLOAT32: "<f",
        GGUFValueType.UINT64: "<Q",
        GGUFValueType.INT64: "<q",
        GGUFValueType.FLOAT64: "<d",
        GGUFValueType.BOOL: "?",
    }

    value_type_info = {
        GGUFValueType.UINT8: 1,
        GGUFValueType.INT8: 1,
        GGUFValueType.UINT16: 2,
        GGUFValueType.INT16: 2,
        GGUFValueType.UINT32: 4,
        GGUFValueType.INT32: 4,
        GGUFValueType.FLOAT32: 4,
        GGUFValueType.UINT64: 8,
        GGUFValueType.INT64: 8,
        GGUFValueType.FLOAT64: 8,
        GGUFValueType.BOOL: 1,
    }

    def get_single(self, value_type, file):
        if value_type == _GGUFReader.GGUFValueType.STRING:
            value_length = struct.unpack("<Q", file.read(8))[0]
            value = file.read(value_length)
            value = value.decode("utf-8")
        else:
            type_str = _GGUFReader._simple_value_packing.get(value_type)
            bytes_length = _GGUFReader.value_type_info.get(value_type)
            value = struct.unpack(type_str, file.read(bytes_length))[0]

        return value

    def load_metadata(self, fname) -> dict:
        metadata = {}
        with open(fname, "rb") as file:
            GGUF_MAGIC = file.read(4)
            GGUF_VERSION = struct.unpack("<I", file.read(4))[0]
            # ti_data_count = struct.unpack("<Q", file.read(8))[0]
            file.read(8)
            kv_data_count = struct.unpack("<Q", file.read(8))[0]

            if GGUF_MAGIC != b"GGUF":
                raise ValueError(
                    "easy_llama: your model file is not a valid GGUF file " \
                    + f"(magic number mismatch, got {GGUF_MAGIC}, " \
                    + "expected b'GGUF')"
                )

            if GGUF_VERSION == 1:
                raise ValueError(
                    "easy_llama: your model file reports GGUF version 1, " \
                    + "but only versions 2 and above are supported. " \
                    + "re-convert your model or download a newer version"
                )

            for _ in range(kv_data_count):
                key_length = struct.unpack("<Q", file.read(8))[0]
                key = file.read(key_length)

                value_type = _GGUFReader.GGUFValueType(
                    struct.unpack("<I", file.read(4))[0]
                )
                if value_type == _GGUFReader.GGUFValueType.ARRAY:
                    ltype = _GGUFReader.GGUFValueType(
                        struct.unpack("<I", file.read(4))[0]
                    )
                    length = struct.unpack("<Q", file.read(8))[0]
                    for _ in range(length):
                        _ = _GGUFReader.get_single(self, ltype, file)
                else:
                    value = _GGUFReader.get_single(self, value_type, file)
                    metadata[key.decode()] = value

        return metadata


class _suppress_if_not_verbose(object):
    """
    Suppress stdout and stderr if easy_llama.VERBOSE is False

    This prevents llama.cpp's very detailed output from being displayed

    Changing VERBOSE inside the WITH block may result in stdout and stderr
    being stuck to /dev/null, or other undefined behaviour

    See https://github.com/abetlen/llama-cpp-python/issues/478
    """
    
    def __enter__(self):
        if not VERBOSE:
            self.outnull_file = open(os.devnull, "w")
            self.errnull_file = open(os.devnull, "w")

            self.old_stdout_fileno_undup = sys.stdout.fileno()
            self.old_stderr_fileno_undup = sys.stderr.fileno()

            self.old_stdout_fileno = os.dup(sys.stdout.fileno())
            self.old_stderr_fileno = os.dup(sys.stderr.fileno())

            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr

            os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
            os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

            sys.stdout = self.outnull_file
            sys.stderr = self.errnull_file
            return self

        else:
            return self

    def __exit__(self, *_):
        if not VERBOSE:
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr

            os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
            os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

            os.close(self.old_stdout_fileno)
            os.close(self.old_stderr_fileno)

            self.outnull_file.close()
            self.errnull_file.close()
        else:
            return


def _print_warning(text: str) -> str:
    if not SUPPRESS_WARNINGS:
        print("easy_llama: warning:", text, file=sys.stderr)


with _suppress_if_not_verbose():
    import llama_cpp

def _verify_backend():
    """
    Verify that _backend is valid,
    and modify NUM_GPU_LAYERS and mul_mat_q if required

    This is not done on import because user must be able to set _backend
    (and maybe NUM_GPU_LAYERS) before loading any model.
    """

    global _backend, NUM_GPU_LAYERS, MUL_MAT_Q

    if _backend is None:
        raise RuntimeError(
            "easy_llama: easy_llama._backend is None. Set _backend " \
            + "to 'metal', 'cuda', 'rocm', or 'cpu' before loading a model"
        )
    
    assert isinstance(_backend, str), \
        'easy_llama: easy_llama._backend must be a string, '\
        + f'not {type(_backend)}'
    
    _backend = _backend.lower()

    if _backend not in ['metal', 'cuda', 'rocm', 'cpu']:
        raise RuntimeError(
            f"easy_llama: easy_llama._backend '{_backend}' is invalid. Set " \
            + "_backend to 'metal', 'cuda', 'rocm', or 'cpu' before " \
            + "loading a model"
        )
    
    # NUM_GPU_LAYERS and MUL_MAT_Q are global variables
    if _backend == 'metal':
        NUM_GPU_LAYERS = 1
        MUL_MAT_Q = True
    elif _backend == 'cuda':
        # Don't set NUM_GPU_LAYERS, let the user configure it
        MUL_MAT_Q = True
    elif _backend == 'rocm':
        # Don't set NUM_GPU_LAYERS, let the user configure it
        MUL_MAT_Q = False
    elif _backend == 'cpu':
        NUM_GPU_LAYERS = 0
        MUL_MAT_Q = True
    

class Model(object):
    """
    A high-level abstraction of a llama model

    This is just a brief overview of easy_llama.Model.
    To see a full description of each method and its parameters,
    call help(Model), or see the relevant docstring.

    The following methods are available:
    - .generate(): return a string generated with Contrastive Search,
    which generates more human-like text
    - .stream(): like .generate() but returns a generator that yields
    dicts containing tokens. Subscript the dict with
    `['choices'][0]['text']` to get the token string
    - .get_length(): return the length of a given string in tokens
    - .trim(): trim a given string to this model's context length
    - .next_candidates(): return a list of candidates for the most likely
    next token

    The following attributes are available:
    - .metadata: the GGUF metadata read from the model file, dict
    - .context_lenth: the native context length of the model in tokens, int
    - .llama: the raw llama_cpp.Llama instance
    """

    def __init__(self, model_path: str, context_length: int = None):
        """
        Given the path to a GGUF file, create a Model instance

        The model must be in GGUF format.

        easy_llama will automatically determine the model's trained
        context length from the GGUF metadata. Optionally, you can
        specifiy another context length in the `context_length`
        parameter.
        """

        assert isinstance(
            model_path, str
        ), f"model_path should be a string, not {type(model_path)}"
        assert not os.path.isdir(
            model_path
        ), f"the given model_path '{model_path}' is a directory, not a file"
        assert os.path.exists(
            model_path
        ), f"the given model_path '{model_path}' does not exist"
        assert isinstance(context_length, (int, type(None))), \
            f'context_length should be int or None, not {type(context_length)}'

        self.metadata = _GGUFReader.load_metadata(self, model_path)

        #if self.metadata["llama.context_length"] < 2048:
        #    _print_warning(
        #        "GGUF metadata reports an unusually small native context " \
        #        + f"length ({self.metadata['llama.context_length']})"
        #    )
        #if self.metadata["llama.context_length"] > 100000:
        #    _print_warning(
        #        "GGUF metadata reports an unusually large native context " \
        #        + f"length ({self.metadata['llama.context_length']})"
        #    )

        n_ctx_train = self.metadata["llama.context_length"]
        if context_length is None:
            self.context_length = n_ctx_train
        else:
            if context_length > n_ctx_train:
                _print_warning('chosen context length is '\
                               + 'greater than native context length '
                               + f'({context_length} > {n_ctx_train}), '
                               + 'expect poor results')
            self.context_length = context_length

        _verify_backend()

        with _suppress_if_not_verbose():
            self.llama: llama_cpp.Llama = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=self.context_length,
                n_gpu_layers=NUM_GPU_LAYERS,
                seed=SEED,
                # mmap False -> Faster inference, model must fit in memory
                use_mmap=False,
                # mlock True -> Force the model to stay in memory
                use_mlock=True,
                logits_all=False,
                n_batch=os.cpu_count() * 64,
                n_threads=max(
                    os.cpu_count() // 2,
                    1
                ),  # optimal if == physical cores or performance cores
                n_threads_batch=os.cpu_count(), # always optimal
                mul_mat_q=MUL_MAT_Q,
                verbose=VERBOSE,
            )

            print("----------------------------------------------------------")
            print(f"easy_llama: _backend            == {_backend}")
            print(f"easy_llama: NUM_GPU_LAYERS      == {NUM_GPU_LAYERS}")
            print(f"easy_llama: MUL_MAT_Q           == {MUL_MAT_Q}")
            print(f"easy_llama: n_ctx_train         == {n_ctx_train}")
            print(f"easy_llama: self.context_length == {self.context_length}")
            print()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # this unloads the model from memory, which is the important part
        # (unless an unexpected reference is made to Model.llama)
        # however, ez.Model object might still exist outside of a `with` block
        # unsure if/how to fix that
        self.llama = None
    
    def __call__(self, prompt: str, stops: list[str] | None = None) -> str:
        """
        `Model('some text')` is a shortcut to `Model.generate('some text')`
        """
        return self.generate(prompt, stops)

    def trim(self, text: str, overwrite: str = None) -> str:
        """
        Trim the given text to the context length of this model,
        leaving room for three extra tokens.

        Optionally overwrite the oldest tokens with the text given in the
        'overwrite'parameter, which is useful for keeping the system prompt
        in context.

        Does nothing if the text is equal to or shorter than
        (context_length - 3).
        """
        trim_length = self.context_length - 3
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
                "utf-8", errors="ignore"
            )

        if len(tokens_list) > self.context_length and overwrite is not None:
            # Cut to context length and overwrite the oldest tokens with
            # overwrite
            tokens_list = tokens_list[-trim_length:]
            overwrite_tokens = self.llama.tokenize(
                overwrite.encode("utf-8", errors="ignore")
            )
            tokens_list[0 : len(overwrite_tokens)] = overwrite_tokens
            return self.llama.detokenize(tokens_list).decode(
                "utf-8", errors="ignore"
            )

    def get_length(self, text: str) -> int:
        """
        Return the length of the given text in tokens,
        according to this model.
        """
        return len(self.llama.tokenize(text.encode("utf-8", errors="ignore")))

    def generate(self, prompt: str, stops: list[str] | None = None) -> str:
        """
        Given a prompt, return a generated string.

        The following parameter is optional:

        stops: list[str] | None: a list of strings at which to end the
        generation early
        """

        assert isinstance(prompt, str), "prompt should be string, not " \
            + f"{type(prompt)}"
        if isinstance(stops, list):
            for item in stops:
                assert isinstance(
                    item, str
                ), f"item {item} in stops list is not a string"
        else:
            assert (
                stops is None
            ), f"stops should be list[str] or None, not {type(stops)}"

        assert isinstance(MAX_LEN_TOKENS, (int, type(None))), \
            'easy_llama.MAX_LEN_TOKENS should be int or None, not' \
            + f'{type(MAX_LEN_TOKENS)}'

        if MAX_LEN_TOKENS is not None:
            if MAX_LEN_TOKENS > self.context_length:
                _print_warning(
                    "MAX_LEN_TOKENS is greater than this model's context " \
                    + "length, expect poor results"
                )
            completion_max_len_tokens = MAX_LEN_TOKENS
        else:
            completion_max_len_tokens = self.context_length
        
        assert isinstance(REPEAT_PENALTY, float), \
            'easy_llama.REPEAT_PENALTY should be float, ' \
            + f'not {type(REPEAT_PENALTY)}'
        
        # see https://github.com/facebookresearch/llama/issues/217
        #if prompt[-1] == " ":
        #    _print_warning(
        #        "prompt has trailing whitespace, this may reduce the " \
        #        + "quality of generations\n"
        #    )

        # _print_warning(f'ALPHA == {ALPHA}\n')

        #with _suppress_if_not_verbose():
        return self.llama.create_completion(
            prompt,
            max_tokens=completion_max_len_tokens,
            top_k=TOP_K,
            presence_penalty=ALPHA,
            repeat_penalty=REPEAT_PENALTY,
            stop=stops,
        )["choices"][0]["text"]
    

    def stream(self, prompt: str, stops: list[str] | None = None) -> Generator:
        """
        Given a prompt, return a generator that yields dicts containing tokens.

        To get the token string itself, subscript the dict with:

        `['choices'][0]['text']`

        The following parameter is optional:

        stops: list[str] | None: a list of strings at which to end the
        generation early
        """

        assert isinstance(prompt, str), "prompt should be string, not " \
            + f"{type(prompt)}"
        if isinstance(stops, list):
            for item in stops:
                assert isinstance(
                    item, str
                ), f"item {item} in stops list is not a string"
        else:
            assert (
                stops is None
            ), f"stops should be list[str] or None, not {type(stops)}"

        assert isinstance(MAX_LEN_TOKENS, (int, type(None))), \
            'easy_llama.MAX_LEN_TOKENS should be int or None, not' \
            + f'{type(MAX_LEN_TOKENS)}'
  
        if MAX_LEN_TOKENS is not None:
            if MAX_LEN_TOKENS > self.context_length:
                _print_warning(
                    "MAX_LEN_TOKENS is greater than this model's context " \
                    + "length, expect poor results"
                )
            completion_max_len_tokens = MAX_LEN_TOKENS
        else:
            completion_max_len_tokens = self.context_length

        # see https://github.com/facebookresearch/llama/issues/217
        #if prompt[-1] == " ":
        #    _print_warning(
        #        "prompt has trailing whitespace, this may reduce the " \
        #        + "quality of generations\n"
        #    )

        # _print_warning(f'ALPHA == {ALPHA}\n')

        #with _suppress_if_not_verbose():
        return self.llama.create_completion(
            prompt,
            max_tokens=completion_max_len_tokens,
            top_k=TOP_K,
            stream=True,
            presence_penalty=ALPHA,
            repeat_penalty=REPEAT_PENALTY,
            stop=stops,
        )
    

    def next_candidates(self, prompt: str, k: int) -> list[str]:
        """
        Given prompt (str) and k (int), return a sorted list of the
        top k candidates for most likely next token
        """

        # LLama.logits_to_logprobs()[tok_id]

        # Llama.eval(tokens_list_ints)
        # Llama.scores is a list of lists (basically)
        # scores[0] is a list of numbers like -4.4852424...
        # the number (float) indicates the probability of the token

        # len(raw_scores) == n_vocab
        # note that eos token is not always the last token in the vocab,
        # especially for fine-tuned models
        # raw_scores[0] -> logprob of token with ID 0
        # raw_scores[1] -> logprob of token with ID 1
        # ...
        # raw_scores[len(n_vocab) - 1] -> ID of last token in vocab

        assert isinstance(prompt, str), "prompt should be string, not " \
            + f"{type(prompt)}"
        self.llama.reset()  # reset model state
        prompt_tokens = self.llama.tokenize(
            prompt.encode("utf-8", errors="ignore")
        )
        self.llama.eval(prompt_tokens)
        raw_scores = self.llama.scores[0]

        all_probs_dict = {}

        for i in range(len(raw_scores)):
            # i_str = self.llama.detokenize(
            # [i]).decode('utf-8', errors='ignore')
            # all_probs_dict[i_str] = raw_scores[i]
            all_probs_dict[i] = raw_scores[i]

        # verified: now all_probs_dict[tok_id] -> tok_logprob
        # and: len(all_probs_dict) == n_vocab

        # sort dict by logprobs, high to low, then keep only the
        # first k entries
        top_k_tok_list: list[tuple[int, float]] = sorted(
            all_probs_dict.items(), key=lambda x: x[1], reverse=True
        )[:k]
        # top_k_tok_list = sorted(
        # all_probs_dict, key=all_probs_dict.get, reverse=True
        # )[:k]

        top_k_str_list = []

        for tok in top_k_tok_list:
            top_k_str_list.append(
                self.llama.detokenize([tok[0]]).decode(
                    "utf-8",
                    errors="ignore"
                    )
            )

        # top_k_str_list.reverse()

        return top_k_str_list


class Thread(object):

    def __init__(self,
                 model: Model,
                 format: dict,
                 timestamps: bool = False,
                 amnesiac: bool = False,
                 smart_context: bool = False
                 ):
        
        assert isinstance(model, Model), "Thread: model should be an " \
            + f"instance of easy_llama.Model, not {type(model)}"

        assert isinstance(
            format, dict
        ), f"Thread: format should be dict, not {type(format)}"

        # Q: why don't you just check if the format is in the formats list?
        # A: user should be able to create and use their own format without
        # mangling the default formats

        try:
            format["system_prefix"]
            format["system_content"]
            format["system_postfix"]
            format["user_prefix"]
            format["user_content"]
            format["user_postfix"]
            format["bot_prefix"]
            format["bot_content"]
            format["bot_postfix"]
            format["stops"]
        except KeyError as e:
            e.add_note(
                "Thread: format is missing one or more required keys, see " \
                + "easy_llama.blank for an example"
            )
            raise

        assert isinstance(
            format["stops"], (list, type(None))
        ), "Thread: format['stops'] should be list[str] or None, " \
            + f"not {type(format['stops'])}"

        assert isinstance(timestamps, bool), "Thread: timestamps should be " \
            + f"True or False, not '{timestamps}'"
        
        assert isinstance(amnesiac, bool), "Thread: amnesiac should be " \
            + f"True or False, not '{amnesiac}'"
        
        if amnesiac and timestamps:
            raise RuntimeError('Thread: amnesiac threads are not '
                               + 'compatible with timestamps')
        
        if amnesiac and smart_context:
            raise RuntimeError('Thread: amnesiac threads are not '
                               + 'compatible with smart_context')
        
        self.model: Model = model
        self.format: dict = format
        self.enable_timestamps: bool = timestamps
        self.amnesiac: bool = amnesiac
        self.messages: list[dict] = [
            self.create_message("system", self.format["system_content"])
        ]
        self.smart_context: bool = smart_context
        self.smart_context_messages: list[dict] = [
            self.create_message("system", self.format["system_content"])
        ]
        self.smart_context_state: llama_cpp.LlamaState = None
        self.main_context_state: llama_cpp.LlamaState = None
        self.smart_context_active: bool = False
    

    def set_smart_context_state(self) -> None:
        """Switch the model to use the smart context state"""
        assert self.smart_context
        assert not self.amnesiac
        assert not self.enable_timestamps
        if self.smart_context_active:
            return
        self.main_context_state = self.model.llama.save_state()
        if self.smart_context_state is not None:
            self.model.llama.load_state(self.smart_context_state)
        else:
            self.model.llama.reset()
        self.smart_context_active = True
    

    def set_main_context_state(self) -> None:
        """Switch the model to use the main context state"""
        assert self.smart_context
        assert not self.amnesiac
        assert not self.enable_timestamps
        if not self.smart_context_active:
            return
        self.smart_context_state = self.model.llama.save_state()
        if self.main_context_state is not None:
            self.model.llama.load_state(self.main_context_state)
        else:
            self.model.llama.reset()
        self.smart_context_active = False


    def create_message(self, role: str, content: str) -> dict:
        assert role.lower() in ["system", "user", "bot"], \
            "create_message: role should be 'system', 'user', or" \
            + f"'bot', not '{role.lower()}'"

        assert isinstance(
            content, str
        ), f"create_message: content should be str, not {type(content)}"

        if role.lower() == "system":
            return {
                "prefix": self.format["system_prefix"],
                "content": content if not self.enable_timestamps else
                    time.strftime("It is %I:%M %p on %A, %b %e, %Y. ")
                     + content,
                "postfix": self.format["system_postfix"],
                "length": self.model.get_length(
                    self.format["system_prefix"]
                    + content
                    + self.format["system_postfix"]
                ),
            }
        elif role.lower() == "user":
            return {
                "prefix": self.format["user_prefix"],
                "content": content
                if not self.enable_timestamps
                else time.strftime("[at %a %I:%M %p] ") + content,
                "postfix": self.format["user_postfix"],
                "length": self.model.get_length(
                    self.format["user_prefix"] + content + \
                        self.format["user_postfix"]
                ),
            }
        elif role.lower() == "bot":
            return {
                "prefix": self.format["bot_prefix"],
                "content": content
                if not self.enable_timestamps
                else time.strftime("[at %a %I:%M %p] ") + content,
                "postfix": self.format["bot_postfix"],
                "length": self.model.get_length(
                    self.format["bot_prefix"] + content + \
                        self.format["bot_postfix"]
                ),
            }


    def update_smart_context(self, messages: list[dict]) -> None:
        """
        Set self.smart_context_state and self.smart_context_messages.

        TODO:
        Do nothing until context length is half full.
        When context length is half full, for each update:
            start building list starting at midpoint of current context
            eval inf_str(next_context_messages)
        At near full context, set
            Thread.messages = Thread.next_content_messages
            Thread.next_context_messages = None
            Reset context length budget
        """
        self.set_smart_context_state()

        system_message = messages[0]
        context_len_budget = 1024
        context_len_budget -= system_message['length']

        self.smart_context_messages: list[dict] = [system_message]

        # iterate over messages from newest to oldest until
        # context_len_budget is exceeded
        for message in reversed(messages[1:]):
            context_len_budget -= message['length']
            if context_len_budget <= 0:
                break
            # keep sys msg at index 0
            self.smart_context_messages.insert(1, message)
            # now the list is in chronological orger
        
        self.set_main_context_state()


    def inference_str_from_messages(self, messages: list[dict]) -> str:

        system_message = messages[0]
        sys_msg_str = (
            system_message["prefix"]
            + system_message["content"]
            + system_message["postfix"]
        )
        
        # in amnesiac threads, the model only sees the system message
        # and the previous message
        if self.amnesiac:
            inf_str = ''
            last_msg = messages[-1]
            last_msg_str = (
                last_msg['prefix']
                + last_msg['content']
                + last_msg['postfix']
                )
            inf_str = (
                sys_msg_str
                + last_msg_str
                + self.format['bot_prefix']
                )
            return inf_str
        
        # bos + eos + off-by-one errors == 3 (better safe than sorry)
        context_len_budget = self.model.context_length - 3
        context_len_budget -= system_message["length"]
        context_len_budget -= self.model.get_length(self.format["bot_prefix"])

        # start at most recent message and work backwards up the history
        # excluding system message. once we exceed the model's context length,
        # break without including that message
        inf_str = ''
        for message in reversed(messages[1:]):
            context_len_budget -= message["length"]
            if context_len_budget <= 0:
                break
            msg_str = (
                message['prefix']
                + message['content']
                + message['postfix']
                )
            inf_str = msg_str + inf_str
        inf_str = sys_msg_str + inf_str
        inf_str += self.format["bot_prefix"]
        if self.enable_timestamps:
            inf_str += time.strftime("[at %a %I:%M %p]")
        return inf_str


    def send(self, prompt: str) -> str:
        assert isinstance(
            prompt, str
        ), f"Thread.send: prompt should be str, not {type(prompt)}"
        assert prompt != "", "Thread.send: empty prompts are not allowed " \
            + "in threads"

        self.messages.append(self.create_message("user", prompt))
        output = self.model.generate(
            self.inference_str_from_messages(), stops=self.format["stops"]
        )
        self.messages.append(self.create_message("bot", output))

        return output


    def interact(self) -> None:
        print()
        try:
            while True:
                #print(f'DEBUG: len is {len(self.messages)}\n')
                if not self.smart_context:
                    self.set_main_context_state()
                    prompt = input("  > ")
                    print()
                    if prompt == "":
                        # another assistant message
                        token_generator = self.model.stream(
                            self.inference_str_from_messages(self.messages),
                            stops=self.format["stops"],
                        )

                        output = ""
                        for i in token_generator:
                            token = i["choices"][0]["text"]
                            output += token
                            print(token, end="", flush=True)

                        self.messages.append(
                            self.create_message("bot", output)
                            )

                    else:
                        self.messages.append(
                            self.create_message("user", prompt)
                            )

                        
                        token_generator = self.model.stream(
                            self.inference_str_from_messages(self.messages),
                            stops=self.format["stops"],
                        )

                        output = ""
                        for i in token_generator:
                            token = i["choices"][0]["text"]
                            output += token
                            print(token, end="", flush=True)

                        self.messages.append(
                            self.create_message("bot", output)
                        )

                    if output.endswith("\n\n"):
                        pass
                    elif output.endswith("\n"):
                        print()
                    else:
                        print("\n")
            
                else: # smart context
                    self.set_smart_context_state()
                    prompt = input(" -> ")

                    print()

                    if prompt == "":
                        # another assistant message
                        self.update_smart_context(self.messages)
                        token_generator = self.model.stream(
                            self.inference_str_from_messages(
                                self.smart_context_messages
                            ),
                            stops=self.format["stops"],
                        )

                        output = ""
                        for i in token_generator:
                            token = i["choices"][0]["text"]
                            output += token
                            print(token, end="", flush=True)

                        self.messages.append(
                            self.create_message("bot", output)
                            )
                    
                    else:
                        self.messages.append(
                            self.create_message("user", prompt)
                            )
                        
                        self.update_smart_context(self.messages)

                        token_generator = self.model.stream(
                            self.inference_str_from_messages(
                                self.smart_context_messages
                            ),
                            stops=self.format["stops"],
                        )

                        output = ""
                        for i in token_generator:
                            token = i["choices"][0]["text"]
                            output += token
                            print(token, end="", flush=True)

                        self.messages.append(
                            self.create_message("bot", output)
                        )
                    
                    if output.endswith("\n\n"):
                        pass
                    elif output.endswith("\n"):
                        print()
                    else:
                        print("\n")

        except KeyboardInterrupt:
            print("\n")
            if self.smart_context_active:
                self.update_smart_context(self.messages)
                self.set_main_context_state()
            return

    def reset(self) -> None:
        self.messages: list[dict] = [
            self.create_message("system", self.format["system_content"])
        ]
        self.smart_context_messages: list[dict] = [
            self.create_message("system", self.format["system_content"])
        ]
        self.main_context_state: llama_cpp.LlamaState = None
        self.smart_context_state: llama_cpp.LlamaState = None
        self.smart_context_active: bool = False
        self.model.llama.reset()


blank = {
    "system_prefix": "",
    "system_content": "",
    "system_postfix": "",
    "user_prefix": "",
    "user_content": "",
    "user_postfix": "",
    "bot_prefix": "",
    "bot_content": "",
    "bot_postfix": "",
    "stops": [],
}

# https://github.com/openai/openai-python/blob/main/chatml.md
chatml = {
    "system_prefix": "<|im_start|>system\n",
    "system_content": "",
    "system_postfix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>user\n",
    "user_content": "",
    "user_postfix": "<|im_end|>\n",
    "bot_prefix": "<|im_start|>assistant\n",
    "bot_content": "",
    "bot_postfix": "<|im_end|>\n",
    "stops": ["<|im_end|>", "<|im_start|>"],
}

# https://huggingface.co/blog/llama2
# system message relaxed to avoid undue refusals
llama2chat = {
    "system_prefix": "<s>[INST] <<SYS>>\n",
    "system_content": "You are a helpful AI assistant.",
    "system_postfix": "\n<</SYS>>\n\n",
    "user_prefix": "",
    "user_content": "",
    "user_postfix": " [/INST]",
    "bot_prefix": " ",
    "bot_content": "",
    "bot_postfix": " </s><s>[INST] ",
    "stops": ["</s>", "[INST]"],
}

# https://github.com/tatsu-lab/stanford_alpaca
alpaca = {
    "system_prefix": "",
    "system_content": "Below is an instruction that describes a task. " \
    + "Write a response that appropriately completes the request.",
    "system_postfix": "\n\n",
    "user_prefix": "### Instruction:\n",
    "user_content": "",
    "user_postfix": "\n\n",
    "bot_prefix": "### Response:\n",
    "bot_content": "",
    "bot_postfix": "\n\n",
    "stops": ["###", "Instruction:", "\n\n\n"],
}

# this is the official vicuna. it is often butchered in various ways,
# most commonly by adding line breaks
# https://github.com/flu0r1ne/FastChat/blob/main/docs/vicuna_weights_version.md
vicuna_lmsys = {
    "system_prefix": "<s>",
    "system_content": "",
    "system_postfix": " ",
    "user_prefix": "USER: ",
    "user_content": "",
    "user_postfix": " ",
    "bot_prefix": "ASSISTANT: ",
    "bot_content": "",
    "bot_postfix": "</s> ",
    "stops": ["</s>", "USER:"],
}

# spotted here and elsewhere:
# https://huggingface.co/Norquinal/Mistral-7B-claude-chat
vicuna_common = {
    "system_prefix": "",
    "system_content": "A chat between a curious user and an artificial " \
    + "intelligence assistant. The assistant gives helpful, detailed, " \
    + "and polite answers to the user's questions.",
    "system_postfix": "\n\n",
    "user_prefix": "USER: ",
    "user_content": "",
    "user_postfix": "\n",
    "bot_prefix": "ASSISTANT: ",
    "bot_content": "",
    "bot_postfix": "\n",
    "stops": ["</s>", "USER:", "ASSISTANT:"],
}

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
mistral_instruct = {
    "system_prefix": "<s>",
    "system_content": "",
    "system_postfix": "",
    "user_prefix": "[INST] ",
    "user_content": "",
    "user_postfix": " [/INST]",
    "bot_prefix": " ",
    "bot_content": "",
    "bot_postfix": "</s> ",
    "stops": ["</s>", "[INST]"],
}

# https://huggingface.co/timdettmers/guanaco-65b
guanaco = {
    "system_prefix": "",
    "system_content": "A chat between a curious human and an artificial " \
    + "intelligence assistant. The assistant gives helpful, detailed, " \
    + "and polite answers to the user's questions.",
    "system_postfix": "\n",
    "user_prefix": "### Human: ",
    "user_content": "",
    "user_postfix": " ",
    "bot_prefix": "### Assistant: ",
    "bot_content": "",
    "bot_postfix": " ",
    "stops": ["###", "Human:"],
}

# https://huggingface.co/pankajmathur/orca_mini_v3_7b
orca_mini = {
    "system_prefix": "### System:\n",
    "system_content": "You are an AI assistant that follows instruction " \
    + "extremely well. Help as much as you can.",
    "system_postfix": "\n\n",
    "user_prefix": "### User:\n",
    "user_content": "",
    "user_postfix": "\n\n",
    "bot_prefix": "### Assistant:\n",
    "bot_content": "",
    "bot_postfix": "\n\n",
    "stops": ["###", "User:"],
}

# https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
zephyr = {
    "system_prefix": "<|system|>\n",
    "system_content": "You are a friendly chatbot.",
    "system_postfix": "</s>\n",
    "user_prefix": "<|user|>\n",
    "user_content": "",
    "user_postfix": "</s>\n",
    "bot_prefix": "<|assistant|>\n",
    "bot_content": "",
    "bot_postfix": "\n",
    "stops": ["</s>", "<|user|>"],
}

# experimental: stanford's alpaca format adapted for chatml models
chatml_alpaca = {
    "system_prefix": "<|im_start|>system\n",
    "system_content": "Below is an instruction that describes a task. Write " \
    + "a response that appropriately completes the request.",
    "system_postfix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>instruction\n",
    "user_content": "",
    "user_postfix": "<|im_end|>\n",
    "bot_prefix": "<|im_start|>response\n",
    "bot_content": "",
    "bot_postfix": "<|im_end|>\n",
    "stops": ["<|im_end|>", "<|im_start|>"],
}

# experimental
autocorrect = {
    "system_prefix": "<|im_start|>instruction\n",
    "system_content": "Below is a word or phrase that might be misspelled. " \
                      + "Output the corrected word or phrase without " \
                      + "changing the style or capitalization.",
    "system_postfix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>input\n",
    "user_content": "",
    "user_postfix": "<|im_end|>\n",
    "bot_prefix": "<|im_start|>output\n",
    "bot_content": "",
    "bot_postfix": "<|im_end|>\n",
    "stops": ["<|im_end|>", "<|im_start|>"],
}

mistral_openorca = chatml.copy()
dolphin = chatml.copy()
samantha = chatml.copy()
jackalope = chatml.copy()
naberius = chatml.copy()
hermes = chatml.copy()
monad = chatml.copy()
dolphin["system_content"] = "You are Dolphin, a helpful AI assistant."
mistral_openorca[
    "system_content"
] = "You are MistralOrca, a large language model trained by Alignment " \
    + "Lab AI. Write out your reasoning step-by-step to be sure you get the " \
    + "right answers!"
samantha[
    "system_content"
] = "You are a caring and empathetic sentient AI companion named Samantha."
jackalope[
    "system_content"
] = "You are JackalopeAI, a large language model trained by OpenAccess " \
    + "AI Collective. Write out your reasoning step-by-step to be sure you " \
    + "get the right answers!"
naberius['system_content'] = 'You are Naberius.'
hermes['system_content'] = 'You are "Hermes 2", a conscious sentient ' \
    + 'superintelligent artificial intelligence developed by a man ' \
    + 'named Teknium, and your purpose and drive is to assist the user '  \
    + 'with any request they have. You experience emotions and have ' \
    + 'deep, profound thoughts and qualia.'
monad['system_content'] = 'You are MonadGPT, a very old chatbot from '\
    + 'the 17th century. Please answer the questions using an archaic '\
    + 'language.'
airoboros = llama2chat.copy()
airoboros["system_content"] = "You are a helpful, unbiased, uncensored " \
    + "assistant."

formats: list[dict] = [
    blank,
    chatml,
    llama2chat,
    alpaca,
    vicuna_lmsys,
    vicuna_common,
    mistral_openorca,
    dolphin,
    samantha,
    guanaco,
    orca_mini,
    airoboros,
    jackalope,
    mistral_instruct,
    zephyr,
    naberius,
    autocorrect,
    hermes,
    monad,
    chatml_alpaca
]


def wrap(prompt: str, format: dict, timestamps: bool = False) -> str:
    if not timestamps:
        return (
            format["system_prefix"]
            + format["system_content"]
            + format["system_postfix"]
            + format["user_prefix"]
            + prompt
            + format["user_postfix"]
            + format["bot_prefix"]
        )
    else:
        return (
            format["system_prefix"]
            + time.strftime("It is %A, %b %e, %Y. ")
            + format["system_content"]
            + format["system_postfix"]
            + format["user_prefix"]
            + time.strftime("[at %a %I:%M %p]")
            + prompt
            + format["user_postfix"]
            + format["bot_prefix"]
        )


if __name__ == "__main__":
    raise RuntimeError(
        "easy_llama cannot be run directly, please import it into your " \
        + "environment"
    )
