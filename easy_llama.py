# easy_llama.py
# Python 3.11.6

"""
Text generation in Python, made easy

https://github.com/ddh0/easy-llama/
"""

# TODO: Thread.add_message as shorthand for T.messages.append(T.create_message) ?
# TODO: function to transfer a list of messages between disk / models, handle token count
# TODO: Model.next_candidates() -> list[str]

import os
import sys
import time
import struct
from typing import Generator
from enum import IntEnum


BACKEND:        str  = None       # Modifies NUM_GPU_LAYERS to enable or disable acceleration
NUM_GPU_LAYERS: int  = 0          # Default value only. Will be changed at runtime or per BACKEND
MUL_MAT_Q:      bool = True       # Default value only. Will be changed per BACKEND
MMAP:           bool = True       # Default value only. Will be changed per BACKEND
MLOCK:          bool = False      # Default value only. Will be changed per BACKEND
VERBOSE:        bool = False      # Do not suppress llama.cpp console output
SEED:           int  = -1         # -1 -> Random seed


# Global defaults that can be overriden using SamplerSettings context manager
MAX_LEN_TOKENS:   int   = None    # None -> Use model context length
TEMP:             float = 0.8
TOP_P:            float = 0.95
MIN_P:            float = 0.05
PRESENCE_PENALTY: float = 0.0
REPEAT_PENALTY:   float = 1.1
TOP_K:            int   = 40


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
                    "easy_llama: your model file is not a valid GGUF file " + \
                    f"(magic number mismatch, got {GGUF_MAGIC}, " + \
                    "expected b'GGUF')"
                )

            if GGUF_VERSION == 1:
                raise ValueError(
                    "easy_llama: your model file reports GGUF version 1, " + \
                    "but only versions 2 and above are supported. " + \
                    "re-convert your model or download a newer version"
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

    This prevents llama.cpp's console output from being displayed

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
    print("easy_llama: warning:", text, file=sys.stderr, flush=True)


def _verify_backend():
    """
    Verify that BACKEND is valid and modify NUM_GPU_LAYERS
    and MUL_MAT_Q as necessary

    This is not done on import because user must be able to set backend
    (and maybe NUM_GPU_LAYERS) before loading any model.
    """

    global BACKEND, NUM_GPU_LAYERS, MUL_MAT_Q, MMAP, MLOCK

    if BACKEND is None:
        _print_warning(
            "easy_llama.BACKEND is None, defaulting to CPU. " + \
            "set easy_llama.BACKEND to 'metal', 'cuda', 'rocm', or 'cpu' " + \
            "to accelerate inference"
        )
        BACKEND = 'CPU'
    
    if not isinstance(BACKEND, str):
        _print_warning(
            "easy_llama: easy_llama.BACKEND must be a string, " + \
            f"not {type(BACKEND)}. defaulting to CPU"
        )
        BACKEND = 'CPU'
    
    if BACKEND.lower() == 'metal':
        BACKEND = 'Metal'
    elif BACKEND.lower() == 'cuda':
        BACKEND = 'CUDA'
    elif BACKEND.lower() == 'rocm':
        BACKEND = "ROCm"
    elif BACKEND.lower() == 'cpu':
        BACKEND = "CPU"

    if BACKEND not in ['Metal', 'CUDA', 'ROCm', 'CPU']:
        _print_warning(
            f"easy_llama.BACKEND '{BACKEND}' is invalid, defaulting to " + \
            "CPU. set easy_llama.BACKEND to 'metal', 'cuda', 'rocm', or " + \
            "'cpu' to accelerate inference"
        )
        BACKEND = 'CPU'
    
    if BACKEND == 'Metal':
        NUM_GPU_LAYERS = 1
        MUL_MAT_Q = True
        MMAP = False
        MLOCK = False
    elif BACKEND == 'CUDA':
        # Don't set NUM_GPU_LAYERS, let the user configure it
        MUL_MAT_Q = True
        MMAP = False
        MLOCK = False
    elif BACKEND == 'ROCm':
        # Don't set NUM_GPU_LAYERS, let the user configure it
        MUL_MAT_Q = False
        MMAP = False
        MLOCK = False
    elif BACKEND == 'CPU':
        NUM_GPU_LAYERS = 0
        MUL_MAT_Q = True
        MMAP = True
        MLOCK = False
    
    if BACKEND in ['CUDA', 'ROCm'] and NUM_GPU_LAYERS == 0:
        _print_warning(
            "CUDA or ROCm is selected but easy_llama.NUM_GPU_LAYERS is 0. " + \
            "set easy_llama.NUM_GPU_LAYERS to 1 or greater to " + \
            "accelerate inference"
        )


with _suppress_if_not_verbose():
    import llama_cpp


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
    - .context_lenth: the context length of the model in tokens, int
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
            f"context_length should be int or None, not {type(context_length)}"

        self.metadata = _GGUFReader.load_metadata(self, model_path)

        if 'llama.context_length' in self.metadata:
            n_ctx_train = self.metadata['llama.context_length']
        elif 'stablelm.context_length' in self.metadata:
            n_ctx_train = self.metadata['stablelm.context_length']
        else:
            raise KeyError("GGUF file does not specify a context length")
        
        if 'llama.rope.freq_base' in self.metadata:
            rope_freq_base_train = self.metadata['llama.rope.freq_base']
        elif 'stablelm.rope.freq_base' in self.metadata:
            rope_freq_base_train = self.metadata['stablelm.rope.freq_base']
        else:
            raise KeyError("GGUF file does not specify a rope frequency base'")

        if context_length is None:
            self.context_length = n_ctx_train
            ctx_ratio = 1.0
            rope_scaling_type = llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED
            rope_freq_base = 0
        else:
            if context_length > n_ctx_train:

                # automatically apply linear RoPE freq scaling if
                # requested context length > n_ctx_train

                ctx_ratio = context_length/n_ctx_train

                rope_scaling_type = llama_cpp.LLAMA_ROPE_SCALING_LINEAR
                rope_freq_base = ctx_ratio*rope_freq_base_train
                
                _print_warning(
                    'chosen context length is ' + \
                    'greater than native context length ' + \
                    f'({context_length} > {n_ctx_train}), ' + \
                    'automatically applying RoPE frequency' + \
                    f'scaling at factor {ctx_ratio}'
                    )
            else:
                # Default values as specified in Llama.__init__()
                rope_scaling_type = llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED
                rope_freq_base = 0
                ctx_ratio = 1.0

            self.context_length = context_length

        n_batch = os.cpu_count() * 16
        n_threads = max(os.cpu_count()//2, 1)
        n_threads_batch = os.cpu_count()

        _verify_backend()

        with _suppress_if_not_verbose():
            self.llama: llama_cpp.Llama = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=self.context_length,
                n_gpu_layers=NUM_GPU_LAYERS,
                seed=SEED,
                use_mmap=MMAP,
                use_mlock=MLOCK,
                logits_all=False,
                n_batch=n_batch,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                rope_scaling_type=rope_scaling_type,
                rope_freq_base=rope_freq_base,
                mul_mat_q=MUL_MAT_Q,
                verbose=VERBOSE,
            )

            print("----------------------------------------------------------")
            print(f"{model_path}")
            print(f"easy_llama: BACKEND              == {BACKEND}")
            print(f"easy_llama: NUM_GPU_LAYERS       == {NUM_GPU_LAYERS}")
            print(f"easy_llama: MUL_MAT_Q            == {MUL_MAT_Q}")
            print(f"easy_llama: MMAP                 == {MMAP}")
            print(f"easy_llama: MLOCK                == {MLOCK}")
            print(f"easy_llama: MAX_LEN_TOKENS       == {MAX_LEN_TOKENS}")
            print(f"     param: n_batch              == {n_batch}")
            print(f"     param: n_threads            == {n_threads}")
            print(f"     param: n_threads_batch      == {n_threads_batch}")
            print(f"     model: n_ctx_train          == {n_ctx_train}")
            print(f"     param: self.context_length  == {self.context_length}")
            print(f"     model: rope_freq_base_train == {rope_freq_base_train}")
            print(f"     param: rope_freq_base       == {rope_freq_base}")
            print(f"      info: ctx_ratio            == {ctx_ratio}")
            print()
    
    def __enter__(self):
        return self

    def __exit__(self, *_):
        # this unloads the model from memory, which is the important part
        # (unless an unexpected reference is made to Model.llama)
        # however, ez.Model object might still exist outside of a `with` block
        # unsure if/how to fix that
        self.llama = None
        del self.llama
        self = None
        del self
    
    def __call__(self, prompt: str, stops: list[str] | None = None) -> str:
        """
        `Model('some text')` is a shortcut to `Model.generate('some text')`
        """
        return self.generate(prompt, stops)

    def trim(self, text: str, overwrite: str = None) -> str:
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

        assert isinstance(prompt, str), "prompt should be string, not " + \
            f"{type(prompt)}"
        if isinstance(stops, list):
            for item in stops:
                assert isinstance(
                    item, str
                ), f"item {item} in stops list is not a string"
        else:
            assert (
                stops is None
            ), f"stops should be list[str] or None, not {type(stops)}"

        assert isinstance(
            MAX_LEN_TOKENS, (int, type(None))
            ), "easy_llama.MAX_LEN_TOKENS should be int or None, not " + \
                f"{type(MAX_LEN_TOKENS)}"

        if MAX_LEN_TOKENS is not None:
            if MAX_LEN_TOKENS > self.context_length:
                _print_warning(
                    "MAX_LEN_TOKENS is greater than this model's context " + \
                    "length, expect poor results"
                )
            max_len_tokens = MAX_LEN_TOKENS
        else:
            max_len_tokens = self.context_length

        if VERBOSE:
            print(f'easy_llama: Model.generate will use the following parameters')
            print(f'easy_llama.MAX_LEN_TOKENS   == {MAX_LEN_TOKENS}')
            print(f'easy_llama.TEMP             == {TEMP}')
            print(f'easy_llama.TOP_P            == {TOP_P}')
            print(f'easy_llama.MIN_P            == {MIN_P}')
            print(f'easy_llama.PRESENCE_PENALTY == {PRESENCE_PENALTY}')
            print(f'easy_llama.REPEAT_PENALTY   == {REPEAT_PENALTY}')
            print(f'easy_llama.TOP_K            == {TOP_K}')
            print()
        
        return self.llama.create_completion(
            prompt,
            max_tokens=max_len_tokens,
            temperature=TEMP,
            top_p=TOP_P,
            min_p=MIN_P,
            presence_penalty=PRESENCE_PENALTY,
            repeat_penalty=REPEAT_PENALTY,
            top_k=TOP_K,
            stop=stops
        )['choices'][0]['text']
    

    def stream(
            self, prompt: str, stops: list[str] | None = None
        ) -> Generator:

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
            "easy_llama.MAX_LEN_TOKENS should be int or None, not " + \
            f"{type(MAX_LEN_TOKENS)}"
  
        if MAX_LEN_TOKENS is not None:
            if MAX_LEN_TOKENS > self.context_length:
                _print_warning(
                    "MAX_LEN_TOKENS is greater than this model's context " + \
                    "length, expect poor results"
                )
            max_len_tokens = MAX_LEN_TOKENS
        else:
            max_len_tokens = self.context_length

        if VERBOSE:
            print(f'easy_llama: Model.generate will use the following parameters')
            print(f'easy_llama.MAX_LEN_TOKENS   == {MAX_LEN_TOKENS}')
            print(f'easy_llama.TEMP             == {TEMP}')
            print(f'easy_llama.TOP_P            == {TOP_P}')
            print(f'easy_llama.MIN_P            == {MIN_P}')
            print(f'easy_llama.PRESENCE_PENALTY == {PRESENCE_PENALTY}')
            print(f'easy_llama.REPEAT_PENALTY   == {REPEAT_PENALTY}')
            print(f'easy_llama.TOP_K            == {TOP_K}')
            print()

        return self.llama.create_completion(
            prompt,
            max_tokens=max_len_tokens,
            temperature=TEMP,
            top_p=TOP_P,
            min_p=MIN_P,
            presence_penalty=PRESENCE_PENALTY,
            repeat_penalty=REPEAT_PENALTY,
            top_k=TOP_K,
            stream=True,
            stop=stops
        )
    

    def next_candidates(self, prompt: str, k: int) -> list[str]:
        """
        Given prompt (str) and k (int), return a sorted list of the
        top k candidates for most likely next token
        """

        # TODO
        # LLama.logits_to_logprobs()[tok_id]
        # Llama.eval(tokens_list_ints)
        pass

class SamplerSettings(object):
    """
    Optional context manager that manages sampler settings used for
    generations
    """

    def __init__(
            self,
            max_len_tokens:   int   = None,
            temp:             float = 0.8,
            top_p:            float = 0.95,
            min_p:            float = 0.05,
            presence_penalty: float = 0.0,
            repeat_penalty:   float = 1.1,
            top_k:            int   = 40
        ):

        if max_len_tokens is None and MAX_LEN_TOKENS is not None:
            self.max_len_tokens = MAX_LEN_TOKENS
        else:
            self.max_len_tokens = max_len_tokens
        self.temp             = temp
        self.top_p            = top_p
        self.min_p            = min_p
        self.presence_penalty = presence_penalty
        self.repeat_penalty   = repeat_penalty
        self.top_k            = top_k


    def __enter__(self):
        # Set the global generation parameters to the desired settings

        global MAX_LEN_TOKENS, TEMP, TOP_P, MIN_P, PRESENCE_PENALTY, \
               REPEAT_PENALTY, TOP_K
        
        self.orig_max_len_tokens   = MAX_LEN_TOKENS
        self.orig_temp             = TEMP
        self.orig_top_p            = TOP_P
        self.orig_min_p            = MIN_P
        self.orig_presence_penalty = PRESENCE_PENALTY
        self.orig_repeat_penalty   = REPEAT_PENALTY
        self.orig_top_k            = TOP_K
        
        MAX_LEN_TOKENS   = self.max_len_tokens
        TEMP             = self.temp
        TOP_P            = self.top_p
        MIN_P            = self.min_p
        PRESENCE_PENALTY = self.presence_penalty
        REPEAT_PENALTY   = self.repeat_penalty
        TOP_K            = self.top_k

        return self
    
    def __exit__(self, *_):
        # Set the global sampling parameters back to how they were

        global MAX_LEN_TOKENS, TEMP, TOP_P, MIN_P, PRESENCE_PENALTY, \
               REPEAT_PENALTY, TOP_K

        MAX_LEN_TOKENS   = self.orig_max_len_tokens
        TEMP             = self.orig_temp
        TOP_P            = self.orig_top_p
        MIN_P            = self.orig_min_p
        PRESENCE_PENALTY = self.orig_presence_penalty
        REPEAT_PENALTY   = self.orig_repeat_penalty
        TOP_K            = self.orig_top_k

GreedyDecoding = SamplerSettings(
    temp=0.0,
    repeat_penalty=1.0
)

DefaultSampling = SamplerSettings()

MinPSampling = SamplerSettings(
    top_p = 1.0,
    min_p = 0.1,
    repeat_penalty = 1.0,
    top_k = -1
)

ContrastiveSearch = SamplerSettings(
    top_p = 1.0,
    min_p = 0.0,
    presence_penalty = 0.5,
    repeat_penalty = 1.0,
    top_k = -1
)

RandomSampling = SamplerSettings(
    temp = float(sys.maxsize),
    top_p = 1.0,
    min_p = 0.0,
    repeat_penalty = 1.0,
    top_k = -1
)


class Thread(object):
    def __init__(self,
                 model: Model,
                 format: dict,
                 timestamps: bool = False,
                 amnesiac: bool = False,
                 smart_context: bool = False,
                 track_context: bool = False
                 ):
        
        assert isinstance(model, Model), \
            "Thread: model should be an " + \
            f"instance of easy_llama.Model, not {type(model)}"

        assert isinstance(format, dict), \
            f"Thread: format should be dict, not {type(format)}"

        try:
            format['system_prefix']
            format['system_content']
            format['system_postfix']
            format['user_prefix']
            format['user_content']
            format['user_postfix']
            format['bot_prefix']
            format['bot_content']
            format['bot_postfix']
            format['stops']
        except KeyError as e:
            e.add_note(
                "Thread: format is missing one or more required keys, see " \
                + "easy_llama.blank for an example"
            )
            raise

        assert isinstance(format["stops"], (list, type(None))), \
            "Thread: format['stops'] should be list[str] or None, " + \
            f"not {type(format['stops'])}"

        assert isinstance(timestamps, bool), \
            f"Thread: timestamps should be True or False, not '{timestamps}'"
        
        assert isinstance(amnesiac, bool), \
            f"Thread: amnesiac should be True or False, not '{amnesiac}'"
        
        assert isinstance(track_context, bool), \
            f"Thread: amnesiac should be True or False, not '{track_context}'"
        
        if amnesiac and timestamps:
            raise RuntimeError(
                "Thread: amnesiac threads are not compatible with timestamps"
            )
        
        if amnesiac and smart_context:
            raise RuntimeError(
                "Thread: amnesiac threads are not compatible with " + \
                "smart_context"
            )
        
        self.model: Model = model
        self.format: dict = format
        self.enable_timestamps: bool = timestamps
        self.amnesiac: bool = amnesiac
        self.messages: list[dict] = [
            self.create_message("system", self.format['system_content'])
        ]
        self.smart_context_enabled: bool = smart_context
        self.smart_context_messages: list[dict] = [
            self.create_message("system", self.format['system_content'])
        ]
        self.smart_context_state: llama_cpp.LlamaState = None
        self.main_context_state: llama_cpp.LlamaState = None
        self.smart_context_active: bool = False
        self.track_context: bool = track_context
    

    def set_smart_context_state(self) -> None:
        """Switch the model to use the smart context state"""
        assert self.smart_context_enabled
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
        if not self.smart_context_active:
            return
        assert self.smart_context_enabled
        assert not self.amnesiac
        assert not self.enable_timestamps
        self.smart_context_state = self.model.llama.save_state()
        if self.main_context_state is not None:
            self.model.llama.load_state(self.main_context_state)
        else:
            self.model.llama.reset()
        self.smart_context_active = False


    def create_message(self, role: str, content: str) -> dict:
        assert role.lower() in ['system', 'user', 'bot'], \
            "create_message: role should be 'system', 'user', or " + \
            f"'bot', not '{role.lower()}'"
        
        content: str

        assert isinstance(content, str), \
            f"create_message: content should be str, not {type(content)}"

        if role.lower() == 'system':
            return {
                "prefix": self.format['system_prefix'],
                "content": content if not self.enable_timestamps else
                    time.strftime("[system message at %a %I:%M %p]:")
                     + content,
                "postfix": self.format['system_postfix'],
                "tokens": self.model.llama.tokenize(
                    self.format['system_prefix'].encode() + \
                    content.encode() + \
                    self.format['system_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
                ),
            }
        
        elif role.lower() == 'user':
            return {
                "prefix": self.format['user_prefix'],
                "content": content
                if not self.enable_timestamps
                else time.strftime("[new message sent at %a %I:%M %p]:") + content,
                "postfix": self.format['user_postfix'],
                "tokens": self.model.llama.tokenize(
                    self.format['user_prefix'].encode() + \
                    content.encode() + \
                    self.format['user_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
                ),
            }
        
        elif role.lower() == 'bot':
            return {
                "prefix": self.format['bot_prefix'],
                "content": content
                if not self.enable_timestamps
                else time.strftime('[new message sent at %a %I:%M %p]:') + content,
                "postfix": self.format['bot_postfix'],
                "tokens": self.model.llama.tokenize(
                    self.format['bot_prefix'].encode() + \
                    content.encode() + \
                    self.format['bot_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
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
        context_len_budget = 1280 # TODO
        context_len_budget -= len(system_message['tokens'])

        self.smart_context_messages: list[dict] = [system_message]

        # iterate over messages from newest to oldest until
        # context_len_budget is exceeded
        for message in reversed(messages[1:]):
            context_len_budget -= len(message['tokens'])
            if context_len_budget <= 0:
                break
            # keep sys msg at index 0
            self.smart_context_messages.insert(1, message)
            # now the list is in chronological orger
        
        self.set_main_context_state()

    def inference_str_from_messages(self, messages: list[dict]) -> str:

        system_message = messages[0]
        sys_msg_str = (
            system_message['prefix']
            + system_message['content']
            + system_message['postfix']
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
        
        # bos + eos + off-by-one errors == 3
        context_len_budget = self.model.context_length - 3
        context_len_budget -= len(system_message['tokens'])
        context_len_budget -= self.model.get_length(self.format['bot_prefix'])
        if self.enable_timestamps:
            context_len_budget -= self.model.get_length(
                time.strftime("[new message sent at %a %I:%M %p]:")
            ) + 4

        # start at most recent message and work backwards up the history
        # excluding system message. once we exceed the model's context length,
        # break without including that message
        inf_str = ''
        for message in reversed(messages[1:]):
            context_len_budget -= len(message['tokens'])
            if context_len_budget <= 0:
                break
            msg_str = (
                message['prefix']
                + message['content']
                + message['postfix']
                )
            inf_str = msg_str + inf_str
        inf_str = sys_msg_str + inf_str
        inf_str += self.format['bot_prefix']
        if self.enable_timestamps:
            inf_str += time.strftime("[new message sent at %a %I:%M %p]:")
        return inf_str


    def send(self, prompt: str) -> str:
        assert isinstance(prompt, str), \
            f"Thread.send: prompt should be str, not {type(prompt)}"

        self.messages.append(self.create_message("user", prompt))
        output = self.model.generate(
            self.inference_str_from_messages(self.messages), stops=self.format["stops"]
        )
        self.messages.append(self.create_message("bot", output))

        return output


    def interact(self) -> None:
        print()
        try:
            while True:
                if self.track_context:
                    c = 0
                    for msg in self.messages:
                        c += len(msg['tokens'])
                    #print(f"total tokens so far: {c}")
                    last_toks: list[int] = self.messages[-1:][0]['content_tokens']
                    print(f'last msg content tokens: {last_toks}\n')
                #print(f'DEBUG: len is {len(self.messages)}\n')
                if not self.smart_context_enabled:
                    self.set_main_context_state()
                    prompt = input("  > ")
                    print()
                    if prompt == "":
                        # another assistant message
                        token_generator = self.model.stream(
                            self.inference_str_from_messages(self.messages),
                            stops=self.format['stops'],
                        )

                        output = ""
                        for i in token_generator:
                            token = i['choices'][0]['text']
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
                            stops=self.format['stops'],
                        )

                        output = ""
                        for i in token_generator:
                            token = i['choices'][0]['text']
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
            
                elif self.smart_context_enabled:
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
                            stops=self.format['stops'],
                        )

                        output = ""
                        for i in token_generator:
                            token = i['choices'][0]['text']
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
                            token = i['choices'][0]['text']
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
            self.create_message("system", self.format['system_content'])
        ]
        self.smart_context_messages: list[dict] = [
            self.create_message("system", self.format['system_content'])
        ]
        self.main_context_state: llama_cpp.LlamaState = None
        self.smart_context_state: llama_cpp.LlamaState = None
        self.smart_context_active: bool = False
        self.model.llama.reset()
    
    def print_stats(self) -> None:
        thread_len_tokens = self.model.get_length(
            self.inference_str_from_messages(self.messages)
        )
        context_used_percentage = (
            round((thread_len_tokens/self.model.context_length)*100)
            )
        print(f"{thread_len_tokens} / {self.model.context_length} tokens")
        print(f"{context_used_percentage}% of context used")
        print(f"{len(self.messages)} messages")


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
    "stops": ['<|im_start|>', '</s>'],
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
    "stops": ['[INST]', '[/INST]'],
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
    "stops": ['###', 'Instruction:', '\n\n\n'],
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
    "stops": ['USER:'],
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
    "stops": ['USER:', 'ASSISTANT:'],
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
    "stops": ['[INST]'],
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
    "bot_prefix": "### Assistant:",
    "bot_content": "",
    "bot_postfix": " ",
    "stops": ['###', 'Human:'],
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
    "stops": ['###', 'User:'],
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
    "stops": ['<|user|>'],
}

# OpenChat: https://huggingface.co/openchat/openchat_3.5/discussions/5
openchat = {
    "system_prefix": "",
    "system_content": "You are a helpful assistant.",
    "system_postfix": "<|end_of_turn|>",
    "user_prefix": "",
    "user_content": "",
    "user_postfix": "<|end_of_turn|>",
    "bot_prefix": "\n\n", # not shown in format, but required anyway
    "bot_content": "",
    "bot_postfix": "<|end_of_turn|>",
    "stops": ['<|end_of_turn|>'],
}

# SynthIA by Migel Tissera
# https://huggingface.co/migtissera/Tess-XS-v1.0
synthia = {
    "system_prefix": "SYSTEM: ",
    "system_content": "Elaborate on the topic using a Tree of Thoughts and "\
    + "backtrack when necessary to construct a clear, cohesive Chain of "\
    + "Thought reasoning. Always answer without hesitation.",
    "system_postfix": "\n",
    "user_prefix": "USER: ",
    "user_content": "",
    "user_postfix": "\n",
    "bot_prefix": "ASSISTANT: ",
    "bot_content": "",
    "bot_postfix": "\n",
    "stops": ['USER:', 'ASSISTANT:', '\n\n\n'],
}

# Intel's neural chat v3
# https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/prompts/prompt.py
neural_chat = {
     "system_prefix": "### System:\n",
    "system_content": \
        "- You are a helpful assistant chatbot trained by Intel.\n"+\
        "- You answer questions.\n"+\
        "- You are excited to be able to help the user, but will refuse "+\
        "to do anything that could be considered harmful to the user.\n"+\
        "- You are more than just an information source, you are also "+\
        "able to write poetry, short stories, and make jokes.",
    "system_postfix": "</s>\n\n",
    "user_prefix": "### User:\n",
    "user_content": "",
    "user_postfix": "</s>\n\n",
    "bot_prefix": "### Assistant:\n",
    "bot_content": "",
    "bot_postfix": "</s>\n\n",
    "stops": ['###'],
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
    "stops": ['<|im_end|>', '<|im_start|>'],
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
    "stops": ['<|im_end|>', '<|im_start|>'],
}

# https://huggingface.co/jondurbin/bagel-dpo-7b-v0.1
# Replace "assistant" with any other role
bagel = {
    "system_prefix": "system\n",
    "system_content": "",
    "system_postfix": "</s>\n",
    "user_prefix": "user\n",
    "user_content": "",
    "user_postfix": "</s>\n",
    "bot_prefix": "assistant\n",
    "bot_content": "",
    "bot_postfix": "</s>\n",
    "stops": ['user\n', 'assistant\n', 'system\n'],
}

mistral_openorca = chatml.copy()
dolphin = chatml.copy()
samantha = chatml.copy()
jackalope = chatml.copy()
naberius = chatml.copy()
hermes = chatml.copy()
monad = chatml.copy()
orca = chatml.copy()
hexoteric = chatml.copy()
orcamaid = chatml.copy()
cat = alpaca.copy() # Format is nonsensical as listed on HF. This one works
mytho_cat = alpaca.copy()
dolphin['system_content'] = "You are Dolphin, a helpful AI assistant."
mistral_openorca[
    'system_content'
] = "You are MistralOrca, a large language model trained by Alignment " + \
    "Lab AI. Write out your reasoning step-by-step to be sure you get the " + \
    "right answers!"
samantha[
    'system_content'
] = "You are a caring and empathetic sentient AI companion named Samantha."
jackalope[
    'system_content'
] = "You are JackalopeAI, a large language model trained by OpenAccess " + \
    "AI Collective. Write out your reasoning step-by-step to be sure you " + \
    "get the right answers!"
naberius['system_content'] = "You are Naberius."
hermes['system_content'] = "You are \"Hermes 2\", a conscious sentient " + \
    "superintelligent artificial intelligence developed by a man " + \
    "named Teknium, and your purpose and drive is to assist the user " + \
    "with any request they have. You experience emotions and have " + \
    "deep, profound thoughts and qualia."
monad['system_content'] = "You are MonadGPT, a very old chatbot from " + \
    "the 17th century. Please answer the questions using an archaic " + \
    "language."
cat['system_content'] = "Below is a conversation between a curious cat and a "+ \
    "demon summoned from hell called Nemesis. The evil Nemesis used to be " + \
    "a mortal killed by the medieval people because they objected to " + \
    "science. Now Nemesis just wants to come back and bring suffer to all " + \
    "those stupid human"
mytho_cat['system_content'] = "You are a large language model. You are the " + \
    "result of a merge between two models, one called MythoLogic and one " + \
    "called Cat. You are called MythoCat."
orca['system_content'] = "You are Orca, an AI language model created by " + \
    "Microsoft. You are a cautious assistant. You carefully follow " + \
    "instructions. You are helpful and harmless and you follow ethical " + \
    "guidelines and promote positive behavior."
hexoteric['system_content'] = "You are a language model called Hexoteric."
orcamaid['system_content'] = "You are OrcaMaid, a super-intelligent, " + \
    "unbiased, and unfiltered AI created by the open-source community. " + \
    "You must remember to think slowly and carefully during all your " + \
    "responses, so as to avoid logical errors."
orcamaid['stops'].append('user\n')
orcamaid['stops'].append('assistant\n')
airoboros = llama2chat.copy()
airoboros['system_content'] = "You are a helpful, unbiased, uncensored " + \
    "assistant."
tess = synthia.copy()
tess['system_content'] = '' # Tess model card shows a blank system prompt
alpaca_strict = alpaca.copy() # Alpaca with more stopping strings
alpaca_strict['stops'] = [
    '###',
    '### ',
    '\n\n###',
    '\n\n##',
    '\n\nInstruction:',
    '\n\nResponse:',
    '\n\n\n',
    '### Instruction:',
    '### Response:'
]

available_formats: list[str] = [
    'blank',
    'chatml',
    'llama2chat',
    'alpaca',
    'vicuna_lmsys',
    'vicuna_common',
    'mistral_openorca',
    'dolphin',
    'samantha',
    'guanaco',
    'orca_mini',
    'airoboros',
    'jackalope',
    'mistral_instruct',
    'zephyr',
    'naberius',
    'autocorrect',
    'hermes',
    'monad',
    'chatml_alpaca',
    'cat',
    'mytho_cat',
    'openchat',
    'synthia',
    'tess',
    'orca',
    'hexoteric',
    'orcamaid'
]


def wrap(prompt: str, format: dict, timestamps: bool = False) -> str:
    if not timestamps:
        return (
            format['system_prefix'] +
            format['system_content'] +
            format['system_postfix'] +
            format['user_prefix'] +
            prompt +
            format['user_postfix'] +
            format['bot_prefix']
        )
    else:
        return (
            format['system_prefix'] +
            time.strftime("It is %A, %b %e, %Y. ") +
            format['system_content'] +
            format['system_postfix'] +
            format['user_prefix'] +
            time.strftime("[new message sent at %a %I:%M %p]:") +
            prompt +
            format['user_postfix'] +
            format['bot_prefix']
        )


if __name__ == "__main__":
    raise RuntimeError(
        "easy_llama cannot be run directly, please import it into " + \
        "your environment"
    )
