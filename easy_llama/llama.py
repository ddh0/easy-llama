# llama.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import ctypes
import sys
import os
import libllama as lib
import struct

from typing    import NoReturn, Optional, Iterable
from io        import BufferedReader
from enum      import IntEnum
from constants import Colors

RESET  = Colors.RESET
GREEN  = Colors.GREEN
BLUE   = Colors.BLUE
GREY   = Colors.GREY
YELLOW = Colors.YELLOW
RED    = Colors.RED

NULL = None
NULLPTR = ctypes.c_void_p(None)

_MAX_SINGLE_TOKEN_TEXT_LENGTH = 256

_SUPPORTED_KV_TYPES = [
    lib.GGMLType.GGML_TYPE_F32,   # lib only supports static types, not
    lib.GGMLType.GGML_TYPE_F16,   # k-types
    lib.GGMLType.GGML_TYPE_Q8_1,
    lib.GGMLType.GGML_TYPE_Q8_0,  # BF16 is also sometimes supported, but not
    lib.GGMLType.GGML_TYPE_Q5_1,  # always, and offers no benefit compared
    lib.GGMLType.GGML_TYPE_Q5_0,  # to F16, so is not included here
    lib.GGMLType.GGML_TYPE_Q4_1,
    lib.GGMLType.GGML_TYPE_Q4_0
]

_DEFAULT_KV_TYPE = lib.GGMLType.GGML_TYPE_F16

_cpu_count = os.cpu_count()

class LlamaNullException(Exception):
    """Raised when a libllama function returns NULL or NULLPTR"""

def null_ptr_check(
    ptr: lib.ptr, ptr_name: str, loc_hint: str
) -> None | NoReturn:
    """
    Ensure that the given object `ptr` is not NULL or NULLPTR

    Raise LlamaNullException on failure

    - ptr:
        The object to check
    - ptr_name:
        The name of the object (for error messages)
    - loc_hint:
        Code location hint used in easy-llama
    """
    if ptr is NULL:
        raise LlamaNullException(
            f"{loc_hint}: {ptr_name} is NULL"
        )
    if ptr is NULLPTR:
        raise LlamaNullException(
            f"{loc_hint}: {ptr_name} is NULLPTR"
        )

# TODO: should remove these print functions before release?

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

def _print_error(text: str) -> None:
    print(
        f"{RESET}easy_llama: {RED}ERROR{RESET}:",
        text, file=sys.stderr, flush=True
    )

def _init_backend_if_needed() -> None:

    # if already initialized, no need to do anything
    if lib._BACKEND_INIT is True:
        return
    
    # most cases
    if sys.byteorder == 'little':
        _print_verbose(
            "host is little-endian"
        )
    # rare
    elif sys.byteorder == 'big':
        _print_warning(
            "host is big-endian, please ensure your GGUF file is also "
            "big-endian"
        )
    # extremely rare
    else:
        _print_warning(
            f"unexpected value for sys.byteorder: {sys.byteorder!r}; "
            "expected 'little' for little-endian host or 'big' for "
            "big-endian host"
        )
    
    # actually load the backend now
    lib.llama_backend_init() # this sets libllama._BACKEND_INIT to True

# NOTE: the optimal n_threads value (for text generation) is equal
#       to the number of physical cores (for homogenous CPUs) or
#       to the number of performance cores (for heterogenous CPUs)
#
#       the optimal n_threads_batch value (for prompt eval) is equal
#       to the total number of logical cores, regardless of
#       their type

def _get_optimal_n_threads() -> int:
    global _cpu_count
    return max(_cpu_count//2, 1)

def _get_optimal_n_threads_batch() -> int:
    global _cpu_count
    return _cpu_count

def _get_random_seed() -> int:
    int.from_bytes(os.urandom(4), sys.byteorder)

def _calculate_rope_freq_base(
        n_ctx_train: int,
        n_ctx_load: int,
        rope_freq_base_train: Optional[float]
    ) -> float:
    """
    Returns the rope_freq_base value at which a model should be loaded
    """

    if n_ctx_load <= n_ctx_train:
        if rope_freq_base_train is None:
            return 0.0
        else:
            return rope_freq_base_train
    
    if rope_freq_base_train in [None, 0.0]:
        _print_error(
            f'n_ctx value {n_ctx_load} > n_ctx_train value {n_ctx_train}, and '
            f'automatic rope_freq_base adjustment is not supported for this '
            f'model; model loading might fail, or the model might not work '
            f'correctly'
        )
        return 0.0
    
    # standard formula -- proportional increase
    return (n_ctx_load/n_ctx_train)*rope_freq_base_train
    # experimental formula -- slightly above proportional increase
    #return ((n_ctx_load/n_ctx_train)**(2**(1/4)))*rope_freq_base_train

def _round_n_ctx(n_ctx: int, n_ctx_train: int) -> int:

    if n_ctx % 512 == 0:
        return n_ctx
    else:
        rounded = (n_ctx + 511) // 512 * 512
        if rounded > n_ctx_train: # do not round beyond n_ctx_train
            return n_ctx_train
        else:
            return rounded

# these values are from llama.cpp/gguf-py/gguf/constants.py
class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

class QuickGGUFReader:
    # ref: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

    # the GGUF format versions that this class supports
    SUPPORTED_GGUF_VERSIONS = [2, 3]
    
    # arguments for struct.unpack() based on gguf value type
    value_packing: dict = {
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

    # length in bytes for each gguf value type
    value_lengths: dict = {
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
    def get_single(
            value_type: GGUFValueType,
            file: BufferedReader
        ) -> str | int | float | bool:
        """Read a single value from an open file"""
        if value_type == GGUFValueType.STRING:
            string_length = QuickGGUFReader.unpack(GGUFValueType.UINT64, file=file)
            value = file.read(string_length)
            # officially, strings that cannot be decoded into utf-8 are invalid
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                _print_warning( # TODO: ?
                    f'UnicodeDecodeError was raised while reading a string '
                    f'from the GGUF metadata. the GGUF format specifies that '
                    f'all strings in file metadata should be valid UTF-8. the '
                    f'affected string will be left blank.'
                )
                value = ''
        else:
            value = QuickGGUFReader.unpack(value_type, file=file)
        return value
    
    @staticmethod
    def load_metadata(
            path_model: os.PathLike[str] | str
        ) -> dict[str, str | int | float | bool | list]:
        """
        Given a path to a GGUF file, peek at its header for metadata

        Return a dictionary where all keys are strings, and values can be
        strings, ints, floats, bools, or lists
        """

        metadata: dict[str, str | int | float | bool | list] = {}
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
                        QuickGGUFReader.get_single(
                            array_value_type,
                            file
                        ) for _ in range(array_length)
                    ]
                    metadata[key.decode()] = array
                else:
                    value = QuickGGUFReader.get_single(
                        value_type,
                        file
                    )
                    metadata[key.decode()] = value

        return metadata

#
# Simple python wrappers
#

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
        null_ptr_check(self.params, "self.params", "_LlamaModel.__init__")
        self.params.devices = (
            ctypes.c_void_p * (len(devices) + 1)
        )(*devices, None) if devices is not None else NULL
        self.params.n_gpu_layers = (
            n_gpu_layers
        ) if n_gpu_layers >= 0 else lib.MAX_OFFLOAD_LAYERS
        self.params.split_mode = (
            split_mode
        ) if split_mode is not None else (
            lib.LlamaSplitMode.LLAMA_SPLIT_MODE_NONE
        )
        self.params.main_gpu = main_gpu if main_gpu is not None else 0
        self.params.tensor_split = (
            ctypes.c_float * len(tensor_split)
        )(*tensor_split) if tensor_split is not None else NULL
        self.params.rpc_servers = (
            rpc_servers.encode('utf-8')
        ) if rpc_servers is not None else NULL
        self.params.progress_callback = (
            progress_callback
        ) if progress_callback is not None else lib.dummy_progress_callback()
        self.params.progress_callback_user_data = (
            progress_callback_user_data
        ) if progress_callback_user_data is not None else NULL
        self.params.kv_overrides = (
            lib.llama_model_kv_override * len(kv_overrides)
        )(*kv_overrides) if kv_overrides is not None else NULL
        self.params.vocab_only = vocab_only
        self.params.use_mmap = use_mmap
        self.params.use_mlock = use_mlock
        self.params.check_tensors = check_tensors

        # load model
        self.model = lib.llama_load_model_from_file(path_model, self.params)
        null_ptr_check(self.model, "self.model", "_LlamaModel.__init__")
    
    def __del__(self):
        self.free()

    def free(self):
        if self.model is not None:
            lib.llama_free_model(self.model)
    
    def n_ctx_train(self) -> int:
        """The native context length of this model"""
        return lib.llama_n_ctx_train(self.model)
    
    # NOTE: easy-llama impl
    def tokenize(
        self,
        text: str,
        add_special: bool,
        parse_special: bool
    ) -> list[int]:
        """
        Convert the provided text into tokens

        - add_special:
            Allow to add BOS and EOS tokens if model is configured to do so.
        - parse_special:
            Allow tokenizing special and/or control tokens which otherwise are
            not exposed and treated as plaintext. Does not insert a leading
            space.
        """
        n_ctx_train = self.n_ctx_train()
        tokens_buf = (ctypes.c_int32 * n_ctx_train)()
        text_len = len(text) + (2 * add_special)
        null_ptr_check(self.model, 'self.model', '_LlamaModel.tokenize')
        n_prompt = -lib.llama_tokenize(
            model=self.model,
            text=text,
            text_len=text_len,
            tokens=tokens_buf,
            n_tokens_max=0,
            add_special=add_special,
            parse_special=parse_special
        )
        print(f"{n_prompt=}")
        #tokens_buf_size = n_prompt # max number of tokens token_buf can hold
        #tokens_buf = (ctypes.c_int32 * tokens_buf_size)()
        del tokens_buf
        tokens_buf = (ctypes.c_int32 * n_ctx_train)()
        n_tokens = lib.llama_tokenize(
            model=self.model,
            text=text,
            text_len=text_len,
            tokens=tokens_buf,
            n_tokens_max=n_ctx_train,
            add_special=add_special,
            parse_special=parse_special
        )
        return list(tokens_buf[:n_tokens])

    # NOTE: llama-cpp-python impl
    # def tokenize(self, text: bytes, add_bos: bool, special: bool):
    #     n_ctx = self.n_ctx_train()
    #     tokens = (lib.llama_token * n_ctx)()
    #     n_tokens = lib.llama_tokenize(
    #         self.model, text, len(text), tokens, n_ctx, add_bos, special
    #     )
    #     if n_tokens < 0:
    #         n_tokens = abs(n_tokens)
    #         tokens = (lib.llama_token * n_tokens)()
    #         n_tokens = lib.llama_tokenize(
    #             self.model, text, len(text), tokens, n_tokens, add_bos, special
    #         )
    #         if n_tokens < 0:
    #             raise RuntimeError(
    #                 f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
    #             )
    #     return list(tokens[:n_tokens])

    def token_to_piece(self, token: int, special: bool) -> bytes:
        """
        Convert a single token ID into utf-8 bytes

        - special:
            If True, special tokens are rendered in the output
        """
        str_buf = ctypes.create_string_buffer(_MAX_SINGLE_TOKEN_TEXT_LENGTH)
        null_ptr_check(self.model, 'self.model', '_LlamaModel.token_to_piece')
        n_bytes = lib.llama_token_to_piece(
            model=self.model,
            token=token,
            buf=str_buf,
            length=_MAX_SINGLE_TOKEN_TEXT_LENGTH,
            lstrip=0, # skip up to 'lstrip' leading spaces
            special=special
        )
        if n_bytes > _MAX_SINGLE_TOKEN_TEXT_LENGTH:
            raise ValueError(
                f"_LlamaModel.token_to_piece: the token with ID {token} "
                f"requires a buffer of size {n_bytes}, but the maximum "
                f"buffer size is {_MAX_SINGLE_TOKEN_TEXT_LENGTH}"
            )
        # NOTE: do not just do str_buf.value.decode() because the token could
        #       possibly be a part of a utf-8 bytestring, but not a valid utf-8
        #       string itself. let the caller handle this
        return str_buf.raw[:n_bytes]

    def detokenize(
        self,
        tokens: Iterable[int],
        special: bool
    ) -> str:
        """
        Convert the provided tokens into a string

        - special:
            If True, special tokens are rendered in the output
        """
        null_ptr_check(self.model, 'self.model', '_LlamaModel.detokenize')
        text_bytes = b""
        for token in tokens:
            text_bytes += self.token_to_piece(token=token, special=special)
        
        try:
            text = text_bytes.decode(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            _print_warning(
                f'UnicodeDecodeError raised during detokenize - ignoring'
            )
            text = text_bytes.decode(encoding="utf-8", errors="ignore")
        
        return text


class _LlamaCtx:

    def __init__(
        self,
        model: _LlamaModel,
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
        null_ptr_check(self.params, "self.params", "_LlamaCtx.__init__")
        self.params.n_ctx = n_ctx
        self.params.n_batch = n_batch
        self.params.n_ubatch = n_ubatch
        if n_seq_max != 1:
            _print_warning(
                f'n_seq_max value {n_seq_max} != 1; this is not recommended'
            )
        self.params.n_seq_max = n_seq_max
        self.params.n_threads = n_threads
        self.params.n_threads_batch = n_threads_batch
        self.params.rope_scaling_type = (
            rope_scaling_type
        ) if rope_scaling_type is not None else (
            lib.LlamaRopeScalingType.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
        )
        self.params.pooling_type = (
            pooling_type
        ) if pooling_type is not None else (
            lib.LlamaPoolingType.LLAMA_POOLING_TYPE_UNSPECIFIED
        )
        self.params.attention_type = (
            attention_type
        ) if attention_type is not None else (
            lib.LlamaAttentionType.LLAMA_ATTENTION_TYPE_UNSPECIFIED
        )
        self.params.rope_freq_base = rope_freq_base
        self.params.rope_freq_scale = rope_freq_scale
        self.params.yarn_ext_factor = yarn_ext_factor
        self.params.yarn_attn_factor = yarn_attn_factor
        self.params.yarn_beta_fast = yarn_beta_fast
        self.params.yarn_beta_slow = yarn_beta_slow
        self.params.yarn_orig_ctx = yarn_orig_ctx
        self.params.defrag_thold = defrag_thold
        self.params.cb_eval = (
            cb_eval
        ) if cb_eval is not None else lib.dummy_eval_callback()
        self.params.cb_eval_user_data = cb_eval_user_data
        self.params.type_k = type_k if type_k is not None else _DEFAULT_KV_TYPE
        self.params.type_v = type_v if type_v is not None else _DEFAULT_KV_TYPE
        _k, _v = self.params.type_k, self.params.type_v
        if _k != _v:
            _print_warning(
                f'type_k value {_k} != type_v value {_v}; this is rarely '
                f'supported, program may fail'
            )
        if _k not in _SUPPORTED_KV_TYPES:
            _print_warning(
                f'type_k value {_k} is unsupported; program may fail'
            )
        if _v not in _SUPPORTED_KV_TYPES:
            _print_warning(
                f'type_v value {_v} is unsupported; program may fail'
            )
        if flash_attn and _v not in [
            lib.GGMLType.GGML_TYPE_F32, lib.GGMLType.GGML_TYPE_F16
        ]:
            _print_warning(
                f'V cache quantization requires flash_attn, program may fail'
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
            self.model.model, self.params
        )
        null_ptr_check(self.ctx, "self.ctx", "_LlamaCtx.__init__")
    
    def __del__(self):
        self.free()

    def free(self):
        if self.ctx is not None:
            lib.llama_free(self.ctx)
    
    def get_model(self) -> _LlamaModel:
        """The `_LlamaModel` that this context is attached to"""
        # NOTE: _LlamaModel is accessible from _LlamaCtx, but not vice-versa
        null_ptr_check(self.model, "self.model", "_LlamaCtx.get_model")
        return self.model
    
    def kv_cache_clear(self):
        """Clear the KV cache - both cell info is erased and KV data is zeroed"""
        null_ptr_check(self.ctx, 'self.ctx', '_LlamaCtx.kv_cache_clear')
        lib.llama_kv_cache_clear(self.ctx)
    
    def n_ctx(self) -> int:
        """The currently loaded context length"""
        null_ptr_check(self.ctx, 'self.ctx', '_LlamaCtx.n_ctx')
        return lib.llama_n_ctx(self.ctx)
    
    def rope_freq_base(self) -> float:
        """The rope_freq_base value this context was loaded with"""
        null_ptr_check(self.params, 'self.params', '_LlamaCtx.rope_freq_base')
        return self.params.rope_freq_base


class _LlamaBatch:

    def __init__(
        self,
        n_tokens: int,
        embd: int,
        n_seq_max: int
    ):
        _init_backend_if_needed()
        self.batch = lib.llama_batch_init(n_tokens, embd, n_seq_max)
        null_ptr_check(self.batch, "self.batch", "_LlamaBatch.__init__")
    
    def __del__(self):
        self.free()

    def free(self):
        if self.batch is not None:
            lib.llama_batch_free(self.batch)

    def get_one(self, tokens: list[int]):
        # TODO: find out why this is commented with "avoid using" in llama.h
        """
        AVOID USING

        Return batch for single sequence of tokens
        """
        tokens_array = (ctypes.c_int * len(tokens))(*tokens)
        batch = lib.llama_batch_get_one(tokens_array, len(tokens))
        null_ptr_check(batch, "batch", "_LlamaBatch.get_one")
        return batch

#
# Llama
#

class Llama:
    """
    Simplified interface for general-purpose Llama model usage
    """
    
    def __init__(
        self,
        path_model: str,
        n_gpu_layers: int = 0,
        use_mmap: bool = True,
        use_mlock: bool = False,
        n_ctx: int = 512,
        rope_freq_base: float = 0.0, # use 0.0 for auto, otherwise unmodified
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        logits_all: bool = False, # required for e.g. candidates, but hurts performance
        offload_kqv: bool = False,
        flash_attn: bool = False
    ):
        if not os.path.exists(path_model):
            raise FileNotFoundError(
                f"Llama: the given path_model {path_model!r} does not exist"
            )
        if os.path.isdir(path_model):
            raise IsADirectoryError(
                f"Llama: the given path_model {path_model!r} is a directory, "
                f"not a GGUF file"
            )
        if not path_model.lower().endswith('.gguf'):
            raise ValueError(
                f"Llama: the given path_model {path_model!r} does not end in "
                f"'.gguf'. easy-llama refuses to load from files that do not "
                f"have the correct file extension."
            )
        
        # peek at metadata from GGUF file header before loading model

        self.metadata = QuickGGUFReader.load_metadata(path_model)

        self._model = _LlamaModel(
            path_model=path_model,
            n_gpu_layers=n_gpu_layers,
            use_mmap=use_mmap,
            use_mlock=use_mlock
        )

        n_ctx_train = self._model.n_ctx_train()

        # use n_ctx unless it's 0 or negative, in that case use n_ctx_train

        if n_ctx <= 0:
            _print_info(
                f'n_ctx value {n_ctx}; using n_ctx_train value '
                f'{n_ctx_train}'
            )
            _n_ctx = n_ctx_train
        else:
            _n_ctx = n_ctx

        # use rope_freq_base unless it == 0.0, in that case use the native
        # rope_freq_base found in the GGUF metadata

        if rope_freq_base == 0.0:
            rope_freq_base_train = None
            for key in self.metadata.keys():
                if key.endswith('.rope.freq_base'):
                    rope_freq_base_train = float(self.metadata[key])
            
            # NOTE: if n_ctx is > n_ctx_train, then rope_freq_base must also
            #       be increased by at least a proportional amount to guarantee
            #       a usable kv cache throughout the entire context
            #
            #       the function _calculate_rope_freq_base handles this

            _rope_freq_base = _calculate_rope_freq_base(
                n_ctx_train=n_ctx_train,
                n_ctx_load=_n_ctx,
                rope_freq_base_train=rope_freq_base_train # can be None
            )
        else:
            _rope_freq_base = rope_freq_base
        
        self._ctx = _LlamaCtx(
            model=self._model,
            n_ctx=_n_ctx,
            n_threads=_get_optimal_n_threads(),
            n_threads_batch=_get_optimal_n_threads_batch(),
            rope_freq_base=_rope_freq_base,
            type_k=type_k,
            type_v=type_v,
            logits_all=logits_all,
            offload_kqv=offload_kqv,
            flash_attn=flash_attn
        )

        actual_n_ctx = self._ctx.n_ctx()
        requested_n_ctx = _n_ctx

        if actual_n_ctx != requested_n_ctx:
            _print_warning(
                f"requested n_ctx value differs from actual n_ctx value; "
                f"requested {requested_n_ctx}, actual {actual_n_ctx}"
            )
        if actual_n_ctx < 512:
            _print_warning(
                f"n_ctx value {actual_n_ctx} is less than 512, which can "
                f"sometimes cause problems with llama.cpp - consider "
                f"increasing it to at least 512"
            )
        if actual_n_ctx % 512 != 0:
            _print_warning(
                f"n_ctx value {actual_n_ctx} is not divisible by 512, which "
                f"can sometimes cause problems with llama.cpp. consider "
                f"changing it to "
                f"{_round_n_ctx(actual_n_ctx, n_ctx_train)}."
            )
        
        if actual_n_ctx > n_ctx_train:
            if _rope_freq_base == rope_freq_base:
                # model not guaranteed to work
                _print_warning(
                    f"n_ctx value {actual_n_ctx} exceeds n_ctx_train value "
                    f"{n_ctx_train}; using native rope_freq_base value "
                    f"{rope_freq_base}; model may not function correctly"
                )
            else:
                # model guaranteed to work, potentially degraded output quality
                _print_warning(
                    f"n_ctx value {actual_n_ctx} exceeds n_ctx_train value "
                    f"{n_ctx_train}; using adjusted rope_freq_base value "
                    f"{rope_freq_base}, native value is "
                    f"{rope_freq_base_train}; model will function with "
                    f"potentially degraded output quality"
                )
    
    def free(self):
        """Free the context and model"""
        self._ctx.free()
        self._model.free()
    
    def decode(self, batch: _LlamaBatch) -> int:
        """
        Decode a batch of tokens
        
        Returns:
        - 0:
            success
        - 1:
            could not find a KV slot for the batch (try reducing the size of
            the batch or increase the context)
        - < 0:
            error. the KV cache state is restored to the state before this 
            call
        """
        null_ptr_check(_LlamaBatch, 'batch', 'Llama.decode')
        return lib.llama_decode(self._ctx, batch.batch)

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

    TestLlama = Llama(
        path_model='./model.gguf',
        n_gpu_layers=-1,
        use_mmap=True,
        use_mlock=False,
        n_ctx=8192,
        logits_all=False,
        offload_kqv=True,
        flash_attn=True
    )
    chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'
    prompt = """Gentlemen, owing to lack of time and adverse circumstances, most people leave this world without thinking too much about it. Those who try get a headache and move on to something else. I belong to the second group. As my career progressed, the amount of space dedicated to me in Who's Who grew and grew, but neither the last issue nor any future ones will explain why I abandoned journalism. This will be the subject of my story, which I wouldn't tell you under other circumstances anyway."""
    prompt = chktxt
    print("-" * 80)
    test_print(f'prompt:\n\n{prompt!r}')

    tokens = TestLlama._model.tokenize(prompt, add_bos=False, special=False)
    print()
    test_print(f'tokenized prompt:\n\n{tokens!r}')

    detok = TestLlama._model.detokenize(tokens, special=True)
    print()
    test_print(f'detokenized prompt:\n\n{detok!r}')

    print(
        f"{'-' * 80}\n"
        f"num prompt characters - {len(prompt)}\n"
        f"num prompt tokens ----- {len(tokens)}\n"
        f"num detok characters -- {len(detok)}"
    )

    print("-" * 80)
