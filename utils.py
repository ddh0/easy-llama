# utils.py
# Python 3.11.6

"""
Submodule containing convenience functions, GGUFReader, and OutputSupressor
"""

import globals
import struct
import time
import sys
import os

from enum import IntEnum

class GGUFReader:
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
        if value_type == GGUFReader.GGUFValueType.STRING:
            value_length = struct.unpack("<Q", file.read(8))[0]
            value = file.read(value_length)
            value = value.decode("utf-8")
        else:
            type_str = GGUFReader._simple_value_packing.get(value_type)
            bytes_length = GGUFReader.value_type_info.get(value_type)
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

                value_type = GGUFReader.GGUFValueType(
                    struct.unpack("<I", file.read(4))[0]
                )
                if value_type == GGUFReader.GGUFValueType.ARRAY:
                    ltype = GGUFReader.GGUFValueType(
                        struct.unpack("<I", file.read(4))[0]
                    )
                    length = struct.unpack("<Q", file.read(8))[0]
                    for _ in range(length):
                        _ = GGUFReader.get_single(self, ltype, file)
                else:
                    value = GGUFReader.get_single(self, value_type, file)
                    metadata[key.decode()] = value

        return metadata

class VerboseOutputSupressor(object):
    """
    Suppress stdout and stderr if easy_llama.globals.VERBOSE is False.
    Otherwise, do nothing.

    This prevents llama.cpp's console output from being displayed

    Changing VERBOSE inside the WITH block may result in stdout and stderr
    being stuck to /dev/null, or other undefined behaviour

    See https://github.com/abetlen/llama-cpp-python/issues/478
    """
    
    def __enter__(self):
        if globals.VERBOSE:
            return self
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

    def __exit__(self, *_):
        if globals.VERBOSE:
            return
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()

def get_timestamp_prefix_str() -> str:
    # helpful: https://strftime.net
    return time.strftime("[%Y, %b %e, %a %l:%M %p] ")

def print_warning(text: str) -> str:
    print("easy_llama: warning:", text, file=sys.stderr, flush=True)

def verify_backend(backend, num_gpu_layers) -> tuple:
    """
    Verify that BACKEND is valid and return a tuple of valid values for
    (backend, num_gpu_layers, mul_mat_q, mmap, mlock)

    This is not done on import because user must be able to set backend
    (and maybe NUM_GPU_LAYERS) before loading any model.
    """

    if backend is None:
        print_warning(
            "easy_llama.globals.BACKEND is None, defaulting to CPU. " + \
            "set easy_llama.globals.BACKEND to 'metal', 'cuda', or 'rocm' " + \
            "to accelerate inference"
        )
        backend = 'CPU'
    
    if not isinstance(backend, str):
        print_warning(
            "easy_llama.globals.BACKEND must be a string, " + \
            f"not {type(backend)}. defaulting to CPU"
        )
        backend = 'CPU'
    
    if backend.lower() == 'metal':
        backend = 'Metal'
    elif backend.lower() == 'cuda':
        backend = 'CUDA'
    elif backend.lower() == 'rocm':
        backend = "ROCm"
    elif backend.lower() == 'cpu':
        backend = "CPU"

    if backend not in ['Metal', 'CUDA', 'ROCm', 'CPU']:
        print_warning(
            f"easy_llama.globals.BACKEND '{backend}' is invalid, defaulting to " + \
            "CPU. set easy_llama.globals.BACKEND to 'metal', 'cuda', or 'rocm' " + \
            "to accelerate inference"
        )
        backend = 'CPU'
    
    if backend == 'Metal':
        num_gpu_layers = 1
        mul_mat_q = True
        mmap = False
        mlock = False
    elif backend == 'CUDA':
        # Don't change NUM_GPU_LAYERS, use global value
        mul_mat_q = True
        mmap = False
        mlock = False
    elif backend == 'ROCm':
        # Don't change NUM_GPU_LAYERS, use global value
        mul_mat_q = False
        mmap = False
        mlock = False
    elif backend == 'CPU':
        num_gpu_layers = 0
        mul_mat_q = True
        mmap = True
        mlock = False
    
    if backend in ['CUDA', 'ROCm'] and num_gpu_layers == 0:
        print_warning(
            "CUDA or ROCm is selected but easy_llama.globals.NUM_GPU_LAYERS is 0. " + \
            "set easy_llama.globals.NUM_GPU_LAYERS to 1 or greater to " + \
            "accelerate inference"
        )
    
    return (backend, num_gpu_layers, mul_mat_q, mmap, mlock)