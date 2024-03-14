# utils.py
# Python 3.11.7

"""
Submodule containing convenience functions and GGUFReader
"""

import globals
import time
import sys
import os

from struct import unpack
from enum import IntEnum

_backend_verified = False

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
            value_length = unpack("<Q", file.read(8))[0]
            value = file.read(value_length)
            value = value.decode("utf-8")
        else:
            type_str = GGUFReader._simple_value_packing.get(value_type)
            bytes_length = GGUFReader.value_type_info.get(value_type)
            value = unpack(type_str, file.read(bytes_length))[0]

        return value

    def load_metadata(self, fname) -> dict:
        metadata = {}
        with open(fname, "rb") as file:
            GGUF_MAGIC = file.read(4)
            GGUF_VERSION = unpack("<I", file.read(4))[0]
            # ti_data_count = struct.unpack("<Q", file.read(8))[0]
            file.read(8)
            kv_data_count = unpack("<Q", file.read(8))[0]

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
                key_length = unpack("<Q", file.read(8))[0]
                key = file.read(key_length)

                value_type = GGUFReader.GGUFValueType(
                    unpack("<I", file.read(4))[0]
                )
                if value_type == GGUFReader.GGUFValueType.ARRAY:
                    ltype = GGUFReader.GGUFValueType(
                        unpack("<I", file.read(4))[0]
                    )
                    length = unpack("<Q", file.read(8))[0]
                    for _ in range(length):
                        _ = GGUFReader.get_single(self, ltype, file)
                else:
                    value = GGUFReader.get_single(self, value_type, file)
                    metadata[key.decode()] = value

        return metadata

def cls() -> None:
    """Clear the terminal"""
    print("\033c\033[3J", end='', flush=True)

def sync_llama_verbose_global(llama):
    """Ensure llama.verbose is synced with globals.VERBOSE"""
    llama.verbose = globals.VERBOSE

def get_timestamp_prefix_str() -> str:
    # helpful: https://strftime.net
    return time.strftime("[%Y, %b %e, %a %l:%M %p] ")

def print_warning(text: str) -> str:
    print("easy_llama: warning:", text, file=sys.stderr, flush=True)

def verify_backend() -> tuple:
    """
    Verify that BACKEND is valid and return a tuple of valid values for
    (mul_mat_q, mmap, mlock, offload_kqv).
    
    Potentially modify values of
    globals.BACKEND and globals.NUM_GPU_LAYERS

    This is not done on import because user must be able to set backend
    (and maybe NUM_GPU_LAYERS) before loading any model.
    """

    global _backend_verified

    backend = globals.BACKEND
    num_gpu_layers = globals.NUM_GPU_LAYERS

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
    
    # for pretty printing in verbose mode
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
    
    globals.BACKEND = backend
    
    if backend in ['CUDA', 'ROCm'] and num_gpu_layers == 0:
        print_warning(
            "CUDA or ROCm is selected but easy_llama.globals.NUM_GPU_LAYERS is 0. " + \
            "set easy_llama.globals.NUM_GPU_LAYERS to 1 or greater to " + \
            "accelerate inference"
        )
    
    if backend == 'Metal':
        # Don't change NUM_GPU_LAYERS, use global value
        mul_mat_q = True
        mmap = False
        mlock = False
        offload_kqv = True
    elif backend == 'CUDA':
        # Don't change NUM_GPU_LAYERS, use global value
        mul_mat_q = True
        mmap = False
        mlock = False
        offload_kqv = True
    elif backend == 'ROCm':
        # Don't change NUM_GPU_LAYERS, use global value
        mul_mat_q = False
        mmap = False
        mlock = False
        offload_kqv = True
    elif backend == 'CPU':
        globals.NUM_GPU_LAYERS = 0
        mul_mat_q = True
        mmap = True
        mlock = False
        offload_kqv = False

    _backend_verified = True

    return (mul_mat_q, mmap, mlock, offload_kqv)

def get_optimal_n_batch(cpu_count: int) -> int:

    if not _backend_verified:
        raise RuntimeError(
            "attempt to run get_optimal_n_batch() before verify_backend()"
        )

    if globals.BACKEND in ["Metal", "CPU"]:
        return cpu_count * 16
    else:
        return max(cpu_count * 48, 512)