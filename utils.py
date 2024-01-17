# utils.py
# Python 3.11.6

"""Submodule containing convenience functions"""

import sys
import time

def get_timestamp_prefix_str() -> str:
    # helpful: https://strftime.net
    return time.strftime("[%Y, %b %e, %a %l:%M %p] ")

def print_warning(text: str) -> str:
    print("easy_llama: warning:", text, file=sys.stderr, flush=True)

def verify_backend(backend, num_gpu_layers, mul_mat_q) -> tuple:
    """
    Verify that BACKEND is valid and modify NUM_GPU_LAYERS
    and MUL_MAT_Q as necessary

    This is not done on import because user must be able to set backend
    (and maybe NUM_GPU_LAYERS) before loading any model.

    Returns tuple of valid values for
    (backend, num_gpu_layers, mul_mat_q, mmap, mlock)
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
        # Don't set NUM_GPU_LAYERS, let the user configure it
        mul_mat_q = True
        mmap = False
        mlock = False
    elif backend == 'ROCm':
        # Don't set NUM_GPU_LAYERS, let the user configure it
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