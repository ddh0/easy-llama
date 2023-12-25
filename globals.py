# globals.py
# Python 3.11.6

"""Submodule containing global constants used by easy_llama"""

BACKEND:        str  = None       # Modifies NUM_GPU_LAYERS to enable or disable acceleration
NUM_GPU_LAYERS: int  = 0          # Default value only. Will be changed at runtime or per BACKEND
MUL_MAT_Q:      bool = True       # Default value only. Will be changed per BACKEND
MMAP:           bool = True       # Default value only. Will be changed per BACKEND
MLOCK:          bool = False      # Default value only. Will be changed per BACKEND
VERBOSE:        bool = False      # Do not suppress llama.cpp console output
SEED:           int  = -1         # -1 -> Random seed