# globals.py
# Python 3.11.6

"""Submodule containing global constants used by easy_llama"""

BACKEND:        str  = None       # Modifies NUM_GPU_LAYERS to enable or disable acceleration
NUM_GPU_LAYERS: int  = 0          # Default value only. Will be changed at runtime or per BACKEND
VERBOSE:        bool = False      # Do not suppress llama.cpp console output