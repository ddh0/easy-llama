# __init__.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""This is easy-llama, a Python wrapper over the C/C++ API (`libllama`) provided by
[`llama.cpp`](https://github.com/ggml-org/llama.cpp). It is primarily intended for developers
and machine learning hobbyists seeking to integrate on-device language models (LLMs) into their
applications.

For more information, visit the project's GitHub repository:
https://github.com/ddh0/easy-llama"""

# package version (pyproject.toml reads from here)

__version__ = '0.2.10'

# submodules

from . import libllama
from . import sampling
from . import formats
from . import server
from . import thread
from . import llama
from . import utils

# shorthands, so you can do `ez.Llama` instead of `ez.llama.Llama`, etc.

from .utils    import get_verbose, set_verbose, get_debug, set_debug
from .sampling import SamplerParams, SamplerPreset, SamplerPresets
from .formats  import PromptFormat, PromptFormats, SystemPrompts
from .thread   import Thread, Role
from .server   import Server
from .llama    import Llama
