# __init__.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from ._version import __version__

"""
Text generation in Python, as easy as possible

For documentation, see:
https://github.com/ddh0/easy-llama/blob/main/DOCS.md
"""

# all submodules

from . import libllama
from . import sampling
from . import formats
from . import thread
from . import llama
from . import webui
from . import utils

# shortcuts, so you can do `ez.Llama` instead of `ez.llama.Llama`, etc.

from .sampling import SamplerParams, SamplerPresets
from .formats  import PromptFormat
from .thread   import Thread
from .llama    import Llama
from .webui    import WebUI
