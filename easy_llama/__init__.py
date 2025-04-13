# __init__.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from ._version import __version__

"""Text generation in Python, as easy as possible

For more information, visit the project's GitHub repository:
https://github.com/ddh0/easy-llama"""

# all submodules

from . import libllama
from . import sampling
from . import formats
from . import server
from . import thread
from . import llama
from . import webui # TODO: remove
from . import utils

# shorthands, so you can do `ez.Llama` instead of `ez.llama.Llama`, etc.

from .sampling import SamplerParams, SamplerPreset, SamplerPresets
from .formats  import PromptFormat, PromptFormats, SystemPrompts
from .llama    import Llama, get_verbose, set_verbose
from .server   import Server
from .thread   import Thread
from .webui    import WebUI # TODO: remove
