# __init__.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from _version import __version__

"""
Text generation in Python, as easy as possible

For documentation, see:
https://github.com/ddh0/easy-llama/blob/main/DOCS.md
"""

# import all submodules

import libllama
import sampling
import formats
import thread
import llama
import webui
import utils
import model

# shortcuts, so you can do `ez.Llama` instead of `ez.llama.Llama`, etc.

from sampling import SamplerParams
from thread   import Thread
from llama    import Llama
from webui    import WebUI
from model    import Model
