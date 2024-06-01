# __init__.py
# https://github.com/ddh0/easy-llama/
from ._version import __version__, __llama_cpp_version__

"""
Text generation in Python, as easy as possible

For documentation, see:
https://github.com/ddh0/easy-llama/blob/main/DOCS.md
"""

from . import samplers
from . import formats
from . import thread
from . import utils
from . import model

from .model  import Model
from .thread import Thread
