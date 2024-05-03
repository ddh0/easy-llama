# easy_llama.py
# https://github.com/ddh0/easy-llama/

"""
Text generation in Python, made easy

For documentation, see:
https://github.com/ddh0/easy-llama/blob/main/DOCS.md"""

__version__ = '0.1.11'

from . import formats
from . import samplers
from . import utils

from .model  import Model
from .thread import Thread
