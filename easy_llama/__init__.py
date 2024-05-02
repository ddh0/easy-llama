# easy_llama.py
# https://github.com/ddh0/easy-llama/

"""Text generation in Python, made easy"""

__version__ = '0.1.8'
__x__ = 'testing'

from . import formats
from . import samplers
from . import utils

from .model  import Model
from .thread import Thread
