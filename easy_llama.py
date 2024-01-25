# easy_llama.py
# Python 3.11.7

"""
Text generation in Python, made easy

https://github.com/ddh0/easy-llama/
"""

import globals
import formats
import samplers
import utils as _utils

from model import Model
from thread import Thread

# TODO: function to transfer a list of messages between disk / models, handle token count
# TODO: Model.next_candidates() -> list[str]

if __name__ == "__main__":
    raise RuntimeError(
        "easy_llama cannot be run directly, please import it into " + \
        "your environment"
    )