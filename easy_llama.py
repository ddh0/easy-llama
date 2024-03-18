# easy_llama.py
# Python 3.11.7
# https://github.com/ddh0/easy-llama/

"""Text generation in Python, made easy"""

import globals
import formats
import samplers
import utils as _utils

from model import Model
from thread import Thread

# TODO: Model.next_candidates() -> list[str]

if __name__ == "__main__":
    raise RuntimeError(
        "easy_llama cannot be run directly, please import it into " + \
        "your environment"
    )
