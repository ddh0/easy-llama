# easy_llama.py
# Python 3.11.6

"""
Text generation in Python, made easy

https://github.com/ddh0/easy-llama/
"""

import globals
import formats
import samplers

from model import Model
from thread import Thread

# TODO: Thread.add_message as shorthand for T.messages.append(T.create_message) ?
# TODO: Model.ingest() ?
# TODO: function to transfer a list of messages between disk / models, handle token count
# TODO: Model.next_candidates() -> list[str]

if __name__ == "__main__":
    raise RuntimeError(
        "easy_llama cannot be run directly, please import it into " + \
        "your environment"
    )