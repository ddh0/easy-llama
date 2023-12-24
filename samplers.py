# samplers.py
# Python 3.11.6

import sys

class SamplerSettings(object):

    def __init__(
            self,
            max_len_tokens:   int   = None,
            temp:             float = 0.8,
            top_p:            float = 0.95,
            min_p:            float = 0.05,
            presence_penalty: float = 0.0,
            repeat_penalty:   float = 1.1,
            top_k:            int   = 40
        ):

        self.max_len_tokens   = max_len_tokens
        self.temp             = temp
        self.top_p            = top_p
        self.min_p            = min_p
        self.presence_penalty = presence_penalty
        self.repeat_penalty   = repeat_penalty
        self.top_k            = top_k


GreedyDecoding = SamplerSettings(
    temp=0.0,
    repeat_penalty=1.0
)

DefaultSampling = SamplerSettings()

MinPSampling = SamplerSettings(
    top_p = 1.0,
    min_p = 0.1,
    repeat_penalty = 1.0,
    top_k = -1
)

ContrastiveSearch = SamplerSettings(
    top_p = 1.0,
    min_p = 0.0,
    presence_penalty = 0.5,
    repeat_penalty = 1.0,
    top_k = -1
)

RandomSampling = SamplerSettings(
    temp = float(sys.maxsize),
    top_p = 1.0,
    min_p = 0.0,
    repeat_penalty = 1.0,
    top_k = -1
)