# samplers.py
# Python 3.11.7

"""Submodule containing SamplerSettings class and some preset samplers"""

from sys import maxsize

MAX_TEMP = float(maxsize)

class SamplerSettings(object):

    def __init__(
            self,
            max_len_tokens:   int   = -1,
            temp:             float = 0.8,
            top_p:            float = 0.95,
            min_p:            float = 0.05,
            presence_penalty: float = 0.0,
            repeat_penalty:   float = 1.1,
            top_k:            int   = 40
        ):

        if not all((all(isinstance(p, int) for p in (
                max_len_tokens, top_k
                    )
                ), all(isinstance(p, float) for p in (
                temp, top_p, min_p, presence_penalty, repeat_penalty
                    )
                )
            )
        ):
            raise TypeError(
                'wrong type for some parameter of SamplerSettings()'
                )

        self.max_len_tokens   = max_len_tokens
        self.temp             = temp
        self.top_p            = top_p
        self.min_p            = min_p
        self.presence_penalty = presence_penalty
        self.repeat_penalty   = repeat_penalty
        self.top_k            = top_k


GreedyDecoding = SamplerSettings(
    temp = 0.0,
    repeat_penalty = 1.0
)

DefaultSampling = SamplerSettings()

AlternativeSampling = SamplerSettings(
    top_p = 1.0,
    min_p = 0.2,
    repeat_penalty = 1.18,
    top_k = -1
)

LowMinPSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.1,
    repeat_penalty = 1.0,
    top_k = -1
)

MinPSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.2,
    repeat_penalty = 1.0,
    top_k = -1
)

StrictMinPSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.3,
    repeat_penalty = 1.0,
    top_k = -1
)

ContrastiveSearch = SamplerSettings(
    temp = 0.0,
    presence_penalty = 0.4,
    repeat_penalty = 1.0
)

WarmContrastiveSearch = SamplerSettings(
    temp = 0.0,
    presence_penalty = 0.8,
    repeat_penalty = 1.0
)

RandomSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.0,
    repeat_penalty = 1.0,
    top_k = -1
)

LowTempSampling = SamplerSettings(
    temp = 0.4
)

HighTempSampling = SamplerSettings(
    temp = 1.2
)