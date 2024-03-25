# samplers.py
# Python 3.11.7
# https://github.com/ddh0/easy-llama/

"""Submodule containing SamplerSettings class and some preset samplers"""

from sys import maxsize

MAX_TEMP = float(maxsize)

class SamplerSettings(object):

    def __init__(
            self,
            max_len_tokens:    int   = -1,
            temp:              float = 0.8,
            top_p:             float = 0.95,
            min_p:             float = 0.05,
            frequency_penalty: float = 0.0,
            presence_penalty:  float = 0.0,
            repeat_penalty:    float = 1.0,
            top_k:             int   = 40
        ):

        if not all((all(isinstance(p, int) for p in (
                max_len_tokens, top_k
                    )
                ), all(isinstance(p, float) for p in (
                temp, top_p, min_p, frequency_penalty,
                presence_penalty, repeat_penalty
                    )
                )
            )
        ):
            raise TypeError(
                'wrong type for some parameter of SamplerSettings()'
                )

        self.max_len_tokens    = max_len_tokens
        self.temp              = temp
        self.top_p             = top_p
        self.min_p             = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty  = presence_penalty
        self.repeat_penalty    = repeat_penalty
        self.top_k             = top_k
    
    def __repr__(self) -> str:
        repr_str = 'SamplerSettings('
        repr_str += f'max_len_tokens={self.max_len_tokens}, '
        repr_str += f'temp={self.temp}, '
        repr_str += f'top_p={self.top_p}, '
        repr_str += f'min_p={self.min_p}, '
        repr_str += f'frequency_penalty={self.frequency_penalty}, '
        repr_str += f'presence_penalty={self.presence_penalty}, '
        repr_str += f'repeat_penalty={self.repeat_penalty}, '
        repr_str += f'top_k={self.top_k})'
        return repr_str

GreedyDecoding = SamplerSettings(
    temp = 0.0,
    repeat_penalty = 1.0
)

DefaultSampling = SamplerSettings()

EasyLlamaSampling = SamplerSettings(
    min_p = 0.2
)

OldDefaultSampling = SamplerSettings(
    repeat_penalty = 1.1
)

SimpleSampling = SamplerSettings(
    temp = 1.0,
    top_p = 1.0,
    min_p = 0.0,
    top_k = -1
)

LowMinPSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.1,
    top_k = -1
)

MinPSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.2,
    top_k = -1
)

StrictMinPSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.5,
    top_k = -1
)

ContrastiveSearch = SamplerSettings(
    temp = 0.0,
    presence_penalty = 0.4
)

WarmContrastiveSearch = SamplerSettings(
    temp = 0.0,
    presence_penalty = 0.8
)

RandomSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.0,
    top_k = -1
)

LowTempSampling = SamplerSettings(
    temp = 0.4
)

HighTempSampling = SamplerSettings(
    temp = 1.2
)
