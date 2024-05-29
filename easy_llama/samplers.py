# samplers.py
# https://github.com/ddh0/easy-llama/
from ._version import __version__, __llama_cpp_version__

"""Submodule containing SamplerSettings class and some preset samplers"""

from sys import maxsize

from .utils import assert_type


MAX_TEMP = float(maxsize)

class SamplerSettings:
    """
    A SamplerSettings object specifies the sampling parameters that will be
    used to control text generation. It is passed as an optional parameter to
    `Thread()`, `Model.generate()`, `Model.stream()`, and
    `Model.stream_print()`.
    """

    param_types: dict[str, type] = {
        'max_len_tokens':    int,
        'temp':              float,
        'top_p':             float,
        'min_p':             float,
        'frequency_penalty': float,
        'presence_penalty':  float,
        'repeat_penalty':    float,
        'top_k':             int
    }

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
        """
        Construct a new SamplerSettings instance
        """

        self.max_len_tokens    = max_len_tokens
        self.temp              = temp
        self.top_p             = top_p
        self.min_p             = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty  = presence_penalty
        self.repeat_penalty    = repeat_penalty
        self.top_k             = top_k

        for sampler_param in SamplerSettings.param_types:
            assert_type(
                getattr(self, sampler_param),
                SamplerSettings.param_types[sampler_param],
                f'{sampler_param} parameter',
                'SamplerSettings'
            )
    
    def __repr__(self) -> str:
        return \
            'SamplerSettings(' \
            f'max_len_tokens={self.max_len_tokens}, ' \
            f'temp={self.temp}, ' \
            f'top_p={self.top_p}, ' \
            f'min_p={self.min_p}, ' \
            f'frequency_penalty={self.frequency_penalty}, ' \
            f'presence_penalty={self.presence_penalty}, ' \
            f'repeat_penalty={self.repeat_penalty}, ' \
            f'top_k={self.top_k})'

# most likely token is always chosen
GreedyDecoding = SamplerSettings(
    temp = 0.0,
)

# reflects llama.cpp
DefaultSampling = SamplerSettings()

# unmodified probability distribution (i.e. what the model actually thinks)
SimpleSampling = SamplerSettings(
    temp = 1.0,
    top_p = 1.0,
    min_p = 0.0,
    top_k = -1
)

# reflects old llama.cpp defaults
ClassicSampling = SamplerSettings(
    min_p = 0.0,
    repeat_penalty = 1.1
)

# halfway between DefaultSampling and SimpleSampling
SemiSampling = SamplerSettings(
    temp = 0.9,
    top_p = 0.975,
    min_p = 0.025,
    top_k = 80
)

# for models with large vocabulary, which tend to run hot
TikTokenSampling = SamplerSettings(
    temp=0.6,
    repeat_penalty=1.1
)

# use min_p as the only active sampler (more permissive)
LowMinPSampling = SamplerSettings(
    temp = 1.0,
    top_p = 1.0,
    min_p = 0.05,
    top_k = -1
)

# use min_p as the only active sampler (moderate)
MinPSampling = SamplerSettings(
    temp = 1.0,
    top_p = 1.0,
    min_p = 0.1,
    top_k = -1
)

# use min_p as the only active sampler (more restrictive)
StrictMinPSampling = SamplerSettings(
    temp = 1.0,
    top_p = 1.0,
    min_p = 0.2,
    top_k = -1
)

# https://arxiv.org/abs/2210.14140
ContrastiveSearch = SamplerSettings(
    temp = 0.0,
    presence_penalty = 0.6
)

# https://arxiv.org/abs/2210.14140
WarmContrastiveSearch = SamplerSettings(
    temp = 0.0,
    presence_penalty = 1.0
)

# outputs completely random tokens from vocab (useless)
RandomSampling = SamplerSettings(
    temp = MAX_TEMP,
    top_p = 1.0,
    min_p = 0.0,
    top_k = -1
)

# default sampling with reduced temperature
LowTempSampling = SamplerSettings(
    temp = 0.4
)

# default sampling with increased temperature
HighTempSampling = SamplerSettings(
    temp = 1.2
)

#
# Samplers below this line are for specific models / model families
#

# https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5
MiquSampling = SamplerSettings(
    temp = 1.0,
    top_p = 1.0,
    min_p = 0.12,
    frequency_penalty = 0.0,
    presence_penalty = 0.0,
    repeat_penalty = 1.05,
    top_k = -1
)
