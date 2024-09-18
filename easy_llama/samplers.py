# samplers.py
# https://github.com/ddh0/easy-llama/

"""Submodule containing SamplerSettings class and some preset samplers"""

from typing import Optional
from sys    import maxsize

from .utils import assert_type, NoneType


MAX_TEMP = float(maxsize)

class SamplerSettings:
    """
    A SamplerSettings object specifies the sampling parameters that will be
    used to control text generation. It is passed as an optional parameter to
    `Thread.__init__()`, `Model.generate()`, `Model.stream()`,
    `Model.stream_print()`, etc.
    """

    param_types: dict[str, tuple[type]] = {
        'max_len_tokens'    : (int,   NoneType),
        'top_k'             : (int,   NoneType),
        'top_p'             : (float, NoneType),
        'min_p'             : (float, NoneType),
        'temp'              : (float, NoneType),
        'frequency_penalty' : (float, NoneType),
        'presence_penalty'  : (float, NoneType),
        'repeat_penalty'    : (float, NoneType)
    }

    def __init__(
        self,
        max_len_tokens    : Optional[int]   = -1,
        top_k             : Optional[int]   = 40,
        top_p             : Optional[float] = 0.95,
        min_p             : Optional[float] = 0.05,
        temp              : Optional[float] = 0.8,
        frequency_penalty : Optional[float] = 0.0,
        presence_penalty  : Optional[float] = 0.0,
        repeat_penalty    : Optional[float] = 1.0
    ):
        """
        Construct a new SamplerSettings instance

        If a sampler parameter is unspecified, the default value from llama.cpp
        is used. If all samplers are unspecified, the behaviour is equivalent
        to `DefaultSampling`.

        If a sampler parameter is explicitly set to `None`, it will be disabled.
        When all samplers are disabled, the behaviour is equivalent to the
        preset `NoSampling` (which is the unmodified probability distribution).

        For greedy decoding, see the preset `GreedyDecoding`.
        """

        self.max_len_tokens = (
            max_len_tokens if max_len_tokens is not None else -1
        )
        self.top_k = (
            top_k if top_k is not None else -1
        )
        self.top_p = (
            top_p if top_p is not None else 1.0
        )
        self.min_p = (
            min_p if min_p is not None else 0.0
        )
        self.temp = (
            temp if temp is not None else 1.0
        )
        self.frequency_penalty = (
            frequency_penalty if frequency_penalty is not None else 0.0
        )
        self.presence_penalty = (
            presence_penalty if presence_penalty is not None else 0.0
        )
        self.repeat_penalty = (
            repeat_penalty if repeat_penalty is not None else 1.0
        )

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
            f'top_k={self.top_k}, ' \
            f'top_p={self.top_p}, ' \
            f'min_p={self.min_p}, ' \
            f'temp={self.temp}, ' \
            f'frequency_penalty={self.frequency_penalty}, ' \
            f'presence_penalty={self.presence_penalty}, ' \
            f'repeat_penalty={self.repeat_penalty}' \
            f')'

# most likely token is always chosen
GreedyDecoding = SamplerSettings(
    temp = 0.0
)

# reflects llama.cpp
DefaultSampling = SamplerSettings()

# unmodified probability distribution (i.e. what the model actually thinks)
NoSampling = SimpleSampling = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = None,
    temp = None
)

# reflects old llama.cpp defaults
ClassicSampling = SamplerSettings(
    min_p = None,
    repeat_penalty = 1.1
)

# halfway between DefaultSampling and SimpleSampling
SemiSampling = SamplerSettings(
    top_k = 80,
    top_p = 0.975,
    min_p = 0.025,
    temp = 0.9
)

# for models with large vocabulary, which tend to run hot
TikTokenSampling = SamplerSettings(
    temp = 0.65
)

# use min_p as the only active sampler (more permissive)
LowMinPSampling = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = 0.01,
    temp = None
)

# use min_p as the only active sampler (moderate)
MinPSampling = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = 0.075,
    temp = None
)

# use min_p as the only active sampler (more restrictive)
StrictMinPSampling = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = 0.2,
    temp = None
)

# https://arxiv.org/abs/2210.14140
ContrastiveSearch = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = None,
    temp = 0.0,
    presence_penalty = 0.6,
)

# https://arxiv.org/abs/2210.14140
WarmContrastiveSearch = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = None,
    temp = 0.0,
    presence_penalty = 1.0
)

# outputs completely random tokens from vocab (useless)
RandomSampling = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = None,
    temp = MAX_TEMP
)

# default sampling with reduced temperature
LowTempSampling = SamplerSettings(
    temp = 0.4
)

# default sampling with increased temperature
HighTempSampling = SamplerSettings(
    temp = 1.1
)

# https://arxiv.org/abs/1904.09751
LowTopPSampling = SamplerSettings(
    top_k=None,
    top_p=0.98,
    min_p=None,
    temp=None
)

# https://arxiv.org/abs/1904.09751
TopPSampling = SamplerSettings(
    top_k=None,
    top_p=0.9,
    min_p=None,
    temp=None
)

# https://arxiv.org/abs/1904.09751
StrictTopPSampling = SamplerSettings(
    top_k=None,
    top_p=0.7,
    min_p=None,
    temp=None
)

#
# Samplers below this line are for specific models / model families
#

# https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5
MidnightMiqu = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = 0.12,
    temp = 1.0,
    repeat_penalty = 1.05
)

# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/generation_config.json
Llama3 = SamplerSettings(
    top_k = None,
    top_p = 0.9,
    min_p = None,
    temp = 0.6
)

#https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
Nemo = MistralNemo = MistralSmall = SamplerSettings(
    top_k = None,
    top_p = None,
    min_p = None,
    temp = 0.3
)
