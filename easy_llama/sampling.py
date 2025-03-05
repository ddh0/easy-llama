# sampling.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""This file provides functionality for defining the sampler parameters used to control
text generation."""

import os
import sys
import ctypes

from typing    import Optional
from .libllama import _internals
from .utils    import null_ptr_check, log, ez_encode

from . import libllama as lib

HIGH_TEMP = 10_000.0

class Llama: # can't import the real Llama - would be circular
    """Type hint denoting a `llama.Llama` instance"""

def _get_random_seed() -> int:
    # uint32_t
    return int.from_bytes(bytes=os.urandom(4), byteorder=sys.byteorder, signed=False)

class SamplerParams:
    """A SamplerParams object is used by a Llama model to define sampling behaviour.

    However, SamplerParams objects also require some information about the Llama model itself,
    such as n_ctx_train, n_vocab, etc. Therefore Llama models and SamplerParams are tightly
    coupled.

    A SamplerPreset (which is a separate class) can be used to define sampling parameters
    without having to specify a Llama object. In turn, the Llama class can use these presets to
    create the actual SamplerParams object it needs for sampling."""

    # NOTE: as of 2025-01-01, the default sampler chain for llama-cli is:
    #
    #       logits -> logit-bias -> penalties -> dry -> top-k -> typical ->
    #       top-p -> min-p -> xtc -> temp-ext -> dist
    #
    #       ----------------------------------------------------------------
    #
    #       as of 2025-01-01, the sampler chain for easy-llama is:
    #
    #       -- ALWAYS APPLIED:
    #
    #       logits -> logit-bias -> top-k (k=128) -> penalties -> dry -> xtc ...
    #
    #       -- IF TEMP <= 0.0:
    #
    #       ... -> greedy
    #
    #       -- IF MIROSTAT v1:
    #
    #       ... -> temp(-ext) -> mirostat-v1
    #
    #       -- IF MIROSTAT v2:
    #
    #       ... -> temp(-ext) -> mirostat-v2
    #
    #       -- DEFAULT CASE:
    #
    #       ... top-k -> typical -> top-p -> min-p -> temp(-ext) -> dist
    #       
    #       ----------------------------------------------------------------
    #
    #       - "temp(-ext)" denotes "temp" if dynatemp_range == 0.0, otherwise
    #         "temp-ext"
    #
    #       - "top-k (k=128)" is always performed before applying penalties
    #         and DRY to improve performance
    #
    #       - note that "logit-bias", "top-k (k=128)", "penalties", and "dry"
    #         are always applied
    
    def __init__( 
        #
        # ref: llama.cpp/common/common.h: struct common_params_sampling { ... }
        #
        self,
        llama: Llama,    # some samplers require info about n_ctx_train, n_vocab, etc.
        seed:  int = -1, # the seed used to initialize llama_sampler; if <= 0, use random seed

        top_k:              int   = 40,    # <= 0 to use vocab size
        top_p:              float = 0.95,  # 1.0 = disabled
        min_p:              float = 0.05,  # 0.0 = disabled
        xtc_probability:    float = 0.0,   # 0.0 = disabled
        xtc_threshold:      float = 0.1,   # > 0.5 disables XTC
        typical_p:          float = 1.0,   # 1.0 = disabled 
        temp:               float = 0.8,   # <= 0.0 to sample greedily
        dynatemp_delta:     float = 0.0,   # 0.0 = disabled
        dynatemp_exponent:  float = 1.0,   # controls how entropy maps to temperature in dynamic temperature sampler
        penalty_last_n:     int   = 64,    # last n tokens to penalize (0 = disable penalty, -1 = context size)
        penalty_repeat:     float = 1.0,   # 1.0 = disabled
        penalty_freq:       float = 0.0,   # 0.0 = disabled
        penalty_present:    float = 0.0,   # 0.0 = disabled
        dry_multiplier:     float = 0.0,   # 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
        dry_base:           float = 1.75,  # 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
        dry_allowed_length: int   = 2,     # tokens extending repetitions beyond this receive penalty
        dry_penalty_last_n: int   = -1,    # how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
        mirostat:           int   = 0,     # 0 = disabled, 1 = mirostat v1, 2 = mirostat v2
        mirostat_tau:       float = 5.0,   # target entropy
        mirostat_eta:       float = 0.1,   # learning rate

        dry_sequence_breakers: list[str] = ["\n", ":", "\"", "*"], # default sequence breakers for DRY

        # TODO: grammar goes here

        logit_bias: Optional[dict[int, float]] = None
    ):
        self.smpl = None
        
        #
        # ref: llama.cpp/common/common.h: common_sampler_init(...) { ... }
        #

        #
        # Store parameter values as attributes
        #

        # NOTE: Changing these attributes will not change the sampling

        self.llama = llama
        self.seed  = seed

        self.top_k              = top_k
        self.top_p              = top_p
        self.min_p              = min_p
        self.xtc_probability    = xtc_probability
        self.xtc_threshold      = xtc_threshold
        self.typical_p          = typical_p
        self.temp               = temp
        self.dynatemp_delta     = dynatemp_delta
        self.dynatemp_exponent  = dynatemp_exponent
        self.penalty_last_n     = penalty_last_n
        self.penalty_repeat     = penalty_repeat
        self.penalty_freq       = penalty_freq
        self.penalty_present    = penalty_present
        self.dry_multiplier     = dry_multiplier
        self.dry_base           = dry_base
        self.dry_allowed_length = dry_allowed_length
        self.dry_penalty_last_n = dry_penalty_last_n
        self.mirostat           = mirostat
        self.mirostat_tau       = mirostat_tau
        self.mirostat_eta       = mirostat_eta
        
        self.dry_sequence_breakers = dry_sequence_breakers

        self.logit_bias = logit_bias

        self._chain_str = ''

        sparams = lib.llama_sampler_chain_default_params()
        null_ptr_check(sparams, 'sparams', 'SamplerParams.__init__')

        smpl = lib.llama_sampler_chain_init(sparams)
        null_ptr_check(smpl, 'smpl', 'SamplerParams.__init__')

        # Logit bias

        if logit_bias is not None:
            self._chain_str += f'{len(logit_bias)} logit biases -> '
            logit_bias_arr = _internals.get_logit_bias_array(logit_bias)
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_logit_bias(
                n_vocab=llama._n_vocab,
                n_logit_bias=len(logit_bias),
                logit_bias=logit_bias_arr
            ))
        
        # Top-K (where k == 128)
        #
        # NOTE: This improves performance by greatly reducing the number of
        #       tokens that the penalties and DRY samplers need to consider. It
        #       should have absolutely no effect on the output except under the
        #       strangest and most unlikely circumstances (like when temp > 10.0
        #       and all other samplers are explicitly disabled).
        #
        #       If you really need to bypass this, you can construct your own
        #       llama_sampler_chain manually. But you probably don't need to.
        #
        self._chain_str += 'top-k 128 -> '
        lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_top_k(k=128))

        # Penalties
        
        if any([penalty_repeat != 1.0, penalty_present != 0.0, penalty_freq != 0.0]):
            _str = 'penalties'
            _str += f' repeat {penalty_repeat:1.2f}' if penalty_repeat != 1.0 else ''
            _str += f' presence {penalty_present:1.2f}' if penalty_present != 0.0 else ''
            _str += f' frequency {penalty_freq:1.2f}' if penalty_freq != 0.0 else ''
            _last_n = "all" if penalty_last_n < 0 else f"last {penalty_last_n}"
            _str += f' {_last_n}  -> '
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_penalties(
                penalty_last_n=penalty_last_n if penalty_last_n >= 0 else llama._n_ctx,
                penalty_repeat=penalty_repeat,
                penalty_freq=penalty_freq,
                penalty_present=penalty_present
            ))
        
        # DRY
        
        if dry_multiplier > 0.0:
            self._chain_str += 'DRY -> '
            # dry == D.R.Y. ("Don't Repeat Yourself")
            # ref: https://github.com/oobabooga/text-generation-webui/pull/5677
            null_ptr_check(llama._vocab, 'llama._vocab', 'SamplerParams.__init__')
            seq_breakers = dry_sequence_breakers
            seq_breakers_bytes = [ez_encode(s) for s in seq_breakers]
            arr = (ctypes.c_char_p * len(seq_breakers_bytes))(*seq_breakers_bytes)
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_dry(
                vocab=llama._vocab,
                n_ctx_train=llama._n_ctx_train,
                dry_multiplier=dry_multiplier,
                dry_base=dry_base,
                dry_allowed_length=dry_allowed_length,
                dry_penalty_last_n=dry_penalty_last_n,
                seq_breakers=arr,
                num_breakers=len(seq_breakers)
            ))
        
        # XTC
        
        if xtc_probability > 0.0:
            self._chain_str += 'XTC -> '
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_xtc(
                    p=xtc_probability,
                    t=xtc_threshold,
                    min_keep=1,
                    seed=seed if seed > 0 else _get_random_seed()
                )
            )
        
        # IF TEMP <= 0.0:

        if temp <= 0.0:
            # ... -> greedy
            self._chain_str += 'greedy'
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_greedy())

        # IF MIROSTAT v1:

        elif mirostat == 1:
            # ... -> temp(-ext) -> mirostat-v1
            if dynatemp_delta != 0.0:
                # dynamic temperature AKA entropy sampling
                self._chain_str += 'temp-ext -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp_ext(
                        t=temp,
                        delta=dynatemp_delta,
                        exponent=dynatemp_exponent
                    )
                )
            else:
                # standard temperature
                self._chain_str += 'temp -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp(t=temp)
                )
            self._chain_str += 'mirostat v1'
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_mirostat(
                    seed=seed if seed > 0 else _get_random_seed(),
                    tau=mirostat_tau,
                    eta=mirostat_eta
                )
            )
        
        # IF MIROSTAT v2:

        elif mirostat == 2:
            # ... -> temp(-ext) -> mirostat-v2
            if dynatemp_delta != 0.0:
                # dynamic temperature AKA entropy sampling
                self._chain_str += 'temp-ext -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp_ext(
                        t=temp,
                        delta=dynatemp_delta,
                        exponent=dynatemp_exponent
                    )
                )
            else:
                # standard temperature
                self._chain_str += 'temp -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp(t=temp)
                )
            self._chain_str += 'mirostat v2'
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_mirostat_v2(
                    seed=seed if seed > 0 else _get_random_seed(),
                    tau=mirostat_tau,
                    eta=mirostat_eta
                )
            )
        
        # DEFAULT CASE

        elif mirostat == 0:
            # ... -> top-k -> typical -> top-p -> min-p -> temp(-ext) -> dist
            if top_k > 0:
                self._chain_str += 'top-k -> '
                lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_top_k(k=top_k))
            if typical_p != 1.0:
                self._chain_str += 'typical -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_typical(p=typical_p, min_keep=1)
                )
            if top_p < 1.0:
                self._chain_str += 'top-p -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_top_p(p=top_p, min_keep=1)
                )
            if min_p > 0.0:
                self._chain_str += 'min-p -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_min_p(p=min_p, min_keep=1)
                )
            if dynatemp_delta != 0.0:
                # dynamic temperature AKA entropy sampling
                self._chain_str += 'temp-ext -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp_ext(
                        t=temp,
                        delta=dynatemp_delta,
                        exponent=dynatemp_exponent
                    )
                )
            else:
                # standard temperature
                self._chain_str += 'temp -> '
                lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_temp(t=temp))
            self._chain_str += 'dist'
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_dist(seed=seed if seed > 0 else _get_random_seed())
            )

        else:
            raise ValueError(
                f'SamplerParams.__init__: unknown mirostat version {mirostat!r}'
            )

        self.smpl = smpl
    
    def __del__(self):
        self.free()
    
    def __repr__(self) -> str:
        return (
            f"SamplerParams("
            f"llama=<Llama object>, "
            f"seed={self.seed}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"xtc_probability={self.xtc_probability}, "
            f"xtc_threshold={self.xtc_threshold}, "
            f"typical_p={self.typical_p}, "
            f"temp={self.temp}, "
            f"dynatemp_delta={self.dynatemp_delta}, "
            f"dynatemp_exponent={self.dynatemp_exponent}, "
            f"penalty_last_n={self.penalty_last_n}, "
            f"penalty_repeat={self.penalty_repeat}, "
            f"penalty_freq={self.penalty_freq}, "
            f"penalty_present={self.penalty_present}, "
            f"dry_multiplayer={self.dry_multiplier}, "
            f"dry_base={self.dry_base}, "
            f"dry_allowed_length={self.dry_allowed_length}, "
            f"dry_penalty_last_n={self.dry_penalty_last_n}, "
            f"mirostat={self.mirostat}, "
            f"mirostat_tau={self.mirostat_tau}, "
            f"mirostat_eta={self.mirostat_eta}, "
            f"dry_sequence_breakers={self.dry_sequence_breakers!r}, "
            f"logit_bias={self.logit_bias!r}"
            f")"
        )
    
    def print_chain(self) -> None:
        log(f'sampler chain: {self._chain_str}')
    
    def free(self) -> None:
        if self.smpl is not None:
            lib.llama_sampler_free(self.smpl)
            self.smpl = None
    
    def reset(self) -> None:
        null_ptr_check(self.smpl, 'self.smpl', 'SamplerParams.reset')
        lib.llama_sampler_reset(self.smpl)

class SamplerPreset:
    """A SamplerPreset object contains all the values necessary to construct a SamplerParams
    object using a Llama model.

    Llama objects use SamplerParam objects to define the sampling parameters, but these
    SamplerParam objects also require some information about the Llama model itself, such as
    n_ctx_train, n_vocab, etc. Therefore Llama models and SamplerParams are tightly coupled.

    A SamplerPreset (this class) can be used to define sampling parameters without having to
    specify a Llama object. In turn, the Llama class can use these presets to create the actual
    SamplerParams object it needs for sampling."""
    
    def __init__(
        self,
        seed:  int = -1, # the seed used to initialize llama_sampler; if <= 0, use random seed

        top_k:              int   = 40,    # <= 0 to use vocab size
        top_p:              float = 0.95,  # 1.0 = disabled
        min_p:              float = 0.05,  # 0.0 = disabled
        xtc_probability:    float = 0.0,   # 0.0 = disabled
        xtc_threshold:      float = 0.1,   # > 0.5 disables XTC
        typical_p:          float = 1.0,   # 1.0 = disabled 
        temp:               float = 0.8,   # <= 0.0 to sample greedily
        dynatemp_delta:     float = 0.0,   # 0.0 = disabled
        dynatemp_exponent:  float = 1.0,   # controls how entropy maps to temperature in dynamic temperature sampler
        penalty_last_n:     int   = 64,    # last n tokens to penalize (0 = disable penalty, -1 = context size)
        penalty_repeat:     float = 1.0,   # 1.0 = disabled
        penalty_freq:       float = 0.0,   # 0.0 = disabled
        penalty_present:    float = 0.0,   # 0.0 = disabled
        dry_multiplier:     float = 0.0,   # 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
        dry_base:           float = 1.75,  # 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
        dry_allowed_length: int   = 2,     # tokens extending repetitions beyond this receive penalty
        dry_penalty_last_n: int   = -1,    # how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
        mirostat:           int   = 0,     # 0 = disabled, 1 = mirostat v1, 2 = mirostat v2
        mirostat_tau:       float = 5.0,   # target entropy
        mirostat_eta:       float = 0.1,   # learning rate

        dry_sequence_breakers: list[str] = ["\n", ":", "\"", "*"], # default sequence breakers for DRY

        # TODO: grammar goes here

        logit_bias: Optional[dict[int, float]] = None
    ):
        self.seed  = seed

        self.top_k              = top_k
        self.top_p              = top_p
        self.min_p              = min_p
        self.xtc_probability    = xtc_probability
        self.xtc_threshold      = xtc_threshold
        self.typical_p          = typical_p
        self.temp               = temp
        self.dynatemp_delta     = dynatemp_delta
        self.dynatemp_exponent  = dynatemp_exponent
        self.penalty_last_n     = penalty_last_n
        self.penalty_repeat     = penalty_repeat
        self.penalty_freq       = penalty_freq
        self.penalty_present    = penalty_present
        self.dry_multiplier     = dry_multiplier
        self.dry_base           = dry_base
        self.dry_allowed_length = dry_allowed_length
        self.dry_penalty_last_n = dry_penalty_last_n
        self.mirostat           = mirostat
        self.mirostat_tau       = mirostat_tau
        self.mirostat_eta       = mirostat_eta
        
        self.dry_sequence_breakers = dry_sequence_breakers

        self.logit_bias = logit_bias

    def __repr__(self) -> str:
        return (
            f"SamplerPreset("
            f"seed={self.seed}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"xtc_probability={self.xtc_probability}, "
            f"xtc_threshold={self.xtc_threshold}, "
            f"typical_p={self.typical_p}, "
            f"temp={self.temp}, "
            f"dynatemp_delta={self.dynatemp_delta}, "
            f"dynatemp_exponent={self.dynatemp_exponent}, "
            f"penalty_last_n={self.penalty_last_n}, "
            f"penalty_repeat={self.penalty_repeat}, "
            f"penalty_freq={self.penalty_freq}, "
            f"penalty_present={self.penalty_present}, "
            f"dry_multiplayer={self.dry_multiplier}, "
            f"dry_base={self.dry_base}, "
            f"dry_allowed_length={self.dry_allowed_length}, "
            f"dry_penalty_last_n={self.dry_penalty_last_n}, "
            f"mirostat={self.mirostat}, "
            f"mirostat_tau={self.mirostat_tau}, "
            f"mirostat_eta={self.mirostat_eta}, "
            f"dry_sequence_breakers={self.dry_sequence_breakers!r}, "
            f"logit_bias={self.logit_bias!r}"
            f")"
        )

class SamplerPresets:
    """This class contains several ready-made `SamplerPreset` objects that can be used to
    control text generation."""
    
    Greedy = SamplerPreset(
        seed = 1,
        top_k = 1,
        temp = 0.0
    )
    """The most likely token is always chosen"""

    Default = SamplerPreset()
    """The default llama.cpp sampling"""

    Neutral = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.0,
        temp = 1.0
    )
    """All samplers neutralized"""

    TikToken = SamplerPreset(
        temp = 0.65
    )
    """For models with large vocabulary, which may tend to run hot"""

    MinPCool = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.1,
        temp = 1.0
    )
    """Use min-p as the only active sampler (less random)"""

    MinPWarm = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.05,
        temp = 1.0
    )
    """Use min-p as the only active sampler (more random)"""

    ContrastiveSearchCool = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.0,
        temp = 0.4,
        penalty_present = 0.6,
    )
    """Constrastive Search as described in https://arxiv.org/abs/2210.14140 (less random)"""

    ContrastiveSearchWarm = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.0,
        temp = 0.8,
        penalty_present = 0.6
    )
    """Constrastive Search as described in https://arxiv.org/abs/2210.14140 (more random)"""

    Cool = SamplerPreset(
        temp = 0.6
    )
    """Decreased temperature (less random)"""

    Warm = SamplerPreset(
        temp = 1.05
    )
    """Increased temperature (more random)"""

    NucleusSamplingCool = SamplerPreset(
        top_k = -1,
        top_p = 0.25,
        min_p = 0.0,
        temp = 1.0
    )
    """Nucleus sampling as described in https://arxiv.org/abs/1904.09751 (less random)"""

    NucleusSamplingWarm = SamplerPreset(
        top_k = -1,
        top_p = 0.9,
        min_p = 0.0,
        temp = 1.0
    )
    """Nucleus sampling as described in https://arxiv.org/abs/1904.09751 (more random)"""

    #
    # Samplers below this line are for specific models / model families
    #

    MidnightMiqu = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.12,
        temp = 1.0,
        penalty_repeat = 1.05
    )
    """[sophosympatheia/Midnight-Miqu-70B-v1.5](https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5)"""

    Llama3 = SamplerPreset(
        top_k = -1,
        top_p = 0.9,
        min_p = 0.0,
        temp = 0.6
    )
    """[meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/)"""

    Llama3Classic = SamplerPreset(
        top_k = 40,
        top_p = 0.95,
        min_p = 0.05,
        temp = 0.65
    )
    """Unofficial preset based on the developer's personal preference"""

    Llama3Cool = SamplerPreset(
        top_k = -1,
        top_p = 0.9,
        min_p = 0.0,
        temp = 0.45
    )
    """Llama3 preset with reduced temperature (less random)"""

    Llama3Warm = SamplerPreset(
        top_k = -1,
        top_p = 0.9,
        min_p = 0.0,
        temp = 0.9
    )
    """Llama3 preset with increased temperature (more random)"""

    Mistral = SamplerPreset(
        temp = 0.3
    )
    """Mistral models tend to require a lower temperature"""

    Qwen2_5Official = SamplerPreset(
        top_k = 20,
        top_p = 0.8,
        min_p = 0.0,
        temp = 0.7,
        penalty_repeat = 1.05
    )
    """[Qwen/Qwen2.5-14B-Instruct/](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/) 
    (official, but not recommended)"""

    Qwen2_5Recommended = SamplerPreset(
        top_k = -1,
        top_p = 0.9,
        min_p = 0.1,
        temp = 1.0
    )
    """[Qwen/Qwen2.5-14B-Instruct/](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/) 
    (unofficial, but recommended)"""

    Cydonia = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.03,
        temp = 1.0,
        logit_bias = {
            1869: -1.0, # "..."
            4618: -1.0, # " ..."
            6202: -1.0, # " Oh"
            6923: -1.0, # "Oh"
        }
    )
    """[TheDrummer/Cydonia-22B-v1.2](https://huggingface.co/TheDrummer/Cydonia-22B-v1.2)
    (unofficial, but recommended)"""
    
    Evathene = SamplerPreset(
        top_k = -1,
        top_p = 0.98,
        min_p = 0.0,
        temp = 1.0
    )
    """[sophosympatheia/Evathene-v1.3](https://huggingface.co/sophosympatheia/Evathene-v1.3)
    (unofficial, but recommended)"""
