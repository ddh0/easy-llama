# sampling.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""This file provides functionality for defining the sampler parameters used to control
text generation."""

# TODO: implement grammar

import os
import sys
import ctypes

from .utils    import null_ptr_check, log, ez_encode
from .libllama import _internals
from typing    import Optional

from . import libllama as lib

HIGH_TEMP = 10_000.0

class Llama: # can't import the real Llama - would be circular
    """Type hint denoting a `llama.Llama` instance"""

def _get_random_seed() -> int:
    # unsigned 32-bit integer
    return int.from_bytes(bytes=os.urandom(4), byteorder=sys.byteorder, signed=False)

class _ParamDefaults:
    """Default sampler parameter values"""
    
    seed: int = -1

    top_k:              int   = 64
    top_p:              float = 0.95
    min_p:              float = 0.05
    xtc_probability:    float = 0.0
    xtc_threshold:      float = 0.1
    typical_p:          float = 1.0
    temp:               float = 1.0
    dynatemp_delta:     float = 0.0
    dynatemp_exponent:  float = 1.0
    penalty_last_n:     int   = 64
    penalty_repeat:     float = 1.0
    penalty_freq:       float = 0.0
    penalty_present:    float = 0.0
    dry_multiplier:     float = 0.0
    dry_base:           float = 1.75
    dry_allowed_length: int   = 4
    dry_penalty_last_n: int   = -1
    mirostat:           int   = 0
    top_n_sigma:        float = -1.0
    mirostat_tau:       float = 5.0
    mirostat_eta:       float = 0.1

    dry_sequence_breakers: list[str] = ["\n", "\r", ":", "\"", "*"]

    # TODO: grammar goes here

    logit_bias: Optional[dict[int, float]] = None

class SamplerParams:
    """A SamplerParams object is used by a Llama model to define sampling behaviour.

    However, SamplerParams objects also require some information about the Llama model itself,
    such as n_ctx_train, n_vocab, etc. Therefore Llama models and SamplerParams are tightly
    coupled.

    A SamplerPreset (which is a separate class) can be used to define sampling parameters
    without having to specify a Llama object. In turn, the Llama class can use these presets to
    create the actual SamplerParams object it needs for sampling."""

    # NOTE: as of 2025-04-04, the default sampler chain for llama-cli is:
    #
    #       logits -> logit-bias -> penalties -> dry -> top-k -> typical -> top-p -> min-p ->
    #       xtc -> temp-ext -> dist
    #
    #       -----------------------------------------------------------------------------------
    #
    #       as of 2025-06-03, the sampler chain for easy-llama is constructed as follows:
    #
    #       -----------------------------------------------------------------------------------
    #
    #       logits -> logit-bias -> penalties -> xtc -> dry ...
    #
    #       -- IF TEMP <= 0.0:
    #
    #       ... -> greedy
    #
    #       -- ELIF MIROSTAT v1:
    #
    #       ... -> temp(-ext) -> mirostat-v1
    #
    #       -- ELIF MIROSTAT v2:
    #
    #       ... -> temp(-ext) -> mirostat-v2
    #
    #       -- ELIF TOP-N-SIGMA > 0:
    #
    #       ... top-n-sigma -> temp(-ext) -> dist
    #
    #       -- ELSE:
    #
    #         ... -> top-k -> typical-p -> top-p -> min-p -> temp(-ext) -> dist
    #       
    #       ----------------------------------------------------------------------------------
    #
    #       - "temp(-ext)" denotes "temp" if dynatemp_range == 0.0, otherwise
    #         "temp-ext"
    #
    
    def __init__(
        #
        # ref: llama.cpp/common/common.h: struct common_params_sampling { ... }
        #
        self,
        llama: Llama, # some samplers require info about n_ctx_train, n_vocab, etc.

        seed:               Optional[int]   = None, # random: <= 0
        top_k:              Optional[int]   = None, # neutral: <= 0
        top_p:              Optional[float] = None, # neutral: 1.0
        min_p:              Optional[float] = None, # neutral: 0.0
        xtc_probability:    Optional[float] = None, # neutral: 0.0
        xtc_threshold:      Optional[float] = None, # disable: > 0.5
        typical_p:          Optional[float] = None, # neutral: 1.0
        temp:               Optional[float] = None, # neutral: 1.0, greedy: <= 0.0
        dynatemp_delta:     Optional[float] = None, # neutral: <= 0.0
        dynatemp_exponent:  Optional[float] = None, # controls how entropy maps to dynamic temperature
        penalty_last_n:     Optional[int]   = None, # disable: 0, n_ctx: -1, last n tokens to penalize
        penalty_repeat:     Optional[float] = None, # neutral: 1.0, should be between 1.0 and ~1.1; values < 1.0 will INCREASE repetition
        penalty_freq:       Optional[float] = None, # neutral: 0.0
        penalty_present:    Optional[float] = None, # neutral: 0.0
        dry_multiplier:     Optional[float] = None, # disable: 0.0, DRY repetition penalty for tokens extending repetition:
        dry_base:           Optional[float] = None, # disable: 0.0, multiplier * base ^ (length of sequence before token - allowed length)
        dry_allowed_length: Optional[int]   = None, # tokens extending repetitions beyond this receive penalty
        dry_penalty_last_n: Optional[int]   = None, # disable: 0, n_ctx: -1, how many tokens to scan for repetitions
        mirostat:           Optional[int]   = None, # disable: 0, use v1: 1, use v2: 2
        top_n_sigma:        Optional[float] = None, # disable: -1.0
        mirostat_tau:       Optional[float] = None, # target entropy for mirostat
        mirostat_eta:       Optional[float] = None, # learning rate for mirostat

        dry_sequence_breakers: Optional[list[str]] = None, # sequence breakers for DRY

        # TODO: grammar goes here

        logit_bias: Optional[dict[int, float]] = None # dictionary of one or more {tok_id: bias}
    ):  
        self.smpl = None

        #
        # Store parameter values as attributes
        #

        # NOTE: Changing these attributes after initialization will not change the sampling. If
        #       you need to change the sampling, construct a new sampler.

        self.llama = llama

        self.seed = seed if seed is not None else _ParamDefaults.seed

        self.top_k              = top_k if top_k is not None else _ParamDefaults.top_k
        self.top_p              = top_p if top_p is not None else _ParamDefaults.top_p
        self.min_p              = min_p if min_p is not None else _ParamDefaults.min_p
        self.xtc_probability    = xtc_probability if xtc_probability is not None else (
            _ParamDefaults.xtc_probability
        )
        self.xtc_threshold      = xtc_threshold if xtc_threshold is not None else (
            _ParamDefaults.xtc_threshold
        )
        self.typical_p          = typical_p if typical_p is not None else (
            _ParamDefaults.typical_p
        )
        self.temp               = temp if temp is not None else _ParamDefaults.temp
        self.dynatemp_delta     = dynatemp_delta if dynatemp_delta is not None else (
            _ParamDefaults.dynatemp_delta
        )
        self.dynatemp_exponent  = dynatemp_exponent if dynatemp_exponent is not None else (
            _ParamDefaults.dynatemp_exponent
        )
        self.penalty_last_n     = penalty_last_n if penalty_last_n is not None else (
            _ParamDefaults.penalty_last_n
        )
        self.penalty_repeat     = penalty_repeat if penalty_repeat is not None else (
            _ParamDefaults.penalty_repeat
        )
        self.penalty_freq       = penalty_freq if penalty_freq is not None else (
            _ParamDefaults.penalty_freq
        )
        self.penalty_present    = penalty_present if penalty_present is not None else (
            _ParamDefaults.penalty_present
        )
        self.dry_multiplier     = dry_multiplier if dry_multiplier is not None else (
            _ParamDefaults.dry_multiplier
        )
        self.dry_base           = dry_base if dry_base is not None else _ParamDefaults.dry_base
        self.dry_allowed_length = dry_allowed_length if dry_allowed_length is not None else (
            _ParamDefaults.dry_allowed_length
        )
        self.dry_penalty_last_n = dry_penalty_last_n if dry_penalty_last_n is not None else (
            _ParamDefaults.dry_penalty_last_n
        )
        self.mirostat           = mirostat if mirostat is not None else _ParamDefaults.mirostat
        self.top_n_sigma        = top_n_sigma if top_n_sigma is not None else (
            _ParamDefaults.top_n_sigma
        )
        self.mirostat_tau       = mirostat_tau if mirostat_tau is not None else (
            _ParamDefaults.mirostat_tau
        )
        self.mirostat_eta       = mirostat_eta if mirostat_eta is not None else (
            _ParamDefaults.mirostat_eta
        )
        
        self.dry_sequence_breakers = (
            dry_sequence_breakers if dry_sequence_breakers is not None else
            _ParamDefaults.dry_sequence_breakers
        )

        self.logit_bias = logit_bias if logit_bias is not None else _ParamDefaults.logit_bias

        self._chain_str = ''

        #
        # ref: llama.cpp/common/common.h: common_sampler_init(...) { ... }
        #

        sparams = lib.llama_sampler_chain_default_params()
        null_ptr_check(sparams, 'sparams', 'SamplerParams.__init__')

        smpl = lib.llama_sampler_chain_init(sparams)
        null_ptr_check(smpl, 'smpl', 'SamplerParams.__init__')

        # Logit bias

        if self.logit_bias is not None and len(self.logit_bias) > 0:
            if len(self.logit_bias) == 1:
                self._chain_str += f'one logit bias -> '
            else:
                self._chain_str += f'{len(self.logit_bias)} logit biases -> '
            logit_bias_arr = _internals.get_logit_bias_array(self.logit_bias)
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_logit_bias(
                n_vocab=self.llama._n_vocab,
                n_logit_bias=len(self.logit_bias),
                logit_bias=logit_bias_arr
            ))

        # Penalties

        self.penalty_last_n = self.penalty_last_n if self.penalty_last_n >= 0 else self.llama.n_ctx()

        if self.penalty_last_n != 0 and any(
            [self.penalty_repeat != 1.0, self.penalty_present != 0.0, self.penalty_freq != 0.0]
        ):
            self._chain_str += f'penalty last:{self.penalty_last_n}'
            self._chain_str += f' rept:{self.penalty_repeat:.3f}' if self.penalty_repeat != 1.0 else ''
            self._chain_str += f' pres:{self.penalty_present:.3f}' if self.penalty_present != 0.0 else ''
            self._chain_str += f' freq:{self.penalty_freq:.3f}' if self.penalty_freq != 0.0 else ''
            self._chain_str += f' -> '
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_penalties(
                penalty_last_n=self.penalty_last_n,
                penalty_repeat=self.penalty_repeat,
                penalty_freq=self.penalty_freq,
                penalty_present=self.penalty_present
            ))
        
        # XTC
        
        if self.xtc_probability > 0.0:
            self._chain_str += f'XTC p:{self.xtc_probability:.2f} t:{self.xtc_threshold:.2f} -> '
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_xtc(
                    p=self.xtc_probability,
                    t=self.xtc_threshold,
                    min_keep=1,
                    seed=self.seed if self.seed > 0 else _get_random_seed()
                )
            )
        
        # DRY
        
        if self.dry_multiplier > 0.0:
            self._chain_str += f'DRY x{self.dry_multiplier:.2f} base:{self.dry_base:.2f} '
            self._chain_str += f'len:{self.dry_allowed_length} -> '
            # dry == D.R.Y. ("Don't Repeat Yourself")
            # ref: https://github.com/oobabooga/text-generation-webui/pull/5677
            null_ptr_check(self.llama._vocab, 'llama._vocab', 'SamplerParams.__init__')
            seq_breakers = self.dry_sequence_breakers
            seq_breakers_bytes = [ez_encode(s) for s in seq_breakers]
            arr = (ctypes.c_char_p * len(seq_breakers_bytes))(*seq_breakers_bytes)
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_dry(
                vocab=self.llama._vocab,
                n_ctx_train=self.llama._n_ctx_train,
                dry_multiplier=self.dry_multiplier,
                dry_base=self.dry_base,
                dry_allowed_length=self.dry_allowed_length,
                dry_penalty_last_n=self.dry_penalty_last_n,
                seq_breakers=arr,
                num_breakers=len(seq_breakers)
            ))
        
        # IF TEMP <= 0.0:

        if self.temp <= 0.0:
            # ... -> greedy
            self._chain_str += 'greedy'
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_greedy())

        # ELIF MIROSTAT v1:

        elif self.mirostat == 1:
            # ... -> temp(-ext) -> mirostat-v1
            if self.dynatemp_delta > 0.0:
                # dynamic temperature AKA entropy sampling
                self._chain_str += f'temp {self.temp:.2f} +/- {self.dynatemp_delta:.2f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp_ext(
                        t=self.temp,
                        delta=self.dynatemp_delta,
                        exponent=self.dynatemp_exponent
                    )
                )
            else:
                # standard temperature
                self._chain_str += f'temp {self.temp:.2f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp(t=self.temp)
                )
            
            self._chain_str += f'mirostat v1 tau:{self.mirostat_tau:.2f} eta:{self.mirostat_eta:.2f}'
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_mirostat(
                    seed=self.seed if self.seed > 0 else _get_random_seed(),
                    tau=self.mirostat_tau,
                    eta=self.mirostat_eta
                )
            )
        
        # ELIF MIROSTAT v2:

        elif self.mirostat == 2:
            # ... -> temp(-ext) -> mirostat-v2
            if self.dynatemp_delta > 0.0:
                # dynamic temperature AKA entropy sampling
                self._chain_str += f'temp-ext {self.temp:.2f} +/- {self.dynatemp_delta:.2f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp_ext(
                        t=self.temp,
                        delta=self.dynatemp_delta,
                        exponent=self.dynatemp_exponent
                    )
                )
            else:
                # standard temperature
                self._chain_str += f'temp {self.temp:.2f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp(t=self.temp)
                )
            
            self._chain_str += f'mirostat v2 tau:{self.mirostat_tau:.2f} eta:{self.mirostat_eta:.2f}'
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_mirostat_v2(
                    seed=self.seed if self.seed > 0 else _get_random_seed(),
                    tau=self.mirostat_tau,
                    eta=self.mirostat_eta
                )
            )
        
        # ELIF TOP-N-SIGMA > 0:
        
        elif self.top_n_sigma > 0.0:
            # ... top-n-sigma -> temp(-ext) -> dist
            self._chain_str += (
                f'top-n-sigma {self.top_n_sigma:.2f} ->  '
            )
            
            lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_top_n_sigma(n=self.top_n_sigma)) 

            if self.dynatemp_delta > 0.0:
                # dynamic temperature AKA entropy sampling
                self._chain_str += f'temp {self.temp:.2f} +/- {self.dynatemp_delta:.2f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp_ext(
                        t=self.temp,
                        delta=self.dynatemp_delta,
                        exponent=self.dynatemp_exponent
                    )
                )
            else:
                # standard temperature
                self._chain_str += f'temp {self.temp:.2f} -> '
                lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_temp(t=self.temp))

            self._chain_str += 'dist'
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_dist(seed=self.seed if self.seed > 0 else _get_random_seed())
            )
        
        # ELSE (DEFAULT CASE):

        else:
            # ... -> top-k -> typical-p -> top-p -> min-p -> temp(-ext) -> ...
            if self.mirostat != 0:
                log(f'unknown mirostat version {self.mirostat}. ignored.', 2)
            if self.top_k > 0:
                self._chain_str += f'top-k {self.top_k} -> '
                lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_top_k(k=self.top_k))
            if self.typical_p != 1.0:
                self._chain_str += f'typical-p {self.typical_p:.2f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_typical(p=self.typical_p, min_keep=1)
                )
            if self.top_p < 1.0:
                self._chain_str += f'top-p {self.top_p:.2f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_top_p(p=self.top_p, min_keep=1)
                )
            if self.min_p > 0.0:
                self._chain_str += f'min-p {self.min_p:.3f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_min_p(p=self.min_p, min_keep=1)
                )
            if self.dynatemp_delta > 0.0:
                # dynamic temperature AKA entropy sampling
                self._chain_str += f'temp {self.temp:.2f} +/- {self.dynatemp_delta:.2f} -> '
                lib.llama_sampler_chain_add(
                    smpl, lib.llama_sampler_init_temp_ext(
                        t=self.temp,
                        delta=self.dynatemp_delta,
                        exponent=self.dynatemp_exponent
                    )
                )
            else:
                # standard temperature
                self._chain_str += f'temp {self.temp:.2f} -> '
                lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_temp(t=self.temp))
        
            self._chain_str += 'dist'
            lib.llama_sampler_chain_add(
                smpl, lib.llama_sampler_init_dist(seed=self.seed if self.seed > 0 else _get_random_seed())
            )
        
        self.smpl = smpl
    
    def __del__(self):
        self.free()
    
    def __repr__(self) -> str:
        return (
            f"SamplerParams("
            f"llama=<Llama instance>, "
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
            f"dry_multiplier={self.dry_multiplier}, "
            f"dry_base={self.dry_base}, "
            f"dry_allowed_length={self.dry_allowed_length}, "
            f"dry_penalty_last_n={self.dry_penalty_last_n}, "
            f"top_n_sigma={self.top_n_sigma}, "
            f"mirostat={self.mirostat}, "
            f"mirostat_tau={self.mirostat_tau}, "
            f"mirostat_eta={self.mirostat_eta}, "
            f"dry_sequence_breakers={self.dry_sequence_breakers!r}, "
            f"logit_bias={self.logit_bias!r}"
            f")"
        )
    
    def print_chain(self) -> None:
        """Print the active sampler chain."""
        log(f'sampler chain: {self._chain_str}')
    
    def free(self) -> None:
        if self.smpl is not None:
            lib.llama_sampler_free(self.smpl)
            self.smpl = None
    
    def reset(self) -> None:
        null_ptr_check(self.smpl, 'self.smpl', 'SamplerParams.reset')
        lib.llama_sampler_reset(self.smpl)

    def to_dict(self) -> dict:
        """Return the sampler parameters as a dictionary."""
        return {
            # not including "llama"
            "seed"                  : self.seed,
            "top_k"                 : self.top_k,
            "top_p"                 : self.top_p,
            "min_p"                 : self.min_p,
            "xtc_probability"       : self.xtc_probability,
            "xtc_threshold"         : self.xtc_threshold,
            "typical_p"             : self.typical_p,
            "temp"                  : self.temp,
            "dynatemp_delta"        : self.dynatemp_delta,
            "dynatemp_exponent"     : self.dynatemp_exponent,
            "penalty_last_n"        : self.penalty_last_n,
            "penalty_repeat"        : self.penalty_repeat,
            "penalty_freq"          : self.penalty_freq,
            "penalty_present"       : self.penalty_present,
            "dry_multiplier"        : self.dry_multiplier,
            "dry_base"              : self.dry_base,
            "dry_allowed_length"    : self.dry_allowed_length,
            "dry_penalty_last_n"    : self.dry_penalty_last_n,
            "mirostat"              : self.mirostat,
            "top_n_sigma"           : self.top_n_sigma,
            "mirostat_tau"          : self.mirostat_tau,
            "mirostat_eta"          : self.mirostat_eta,
            "dry_sequence_breakers" : self.dry_sequence_breakers,
            "logit_bias"            : self.logit_bias
        }

    @classmethod
    def from_dict(cls, llama: Llama, params_dict: dict):
        """Creates a SamplerParams instance from a dictionary.

        Args:
            llama: The Llama object associated with these parameters.
            params_dict: A dictionary containing the sampler parameters.

        Returns:
            A new SamplerParams instance.
        """
        # Create a copy to avoid modifying the original dictionary
        # and remove keys that are not constructor arguments (like 'llama' if present)
        filtered_params = {
            k: v for k, v in params_dict.items()
            if k in SamplerPreset.__init__.__code__.co_varnames
        }
        return cls(llama=llama, **filtered_params)

# TODO ensure the below class exactly mirrors llama.cpp names / styling / structure

class CustomSamplerChain:
    """Allows for the creation of arbitrary custom sampler chains.

    Unlike `SamplerParams`, which constructs a sampler chain based on a fixed set of logic,
    this class allows the user to add samplers one by one in any order.

    Each `add_*` method adds a new sampler to the chain and returns `self`, allowing for
    method chaining. This object can then be used in place of a `SamplerParams` object."""
    def __init__(self, llama: Llama):
        """Initializes an empty custom sampler chain.

        Args:
            llama: The Llama instance this sampler chain will be associated with."""
        self.llama = llama
        self.smpl = None
        self._chain_str = ''

        sparams = lib.llama_sampler_chain_default_params()
        null_ptr_check(sparams, 'sparams', 'CustomSamplerChain.__init__')

        smpl = lib.llama_sampler_chain_init(sparams)
        null_ptr_check(smpl, 'smpl', 'CustomSamplerChain.__init__')

        self.smpl = smpl

    def __del__(self):
        self.free()

    def free(self) -> None:
        if self.smpl is not None:
            lib.llama_sampler_free(self.smpl)
            self.smpl = None

    def reset(self) -> None:
        """Resets the state of the samplers in the chain."""
        null_ptr_check(self.smpl, 'self.smpl', 'CustomSamplerChain.reset')
        lib.llama_sampler_reset(self.smpl)

    def print_chain(self) -> None:
        """Print the active sampler chain as it has been constructed."""
        log(f'sampler chain: {self._chain_str}')

    def _add_to_chain_str(self, s: str):
        if self._chain_str:
            self._chain_str += ' -> '
        self._chain_str += s

    #
    # The methods below correspond to the different types of samplers
    # that can be added to the chain.
    #

    def add_logit_bias(self, logit_bias: Optional[dict[int, float]]) -> 'CustomSamplerChain':
        """Adds a logit bias sampler."""
        if logit_bias is not None and len(logit_bias) > 0:
            if len(logit_bias) == 1:
                self._add_to_chain_str(f'one logit bias')
            else:
                self._add_to_chain_str(f'{len(logit_bias)} logit biases')
            logit_bias_arr = _internals.get_logit_bias_array(logit_bias)
            lib.llama_sampler_chain_add(self.smpl, lib.llama_sampler_init_logit_bias(
                n_vocab=self.llama._n_vocab,
                n_logit_bias=len(logit_bias),
                logit_bias=logit_bias_arr
            ))
        return self

    def add_penalties(
        self,
        penalty_last_n: Optional[int] = None,
        penalty_repeat: Optional[float] = None,
        penalty_freq: Optional[float] = None,
        penalty_present: Optional[float] = None
    ) -> 'CustomSamplerChain':
        """Adds a penalties sampler."""
        _penalty_last_n = penalty_last_n if penalty_last_n is not None else (
            _ParamDefaults.penalty_last_n
        )
        _penalty_repeat = penalty_repeat if penalty_repeat is not None else (
            _ParamDefaults.penalty_repeat
        )
        _penalty_freq = penalty_freq if penalty_freq is not None else (
            _ParamDefaults.penalty_freq
        )
        _penalty_present = penalty_present if penalty_present is not None else (
            _ParamDefaults.penalty_present
        )

        final_last_n = _penalty_last_n if _penalty_last_n >= 0 else self.llama.n_ctx()

        if final_last_n != 0 and any(
            [_penalty_repeat != 1.0, _penalty_present != 0.0, _penalty_freq != 0.0]
        ):
            chain_str = f'penalty last:{final_last_n}'
            if _penalty_repeat != 1.0: chain_str += f' rept:{_penalty_repeat:.3f}'
            if _penalty_present != 0.0: chain_str += f' pres:{_penalty_present:.3f}'
            if _penalty_freq != 0.0: chain_str += f' freq:{_penalty_freq:.3f}'
            self._add_to_chain_str(chain_str)

            lib.llama_sampler_chain_add(self.smpl, lib.llama_sampler_init_penalties(
                penalty_last_n=final_last_n,
                penalty_repeat=_penalty_repeat,
                penalty_freq=_penalty_freq,
                penalty_present=_penalty_present
            ))
        return self

    def add_xtc(
        self,
        p: Optional[float] = None,
        t: Optional[float] = None,
        seed: Optional[int] = None
    ) -> 'CustomSamplerChain':
        """Adds an XTC sampler."""
        _p = p if p is not None else _ParamDefaults.xtc_probability
        _t = t if t is not None else _ParamDefaults.xtc_threshold
        _seed = seed if seed is not None else _ParamDefaults.seed

        if _p > 0.0:
            self._add_to_chain_str(f'XTC p:{_p:.2f} t:{_t:.2f}')
            lib.llama_sampler_chain_add(
                self.smpl, lib.llama_sampler_init_xtc(
                    p=_p,
                    t=_t,
                    min_keep=1,
                    seed=_seed if _seed > 0 else _get_random_seed()
                )
            )
        return self

    def add_dry(
        self,
        multiplier: Optional[float] = None,
        base: Optional[float] = None,
        allowed_length: Optional[int] = None,
        penalty_last_n: Optional[int] = None,
        sequence_breakers: Optional[list[str]] = None
    ) -> 'CustomSamplerChain':
        """Adds a DRY sampler."""
        _multiplier = multiplier if multiplier is not None else _ParamDefaults.dry_multiplier
        _base = base if base is not None else _ParamDefaults.dry_base
        _allowed_length = allowed_length if allowed_length is not None else (
            _ParamDefaults.dry_allowed_length
        )
        _penalty_last_n = penalty_last_n if penalty_last_n is not None else (
            _ParamDefaults.dry_penalty_last_n
        )
        _seq_breakers = sequence_breakers if sequence_breakers is not None else (
            _ParamDefaults.dry_sequence_breakers
        )

        if _multiplier > 0.0:
            self._add_to_chain_str(
                f'DRY x{_multiplier:.2f} base:{_base:.2f} len:{_allowed_length}'
            )
            null_ptr_check(self.llama._vocab, 'llama._vocab', 'CustomSamplerChain.add_dry')
            seq_breakers_bytes = [ez_encode(s) for s in _seq_breakers]
            arr = (ctypes.c_char_p * len(seq_breakers_bytes))(*seq_breakers_bytes)
            lib.llama_sampler_chain_add(self.smpl, lib.llama_sampler_init_dry(
                vocab=self.llama._vocab,
                n_ctx_train=self.llama._n_ctx_train,
                dry_multiplier=_multiplier,
                dry_base=_base,
                dry_allowed_length=_allowed_length,
                dry_penalty_last_n=_penalty_last_n,
                seq_breakers=arr,
                num_breakers=len(_seq_breakers)
            ))
        return self

    def add_greedy(self) -> 'CustomSamplerChain':
        """Adds a greedy sampler (always picks the most likely token)."""
        self._add_to_chain_str('greedy')
        lib.llama_sampler_chain_add(self.smpl, lib.llama_sampler_init_greedy())
        return self

    def add_top_k(self, k: Optional[int] = None) -> 'CustomSamplerChain':
        """Adds a top-k sampler."""
        _k = k if k is not None else _ParamDefaults.top_k
        if _k > 0:
            self._add_to_chain_str(f'top-k {_k}')
            lib.llama_sampler_chain_add(self.smpl, lib.llama_sampler_init_top_k(k=_k))
        return self

    def add_typical_p(self, p: Optional[float] = None) -> 'CustomSamplerChain':
        """Adds a typical-p sampler."""
        _p = p if p is not None else _ParamDefaults.typical_p
        if _p < 1.0:
            self._add_to_chain_str(f'typical-p {_p:.2f}')
            lib.llama_sampler_chain_add(
                self.smpl, lib.llama_sampler_init_typical(p=_p, min_keep=1)
            )
        return self

    def add_top_p(self, p: Optional[float] = None) -> 'CustomSamplerChain':
        """Adds a top-p sampler."""
        _p = p if p is not None else _ParamDefaults.top_p
        if _p < 1.0:
            self._add_to_chain_str(f'top-p {_p:.2f}')
            lib.llama_sampler_chain_add(
                self.smpl, lib.llama_sampler_init_top_p(p=_p, min_keep=1)
            )
        return self

    def add_min_p(self, p: Optional[float] = None) -> 'CustomSamplerChain':
        """Adds a min-p sampler."""
        _p = p if p is not None else _ParamDefaults.min_p
        if _p > 0.0:
            self._add_to_chain_str(f'min-p {_p:.3f}')
            lib.llama_sampler_chain_add(
                self.smpl, lib.llama_sampler_init_min_p(p=_p, min_keep=1)
            )
        return self
    
    def add_top_n_sigma(self, n: Optional[float] = None) -> 'CustomSamplerChain':
        """Adds a top-n-sigma sampler."""
        _n = n if n is not None else _ParamDefaults.top_n_sigma
        if _n > 0.0:
            self._add_to_chain_str(f'top-n-sigma {_n:.2f}')
            lib.llama_sampler_chain_add(self.smpl, lib.llama_sampler_init_top_n_sigma(n=_n))
        return self

    def add_temp(self, t: Optional[float] = None) -> 'CustomSamplerChain':
        """Adds a temperature sampler."""
        _t = t if t is not None else _ParamDefaults.temp
        self._add_to_chain_str(f'temp {_t:.2f}')
        lib.llama_sampler_chain_add(self.smpl, lib.llama_sampler_init_temp(t=_t))
        return self

    def add_temp_ext(
        self,
        t: Optional[float] = None,
        delta: Optional[float] = None,
        exponent: Optional[float] = None
    ) -> 'CustomSamplerChain':
        """Adds a dynamic temperature (entropy) sampler."""
        _t = t if t is not None else _ParamDefaults.temp
        _delta = delta if delta is not None else _ParamDefaults.dynatemp_delta
        _exponent = exponent if exponent is not None else _ParamDefaults.dynatemp_exponent

        self._add_to_chain_str(f'temp {_t:.2f} +/- {abs(_delta):.2f}')
        lib.llama_sampler_chain_add(
            self.smpl, lib.llama_sampler_init_temp_ext(
                t=_t,
                delta=abs(_delta),
                exponent=_exponent
            )
        )
        return self

    def add_mirostat_v1(
        self,
        tau: Optional[float] = None,
        eta: Optional[float] = None,
        seed: Optional[int] = None
    ) -> 'CustomSamplerChain':
        """Adds a mirostat v1 sampler."""
        _tau = tau if tau is not None else _ParamDefaults.mirostat_tau
        _eta = eta if eta is not None else _ParamDefaults.mirostat_eta
        _seed = seed if seed is not None else _ParamDefaults.seed

        self._add_to_chain_str(f'mirostat v1 tau:{_tau:.2f} eta:{_eta:.2f}')
        lib.llama_sampler_chain_add(
            self.smpl, lib.llama_sampler_init_mirostat(
                seed=_seed if _seed > 0 else _get_random_seed(),
                tau=_tau,
                eta=_eta
            )
        )
        return self

    def add_mirostat_v2(
        self,
        tau: Optional[float] = None,
        eta: Optional[float] = None,
        seed: Optional[int] = None
    ) -> 'CustomSamplerChain':
        """Adds a mirostat v2 sampler."""
        _tau = tau if tau is not None else _ParamDefaults.mirostat_tau
        _eta = eta if eta is not None else _ParamDefaults.mirostat_eta
        _seed = seed if seed is not None else _ParamDefaults.seed

        self._add_to_chain_str(f'mirostat v2 tau:{_tau:.2f} eta:{_eta:.2f}')
        lib.llama_sampler_chain_add(
            self.smpl, lib.llama_sampler_init_mirostat_v2(
                seed=_seed if _seed > 0 else _get_random_seed(),
                tau=_tau,
                eta=_eta
            )
        )
        return self

    def add_dist(self, seed: Optional[int] = None) -> 'CustomSamplerChain':
        """Adds a distribution sampler, which performs the random sampling from the modified
        logit distribution. This should usually be the last sampler in the chain."""
        _seed = seed if seed is not None else _ParamDefaults.seed
        self._add_to_chain_str('dist')
        lib.llama_sampler_chain_add(
            self.smpl, lib.llama_sampler_init_dist(
                seed=_seed if _seed > 0 else _get_random_seed()
        ))
        return self


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

        seed:               Optional[int]   = None, # random: <= 0
        top_k:              Optional[int]   = None, # neutral: <= 0
        top_p:              Optional[float] = None, # neutral: 1.0
        min_p:              Optional[float] = None, # neutral: 0.0
        xtc_probability:    Optional[float] = None, # neutral: 0.0
        xtc_threshold:      Optional[float] = None, # disable: > 0.5
        typical_p:          Optional[float] = None, # neutral: 1.0
        temp:               Optional[float] = None, # neutral: 1.0, greedy: <= 0.0
        dynatemp_delta:     Optional[float] = None, # neutral: <= 0.0
        dynatemp_exponent:  Optional[float] = None, # controls how entropy maps to dynamic temperature
        penalty_last_n:     Optional[int]   = None, # disable: 0, n_ctx: -1, last n tokens to penalize
        penalty_repeat:     Optional[float] = None, # neutral: 1.0, should be between 1.0 and ~1.1; values < 1.0 will INCREASE repetition
        penalty_freq:       Optional[float] = None, # neutral: 0.0
        penalty_present:    Optional[float] = None, # neutral: 0.0
        dry_multiplier:     Optional[float] = None, # disable: 0.0, DRY repetition penalty for tokens extending repetition:
        dry_base:           Optional[float] = None, # disable: 0.0, multiplier * base ^ (length of sequence before token - allowed length)
        dry_allowed_length: Optional[int]   = None, # tokens extending repetitions beyond this receive penalty
        dry_penalty_last_n: Optional[int]   = None, # disable: 0, n_ctx: -1, how many tokens to scan for repetitions
        mirostat:           Optional[int]   = None, # disable: 0, use v1: 1, use v2: 2
        top_n_sigma:        Optional[float] = None, # disable: -1.0
        mirostat_tau:       Optional[float] = None, # target entropy for mirostat
        mirostat_eta:       Optional[float] = None, # learning rate for mirostat

        dry_sequence_breakers: Optional[list[str]] = None, # sequence breakers for DRY

        # TODO: grammar goes here

        logit_bias: Optional[dict[int, float]] = None # dictionary of one or more {tok_id: bias}
    ):
        self.seed = seed if seed is not None else _ParamDefaults.seed

        self.top_k              = top_k if top_k is not None else _ParamDefaults.top_k
        self.top_p              = top_p if top_p is not None else _ParamDefaults.top_p
        self.min_p              = min_p if min_p is not None else _ParamDefaults.min_p
        self.xtc_probability    = xtc_probability if xtc_probability is not None else (
            _ParamDefaults.xtc_probability
        )
        self.xtc_threshold      = xtc_threshold if xtc_threshold is not None else (
            _ParamDefaults.xtc_threshold
        )
        self.typical_p          = typical_p if typical_p is not None else (
            _ParamDefaults.typical_p
        )
        self.temp               = temp if temp is not None else _ParamDefaults.temp
        self.dynatemp_delta     = dynatemp_delta if dynatemp_delta is not None else (
            _ParamDefaults.dynatemp_delta
        )
        self.dynatemp_exponent  = dynatemp_exponent if dynatemp_exponent is not None else (
            _ParamDefaults.dynatemp_exponent
        )
        self.penalty_last_n     = penalty_last_n if penalty_last_n is not None else (
            _ParamDefaults.penalty_last_n
        )
        self.penalty_repeat     = penalty_repeat if penalty_repeat is not None else (
            _ParamDefaults.penalty_repeat
        )
        self.penalty_freq       = penalty_freq if penalty_freq is not None else (
            _ParamDefaults.penalty_freq
        )
        self.penalty_present    = penalty_present if penalty_present is not None else (
            _ParamDefaults.penalty_present
        )
        self.dry_multiplier     = dry_multiplier if dry_multiplier is not None else (
            _ParamDefaults.dry_multiplier
        )
        self.dry_base           = dry_base if dry_base is not None else _ParamDefaults.dry_base
        self.dry_allowed_length = dry_allowed_length if dry_allowed_length is not None else (
            _ParamDefaults.dry_allowed_length
        )
        self.dry_penalty_last_n = dry_penalty_last_n if dry_penalty_last_n is not None else (
            _ParamDefaults.dry_penalty_last_n
        )
        self.mirostat           = mirostat if mirostat is not None else _ParamDefaults.mirostat
        self.top_n_sigma        = top_n_sigma if top_n_sigma is not None else (
            _ParamDefaults.top_n_sigma
        )
        self.mirostat_tau       = mirostat_tau if mirostat_tau is not None else (
            _ParamDefaults.mirostat_tau
        )
        self.mirostat_eta       = mirostat_eta if mirostat_eta is not None else (
            _ParamDefaults.mirostat_eta
        )
        
        self.dry_sequence_breakers = (
            dry_sequence_breakers if dry_sequence_breakers is not None else
            _ParamDefaults.dry_sequence_breakers
        )

        self.logit_bias = logit_bias if logit_bias is not None else _ParamDefaults.logit_bias

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
            f"dry_multiplier={self.dry_multiplier}, "
            f"dry_base={self.dry_base}, "
            f"dry_allowed_length={self.dry_allowed_length}, "
            f"dry_penalty_last_n={self.dry_penalty_last_n}, "
            f"mirostat={self.mirostat}, "
            f"top_n_sigma={self.top_n_sigma}, "
            f"mirostat_tau={self.mirostat_tau}, "
            f"mirostat_eta={self.mirostat_eta}, "
            f"dry_sequence_breakers={self.dry_sequence_breakers!r}, "
            f"logit_bias={self.logit_bias!r}"
            f")"
        )

    def as_dict(self) -> dict:
        """Returns the sampler parameters as a dictionary."""
        return {
            "seed"                  : self.seed,
            "top_k"                 : self.top_k,
            "top_p"                 : self.top_p,
            "min_p"                 : self.min_p,
            "xtc_probability"       : self.xtc_probability,
            "xtc_threshold"         : self.xtc_threshold,
            "typical_p"             : self.typical_p,
            "temp"                  : self.temp,
            "dynatemp_delta"        : self.dynatemp_delta,
            "dynatemp_exponent"     : self.dynatemp_exponent,
            "penalty_last_n"        : self.penalty_last_n,
            "penalty_repeat"        : self.penalty_repeat,
            "penalty_freq"          : self.penalty_freq,
            "penalty_present"       : self.penalty_present,
            "dry_multiplier"        : self.dry_multiplier,
            "dry_base"              : self.dry_base,
            "dry_allowed_length"    : self.dry_allowed_length,
            "dry_penalty_last_n"    : self.dry_penalty_last_n,
            "mirostat"              : self.mirostat,
            "top_n_sigma"           : self.top_n_sigma,
            "mirostat_tau"          : self.mirostat_tau,
            "mirostat_eta"          : self.mirostat_eta,
            "dry_sequence_breakers" : self.dry_sequence_breakers,
            "logit_bias"            : self.logit_bias
        }

    @classmethod
    def from_dict(cls, params_dict: dict):
        """Creates a SamplerPreset instance from a dictionary.

        Args:
            params_dict: A dictionary containing the sampler parameters.

        Returns:
            A new SamplerPreset instance.
        """
        # Create a copy to avoid modifying the original dictionary
        # and remove keys that are not constructor arguments
        filtered_params = {k: v for k, v in params_dict.items() if k in cls.__init__.__code__.co_varnames}
        return cls(**filtered_params)


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
    """The default easy-llama sampler preset"""

    LlamaCPP = SamplerPreset(
        top_k = 40,
        top_p = 0.95,
        min_p = 0.05,
        temp = 0.8
    )
    """The default llama.cpp sampler preset"""

    Cool = SamplerPreset(
        top_k = 32,
        top_p = 0.95,
        min_p = 0.05,
        temp = 0.3
    )
    """The recommended easy-llama sampler preset for predictable output"""

    Warm = SamplerPreset(
        top_k = 64,
        top_p = 0.95,
        min_p = 0.05,
        temp = 1.25
    )
    """The recommended easy-llama sampler preset for creative yet coherent output"""

    Neutral = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.0,
        temp = 1.0
    )
    """All samplers neutralized"""

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

    TopNSigma = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.0,
        temp = 1.0,
        top_n_sigma = 1.0
    )
    """Top-nσ as described on [arXiv](https://arxiv.org/pdf/2411.07641) and
    in [llama.cpp#11223](https://github.com/ggml-org/llama.cpp/pull/11223)"""

    TopNSigmaRandom = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.0,
        temp = 9999.9,
        top_n_sigma = 1.0
    )
    """Top-nσ as described on [arXiv](https://arxiv.org/pdf/2411.07641) and
    in [llama.cpp#11223](https://github.com/ggml-org/llama.cpp/pull/11223), except that
    `temp = 9999.9` to randomly select any token that is determined to be valid
    by Top-nσ."""

    DRY = SamplerPreset(dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2)
    """https://github.com/oobabooga/text-generation-webui/pull/5677"""

    XTC = SamplerPreset(
        top_k = -1,
        top_p = 1.0,
        min_p = 0.02,
        xtc_probability = 0.5,
        xtc_threshold = 0.1
    )
    """https://github.com/oobabooga/text-generation-webui/pull/6335"""

    #
    # Samplers below this line are for specific models / model families
    #

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
        temp = 1.2
    )
    """Llama3 preset with increased temperature (more random)"""

    Mistral = SamplerPreset(
        temp = 0.3
    )
    """Mistral models tend to require a lower temperature"""

    Magistral = SamplerPreset(
        top_k = -1,
        top_p = 0.95,
        min_p = 0.0,
        temp = 0.7
    )
    """[mistralai/Magistral-Small-2506](https://huggingface.co/mistralai/Magistral-Small-2506)"""

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
        temp = 1.1
    )
    """[Qwen/Qwen2.5-14B-Instruct/](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/) 
    (unofficial, but recommended)"""

    Qwen3Thinking = SamplerPreset(
        top_k = 20,
        top_p = 0.95,
        min_p = 0.0,
        temp = 0.6
    )
    """[Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/README.md)"""

    Qwen3NoThinking = SamplerPreset(
        top_k = 20,
        top_p = 0.8,
        min_p = 0.0,
        temp = 0.7
    )
    """[Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/README.md)"""
