# thread.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from llama    import Llama
from typing   import Optional
from formats  import PromptFormat
from sampling import SamplerParams

from utils import null_ptr_check, print_info, print_warning, print_error, assert_type

class Thread:

    def __init__(
        self,
        llama: Llama,
        prompt_format: PromptFormat,
        sampler_params: Optional[SamplerParams] = None
    ) -> None:
        assert_type(llama, Llama, 'llama', 'Thread.__init__')
        assert_type(prompt_format, PromptFormat, 'prompt_format', 'Thread.__init__')
        if sampler_params is None:
            self.sampler_params = llama._default_sampler_params
        else:
            assert_type(sampler_params, SamplerParams, 'sampler_params', 'Thread.__init__')
        llama._validate_model_state()
        _sampler_params = sampler_params if sampler_params is not None else (
            llama._default_sampler_params
        )
        self.llama = llama
        self.prompt_format = prompt_format
        