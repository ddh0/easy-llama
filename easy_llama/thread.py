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
            _params: SamplerParams = llama._default_sampler_params
        else:
            assert_type(sampler_params, SamplerParams, 'sampler_params', 'Thread.__init__')
            _params = sampler_params
        
        llama._validate_model_state()

        self.llama = llama
        self.prompt_format = prompt_format
        self.sampler_params = _params

        self.messages: list[dict[str, str]] = []

        system_prompt = prompt_format.system_prompt()

        if system_prompt != '':
            self.messages.append(
                {
                    'role': 'system',
                    'content': system_prompt
                }
            )
    
    def get_input_ids(self, role: Optional[str] = 'bot') -> list[int]:
        input_ids = []
        if len(self.messages) > 0:
            # the first message requires `add_special=True` in order to set
            # the BOS token correctly
            first_msg = self.messages[0]
            if first_msg['role'].lower() == 'system':
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.system_prefix().encode(),
                    add_special=True,
                    parse_special=True
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.system_prompt().encode(),
                    add_special=False,
                    parse_special=False
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.system_suffix().encode(),
                    add_special=False,
                    parse_special=True
                ))
            elif first_msg['role'].lower() == 'user':
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.user_prefix().encode(),
                    add_special=True,
                    parse_special=True
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=first_msg['content'].encode(),
                    add_special=False,
                    parse_special=False
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.user_suffix().encode(),
                    add_special=False,
                    parse_special=True
                ))
            elif first_msg['role'].lower() == 'bot':
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.bot_prefix().encode(),
                    add_special=True,
                    parse_special=True
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=first_msg['content'].encode(),
                    add_special=False,
                    parse_special=False
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.bot_suffix().encode(),
                    add_special=False,
                    parse_special=True
                ))
            else:
                raise ValueError(
                    f'Thread.get_input_ids: first message has invalid role '
                    f'{role!r}'
                )
            # all the other messages are treated the same
            i = 0
            for msg in self.messages[1:]:
                i += 1
                if msg['role'].lower() == 'system':
                    raise ValueError(
                        f'Thread.get_input_ids: multiple system messages are '
                        f'not supported'
                    )
                elif msg['role'].lower() == 'user':
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=self.prompt_format.user_prefix().encode(),
                        add_special=False,
                        parse_special=True
                    ))
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=msg['content'].encode(),
                        add_special=False,
                        parse_special=False
                    ))
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=self.prompt_format.user_suffix().encode(),
                        add_special=False,
                        parse_special=True
                    ))
                elif msg['role'].lower() == 'bot':
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=self.prompt_format.bot_prefix().encode(),
                        add_special=False,
                        parse_special=True
                    ))
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=msg['content'].encode(),
                        add_special=False,
                        parse_special=False
                    ))
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=self.prompt_format.bot_suffix().encode(),
                        add_special=False,
                        parse_special=True
                    ))
                else:
                    raise ValueError(
                        f'Thread.get_input_ids: message {i} has invalid role '
                        f'{role!r}'
                    )
        if role is not None:
            # append the role prefix tokens to the end
            if role.lower() == 'system':
                raise ValueError(
                    f'Thread.get_input_ids: multiple system messages are not '
                    f'supported'
                )
            elif role.lower() == 'user':
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.user_prefix().encode(),
                    add_special=False,
                    parse_special=True
                ))
            elif role.lower() == 'bot':
                input_ids.extend(self.llama.tokenize(
                    text_bytes=self.prompt_format.bot_prefix().encode(),
                    add_special=False,
                    parse_special=True
                ))
            else:
                raise ValueError(f'Thread.get_input_ids: invalid role {role!r}')
        # input_ids is fully constructed
        n_input_ids = len(input_ids)
        print_info(
            f'Thread.get_input_ids: converted {len(self.messages)} messages to '
            f'{n_input_ids} tokens'
        )
        if n_input_ids >= self.llama._n_ctx:
            print_warning(
                f'Thread.get_input_ids: length of input_ids {n_input_ids} '
                f'equals or exceeds the current context length '
                f'{self.llama._n_ctx}'
            )
        return input_ids
    
    def send(self, content: str) -> str:
        """
        Send a message in this thread and return the generated response
        """
        self.messages.append({
            'role': 'user',
            'content': content
        })
        response_toks = self.llama.generate(
            input_tokens=self.get_input_ids(),
            n_predict=-1,
            stop_tokens=self.llama.eog_tokens,
            sampler_params=self.sampler_params
        )
        response_bytes = self.llama.detokenize(response_toks, special=False)
        response_txt = response_bytes.decode()
        self.messages.append({
            'role': 'bot',
            'content': response_txt
        })
        return response_txt
