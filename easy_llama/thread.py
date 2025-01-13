# thread.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import sys
import contextlib

from .utils    import (
    _SupportsWriteAndFlush, Colors, print_warning, assert_type, ez_encode, ez_decode
)
from typing    import Optional
from .formats  import PromptFormat
from .sampling import SamplerParams

from . import llama as _llama

@contextlib.contextmanager
def KeyboardInterruptHandler():
    _llama.print_info_if_verbose('Press CTRL+C to exit')
    try:
        yield
    except KeyboardInterrupt:
        print(Colors.RESET, end='\n\n', flush=True)

class Thread:

    valid_system_roles = ['system', 'developer'         ]
    valid_user_roles   = ['user',   'human'             ]
    valid_bot_roles    = ['bot',    'assistant', 'model']

    all_valid_roles = valid_system_roles + valid_user_roles + valid_bot_roles

    def __init__(
        self,
        llama: _llama.Llama,
        prompt_format: PromptFormat,
        sampler_params: Optional[SamplerParams] = None,
        messages: Optional[list[dict[str, str]]] = None
    ) -> None:
        
        assert_type(llama, _llama.Llama, 'llama', 'Thread.__init__')
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

        if messages is None:
            self.messages: list[dict[str, str]] = []

            system_prompt = prompt_format.system_prompt()

            if system_prompt != '':
                self.messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
        else:
            self.messages = messages
        
        # save the original messages for self.reset()
        self._orig_messages = self.messages.copy()
    
    def __repr__(self) -> str:
        return (
            f"Thread("
            f"llama={self.llama!r}, "
            f"prompt_format={self.prompt_format!r}, "
            f"sampler_params={self.sampler_params!r}, "
            f"messages={self.messages!r}"
            f")"
        )
    
    def get_input_ids(self, role: Optional[str] = 'bot') -> list[int]:
        """
        Get a list of token IDs in this thread, to be used for inference

        - role:
            The role for which inference will be performed (usually 'bot'). Can be 'system',
            'user', 'bot', or None. If None, no role prefix will be appended (this is useful 
            when you just want to get all the tokens in this Thread but are not going to do
            inference).
        """
        input_ids = []
        if len(self.messages) > 0:
            # the prefix of the first message requires `add_special=True` in order to set
            # the BOS token correctly
            first_msg = self.messages[0]
            if first_msg['role'].lower() in Thread.valid_system_roles:
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.system_prefix()),
                    add_special=True,
                    parse_special=True
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.system_prompt()),
                    add_special=False,
                    parse_special=False
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.system_suffix()),
                    add_special=False,
                    parse_special=True
                ))
            elif first_msg['role'].lower() in Thread.valid_user_roles:
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.user_prefix()),
                    add_special=True,
                    parse_special=True
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(first_msg['content']),
                    add_special=False,
                    parse_special=False
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.user_suffix()),
                    add_special=False,
                    parse_special=True
                ))
            elif first_msg['role'].lower() in Thread.valid_bot_roles:
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.bot_prefix()),
                    add_special=True,
                    parse_special=True
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(first_msg['content']),
                    add_special=False,
                    parse_special=False
                ))
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.bot_suffix()),
                    add_special=False,
                    parse_special=True
                ))
            else:
                raise ValueError(
                    f'Thread.get_input_ids: first message has invalid role {role!r}'
                )
            # all the other messages are treated the same
            i = 0
            for msg in self.messages[1:]:
                i += 1
                if msg['role'].lower() in Thread.valid_system_roles:
                    raise ValueError(
                        f'Thread.get_input_ids: multiple system messages are not supported'
                    )
                elif msg['role'].lower() in Thread.valid_user_roles:
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=ez_encode(self.prompt_format.user_prefix()),
                        add_special=False,
                        parse_special=True
                    ))
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=ez_encode(msg['content']),
                        add_special=False,
                        parse_special=False
                    ))
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=ez_encode(self.prompt_format.user_suffix()),
                        add_special=False,
                        parse_special=True
                    ))
                elif msg['role'].lower() in Thread.valid_bot_roles:
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=ez_encode(self.prompt_format.bot_prefix()),
                        add_special=False,
                        parse_special=True
                    ))
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=ez_encode(msg['content']),
                        add_special=False,
                        parse_special=False
                    ))
                    input_ids.extend(self.llama.tokenize(
                        text_bytes=ez_encode(self.prompt_format.bot_suffix()),
                        add_special=False,
                        parse_special=True
                    ))
                else:
                    raise ValueError(
                        f'Thread.get_input_ids: message {i} has invalid role {role!r}'
                    )
        if role is not None:
            # append the role prefix tokens to the end
            # (if role is None, no prefix is appended)
            if role.lower() in Thread.valid_system_roles:
                raise ValueError(
                    f'Thread.get_input_ids: multiple system messages are not '
                    f'supported'
                )
            elif role.lower() in Thread.valid_user_roles:
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.user_prefix()),
                    add_special=False,
                    parse_special=True
                ))
            elif role.lower() in Thread.valid_bot_roles:
                input_ids.extend(self.llama.tokenize(
                    text_bytes=ez_encode(self.prompt_format.bot_prefix()),
                    add_special=False,
                    parse_special=True
                ))
            else:
                raise ValueError(f'Thread.get_input_ids: invalid role {role!r}')
        # input_ids is now fully constructed
        n_input_ids = len(input_ids)
        _llama.print_info_if_verbose(
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
            input_tokens=self.get_input_ids(role='bot'),
            n_predict=-1,
            stop_tokens=self.llama.eog_tokens,
            sampler_params=self.sampler_params
        )
        response_bytes = self.llama.detokenize(response_toks, special=False)
        response_txt = ez_decode(response_bytes)
        self.messages.append({
            'role': 'bot',
            'content': response_txt
        })
        return response_txt

    def as_string(self) -> str:
        """Return this thread's message history as a string"""
        result_str = ''
        for msg in self.messages:
            if msg['role'].lower() in Thread.valid_system_roles:
                result_str += ''.join([
                    self.prompt_format.system_prefix(),
                    msg['content'],
                    self.prompt_format.system_suffix()
                ])
            elif msg['role'].lower() in Thread.valid_user_roles:
                result_str += ''.join([
                    self.prompt_format.user_prefix(),
                    msg['content'],
                    self.prompt_format.user_suffix()
                ])
            elif msg['role'].lower() in Thread.valid_bot_roles:
                result_str += ''.join([
                    self.prompt_format.bot_prefix(),
                    msg['content'],
                    self.prompt_format.bot_suffix()
                ])
            else:
                raise ValueError(f"Thread.as_string: invalid message role {msg['role']!r}")
        return result_str
    
    def add_message(self, role: str, content: str) -> None:
        if role.lower() == 'system':
            self.messages.append({'role': 'system', 'content': content})
        elif role.lower() == 'user':
            self.messages.append({'role': 'user', 'content': content})
        elif role.lower() == 'bot':
            self.messages.append({'role': 'bot', 'content': content})
        else:
            raise ValueError(f'Thread.add_message: invalid role {role!r}')
    
    def warmup(self) -> None:
        input_ids = self.get_input_ids()
        if self.llama._first_valid_pos(input_ids) < len(input_ids):
            _llama.print_info_if_verbose(
                'Thread.warmup: processing thread content with model ...'
            )
            self.llama.generate(input_tokens=input_ids, n_predict=0)
        # if the above condition is not True, the thread is already in the cache, so
        # nothing needs to be done
        _llama.print_info_if_verbose('Thread.warmup: done')
    
    def interact(self, stream: bool = True) -> None:
        R = Colors.RESET
        B = Colors.BLUE
        G = Colors.GREEN
        with KeyboardInterruptHandler():
            print()
            while True:
                user_input = input(f'{R}  > {G}')
                print(R, end='\n', flush=True)
                if stream:
                    self.messages.append({
                        'role': 'user',
                        'content': user_input
                    })
                    input_ids = self.get_input_ids()
                    tok_gen = self.llama.stream(
                        input_tokens=input_ids,
                        n_predict=-1,
                        stop_tokens=self.llama.eog_tokens,
                        sampler_params=self.sampler_params
                    )
                    response = b''
                    for tok in tok_gen:
                        tok_bytes = self.llama.token_to_piece(tok, special=False)
                        response += tok_bytes
                        tok_txt = ez_decode(tok_bytes)
                        print(f'{B}{tok_txt}{R}', end='', flush=True)
                    self.messages.append({
                        'role': 'bot',
                        'content': ez_decode(response)
                    })
                    print()
                    if not _llama.get_verbose():
                        print()
                else:
                    response = self.send(user_input)
                    print(f'\n{B}{response}{R}\n')
    
    def summarize(self) -> str:
        """Generate a summary of this thread"""
        thread_as_string = self.as_string()
        orig_thread_messages = self.messages.copy()
        self.messages = [
            {
                'role': 'system',
                'content': 'Follow the given instructions exactly. Do not add any unnecessary '
                           'information.'
            },
            {
                'role': 'user',
                'content': 'Take a moment to read through the following conversation '
                           'carefully. When you\'re done, write a single paragraph that '
                           'explains all of the most relevant details.'
                           f'\n\n```\n{thread_as_string}\n```\n\n'
                           'Now that you\'ve read the above conversation, provide a summary '
                           'in the form of a single paragraph.'
            }
        ]
        input_ids = self.get_input_ids() # uses the above messages
        output_ids = self.llama.generate(input_tokens=input_ids, n_predict=300)
        summary = self.llama.detokenize(output_ids, special=False)
        self.messages = orig_thread_messages.copy()
        return ez_decode(summary)

    def print_stats(
        self,
        file: _SupportsWriteAndFlush = sys.stderr,
    ) -> None:
        """Print stats about the context usage in this thread"""
        n_thread_tokens = len(self.get_input_ids(role=None))
        n_msgs = len(self.messages)
        n_ctx = self.llama._n_ctx
        c = (n_thread_tokens/n_ctx) * 100
        ctx_used_pct = int(c) + (c > int(c)) # round up to next integer
        print(f"{n_thread_tokens} / {n_ctx} tokens", file=file)
        print(f"{ctx_used_pct}% of context used", file=file)
        print(f"{n_msgs} messages", file=file)
    
    def reset(self) -> None:
        self.messages = self._orig_messages.copy()
