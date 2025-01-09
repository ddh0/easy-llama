# thread.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import sys
import llama as _llama
import contextlib

from typing   import Optional
from formats  import PromptFormat
from sampling import SamplerParams

from utils import (
    _SupportsWriteAndFlush, Colors, print_verbose, print_info, print_warning,
    assert_type
)

@contextlib.contextmanager
def KeyboardInterruptHandler():
    print('Press CTRL+C to exit ...')
    try:
        yield
    except KeyboardInterrupt:
        print(Colors.RESET, flush=True)

class Thread:

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
                self.messages.append(
                    {
                        'role': 'system',
                        'content': system_prompt
                    }
                )
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
            # (if role is None, no prefix is appended)
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
        response_txt = response_bytes.decode()
        self.messages.append({
            'role': 'bot',
            'content': response_txt
        })
        return response_txt

    def as_string(self) -> str:
        """Return this thread's message history as a string"""
        result_str = ''
        for msg in self.messages:
            if msg['role'].lower() == 'system':
                result_str += ''.join([
                    self.prompt_format.system_prefix(),
                    msg['content'],
                    self.prompt_format.system_suffix()
                ])
            elif msg['role'].lower() == 'user':
                result_str += ''.join([
                    self.prompt_format.user_prefix(),
                    msg['content'],
                    self.prompt_format.user_suffix()
                ])
            elif msg['role'].lower() == 'bot':
                result_str += ''.join([
                    self.prompt_format.bot_prefix(),
                    msg['content'],
                    self.prompt_format.bot_suffix()
                ])
            else:
                raise ValueError(
                    f"Thread.as_string: invalid message role {msg['role']!r}"
                )
        return result_str
    
    def warmup(self) -> None:
        input_ids = self.get_input_ids()
        if self.llama._first_valid_pos(input_ids) < len(input_ids):
            print_info('Thread.warmup: processing thread content with model ...')
            self.llama.generate(
                input_tokens=input_ids,
                n_predict=0
            )
        print_info('Thread.warmup: done')
    
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
                        tok_txt = tok_bytes.decode(errors='ignore')
                        print(f'{B}{tok_txt}{R}', end='', flush=True)
                    self.messages.append({
                        'role': 'bot',
                        'content': response.decode()
                    })
                    print()
                    if not _llama.verbose:
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
                'content': 'Follow the given instructions exactly. Do not add '
                           'any unnecessary information.'
            },
            {
                'role': 'user',
                'content': 'Take a moment to read through the following '
                           'conversation carefully. When you\'re done, write a '
                           'single paragraph that explains all of the most '
                           'relevant details.'
                           f'\n\n```\n{thread_as_string}\n```\n\n'
                           'Now that you\'ve read the above conversation, '
                           'provide a summary in the form of a single '
                           'paragraph.'
            }
        ]
        input_ids = self.get_input_ids(role='bot')
        output_ids = self.llama.generate(
            input_tokens=input_ids,
            n_predict=300
        )
        summary = self.llama.detokenize(output_ids, special=False)
        self.messages = orig_thread_messages.copy()
        return summary.decode()

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
