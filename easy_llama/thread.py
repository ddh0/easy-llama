# thread.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""This file provides functionality for multi-turn conversations with Llama models."""

import sys
import jinja2

from typing    import Optional
from .sampling import SamplerPreset
from .utils    import (
    _SupportsWriteAndFlush, ANSI, log, ez_encode, ez_decode, suppress_output,
    KeyboardInterruptHandler
)

from . import llama as _llama # avoid confusion with Thread.llama attribute


class Thread:

    valid_system_roles = ['system', 'developer'                         ]
    valid_user_roles   = ['user',   'human'                             ]
    valid_bot_roles    = ['bot',    'assistant', 'model', 'gpt', 'llama']

    all_valid_roles = valid_system_roles + valid_user_roles + valid_bot_roles

    def __init__(
        self,
        llama: _llama.Llama,
        sampler_preset: Optional[SamplerPreset] = None
    ) -> None:
        
        if not isinstance(llama, _llama.Llama):
            raise TypeError(
                f'Thread.__init__: llama must be an instance of llama.Llama, not {type(llama)}'
            )
        
        llama._validate_model_state()

        self.llama = llama
        self.sampler_preset = sampler_preset if isinstance(sampler_preset, SamplerPreset) else (
            SamplerPreset()
        )
        
        self.messages: list[dict[str, str]] = []
        self._initial_messages = self.messages.copy()
        self._stop_tokens = self.llama.eog_tokens
    
    def __repr__(self) -> str:
        return (
            f"Thread("
            f"llama={self.llama!r}, "
            f"sampler_preset={self.sampler_preset!r}, "
            f"messages={self.messages!r}"
            f")"
        )

    def _render_msgs(
        self,
        add_generation_prompt: bool = True,
        **kwargs
    ) -> str:
        """Render the chat template template using the messages in this Thread"""
        
        jinja_messages = []
        
        #
        # convert all messages in the thread into the format expected by the chat template
        #

        for message in self.messages:
            try:
                role = message['role']
            except KeyError:
                log(f'_render_msgs: skipping message with no role!', 2)
                continue
            try:
                content = message['content']
            except KeyError:
                log(f'_render_msgs: skipping message with no content!', 2)
                continue

            role = role.lower()
            if len(content) == 0:
                log(f'_render_msgs: {role} message content is empty', 2)

            if role in Thread.valid_system_roles:
                jinja_messages.append({'role': 'system', 'content': content})
            elif role in Thread.valid_user_roles:
                jinja_messages.append({'role': 'user', 'content': content})
            elif role in Thread.valid_bot_roles:
                jinja_messages.append({'role': 'assistant', 'content': content})
            else:
                log(f'_render_msgs: skipping message with invalid role {role!r}!', 2)
        
        #
        # get the chat template string from the model's metadata
        #

        chat_template_str = self.llama.chat_template()
        if chat_template_str is None:
            exc_str = (
                f"_render_msgs: unable to render messages becuase the model has no chat "
                f"template! re-convert your model or download a newer version."
            )
            log(exc_str, 3)
            raise ValueError(exc_str)

        #
        # convert the template string to an actual jinja2 template
        #

        try:
            template = jinja2.Template(chat_template_str, undefined=jinja2.StrictUndefined)
        except Exception as e:
            exc_str = f"_render_msgs: error creating chat template: {type(e).__name__}: {e}"
            log(exc_str, 3)
            raise ValueError(exc_str) from e
        
        #
        # set the context data used when rendering the template
        #
        
        context = {
            'messages': jinja_messages,
            'add_generation_prompt': add_generation_prompt,
            **kwargs
        }

        #
        # finally, render the chat template and return the resulting string
        #

        try:
            return template.render(context)
        except Exception as e:
            exc_str = (
                f"_render_msgs: error rendering chat template with content: "
                f"{type(e).__name__}: {e}"
            )
            log(exc_str, 3)
            raise ValueError(exc_str) from e

    def get_input_ids(self, add_generation_prompt: bool = True, **kwargs) -> list[int]:
        """Get a list of token IDs in this thread, to be used for inference

        - add_generation_prompt:
            Whether or not to include the bot prefix tokens at the end of the input IDs.
            
        Any additional kwargs are passed as context for rendering the chat template."""

        # render the messages using the provided context
        thread_chat_string = self._render_msgs(add_generation_prompt, **kwargs)

        # tokenize the rendered chat template
        input_ids = self.llama.tokenize(
            text_bytes=ez_encode(thread_chat_string),
            add_special=True,
            parse_special=True
        )
        
        return input_ids
    
    def send(self, content: str, n_predict: Optional[int] = None) -> str:
        """Send a message in this thread and return the generated response. This adds your
        message and the bot's message to the thread."""
        self.messages.append({
            'role': 'user',
            'content': content
        })
        response_toks = self.llama.generate(
            input_tokens=self.get_input_ids(add_generation_prompt=True),
            n_predict=n_predict if n_predict is not None else -1,
            stop_tokens=self._stop_tokens,
            sampler_preset=self.sampler_preset
        )
        response_txt = self.llama.detokenize(response_toks, special=False)
        self.messages.append({
            'role': 'bot',
            'content': response_txt
        })
        return response_txt

    def as_string(self, **kwargs) -> str:
        """Return this thread's message history as a string"""
        return self._render_msgs(add_generation_prompt=False, **kwargs)
    
    def add_message(self, role: str, content: str) -> None:
        """Append a message to `Thread.messages` with the specified role and content

        - role:
            The role of the message, for example 'system', 'user', or 'bot'.
        - content:
            The text content of the message."""
        if role.lower() in Thread.valid_system_roles:
            self.messages.append({'role': 'system', 'content': content})
        elif role.lower() in Thread.valid_user_roles:
            self.messages.append({'role': 'user', 'content': content})
        elif role.lower() in Thread.valid_bot_roles:
            self.messages.append({'role': 'bot', 'content': content})
        else:
            raise ValueError(f'Thread.add_message: invalid role {role!r}')
    
    def warmup(self) -> None:
        input_ids = self.get_input_ids()
        if self.llama._first_valid_pos(input_ids) < len(input_ids):
            _llama.log_verbose('Thread.warmup: processing thread content with model ...')
            with suppress_output(disable=_llama.get_verbose()):
                self.llama.generate(input_tokens=input_ids, n_predict=0)
        
        # if the above condition is not True, the thread is already in the cache, so
        # nothing needs to be done
        _llama.log_verbose('Thread.warmup: done')
    
    def interact(self, stream: bool = True) -> None:
        """Start an interactive terminal-based chat using this thread"""
        R = ANSI.MODE_RESET_ALL
        B = ANSI.FG_BRIGHT_CYAN
        G = ANSI.FG_BRIGHT_GREEN
        with KeyboardInterruptHandler():
            print()
            while True:
                user_input = input(f'{R}  > {G}')
                print(R, end='\n', flush=True)
                
                if stream:
                    self.messages.append({'role': 'user', 'content': user_input})
                    input_ids = self.get_input_ids()
                    
                    tok_gen = self.llama.stream(
                        input_tokens=input_ids,
                        n_predict=-1,
                        stop_tokens=self._stop_tokens,
                        sampler_preset=self.sampler_preset
                    )
                    
                    response_toks = []
                    detok_bytes_buffer = b''
                    
                    for tok in tok_gen:
                        response_toks.append(tok)
                        #
                        # detok_bytes_buffer holds any incomplete UTF-8 characters until they
                        # are completed by future tokens
                        # 
                        # for example, emojis are often split between two tokens, with one or
                        # both of those tokens not being valid UTF-8 on its own
                        #
                        detok_bytes_buffer += self.llama.token_to_piece(tok, special=False)
                        try:
                            detok_txt = detok_bytes_buffer.decode('utf-8', errors='strict')
                        except UnicodeDecodeError:
                            pass # try again on next token
                        else:
                            detok_bytes_buffer = b''
                            print(f'{B}{detok_txt}{R}', end='', flush=True)
                    
                    # print any leftover bytes (though ideally there should be none)
                    if detok_bytes_buffer != b'':
                        leftover_txt = ez_decode(detok_bytes_buffer)
                        print(f'{B}{leftover_txt}{R}', end='', flush=True)
                    
                    self.messages.append({
                        'role': 'bot',
                        'content': self.llama.detokenize(response_toks, special=False)
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
        return summary

    def print_stats(self, file: Optional[_SupportsWriteAndFlush] = None) -> None:
        """Print stats about the context usage in this thread"""
        _file = sys.stdout if file is None else file
        with suppress_output():
            input_ids = self.get_input_ids(add_generation_prompt=False)
        n_thread_tokens = len(input_ids)
        n_msgs = len(self.messages)
        n_ctx = self.llama._n_ctx
        c = (n_thread_tokens/n_ctx) * 100
        ctx_used_pct = int(c) + (c > int(c)) # round up to next integer
        print(f"{n_thread_tokens} / {n_ctx} tokens", file=_file)
        print(f"{ctx_used_pct}% of context used", file=_file)
        print(f"{n_msgs} messages", file=_file)
    
    def reset(self) -> None:
        self.messages: list[dict[str, str]] = []
