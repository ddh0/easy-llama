# thread.py
# https://github.com/ddh0/easy-llama/

"""Submodule containing the Thread class, used for interaction with a Model"""

import sys
import uuid

from .utils    import (
    _SupportsWriteAndFlush,
    UnreachableException,
    print_verbose,
    assert_type,
    RESET_ALL,
    truncate,
    NoneType,
    cls
)

from .samplers import SamplerSettings, print_sampler_settings
from .model    import Model, ExceededContextLengthException
from typing    import Optional, Literal, Callable
from .formats  import AdvancedFormat

from .formats import blank as formats_blank


class Message(dict):
    """
    A dictionary representing a single message within a Thread

    Works just like a normal `dict`, but adds a new method:
    - `.as_string` - Return the full message string

    Messages should have these keys:
    - `role` -  The role of the speaker: 'system', 'user', or 'bot'
    - `prefix` - The text that prefixes the message content
    - `content` - The actual content of the message
    - `suffix` - The text that suffixes the message content
    """

    def __repr__(self) -> str:
        return \
            "Message({" \
            f"'role': {self['role']!r}, " \
            f"'prefix': {self['prefix']!r}, " \
            f"'content': {self['content']!r}, " \
            f"'suffix': {self['suffix']!r}" \
            "})"
    
    def __len__(self) -> int:
        return len(self.as_string())

    def as_string(self) -> str:
        """Return the full message string (prefix + content + suffix)"""
        return self['prefix'] + self['content'] + self['suffix']


class Thread:
    """
    Provide functionality to facilitate easy interactions with a Llama model

    The following methods are available:
    - create_message:
        Create and return a message with the specified role and content
    - len_messages:
        Return the total length of all messages in tokens
    - add_message:
        Create a message and add it to the list of messages
    - inference_str_from_messages:
        Using the list of messages, construct a string suitable for inference
    - send:
        Send a message in this thread and return the generated response
    - interact:
        Start an interactive, terminal-based chat
    - reset:
        Reset the thread to its original state
    - as_string:
        Return the content of the thread as a single string
    - print_stats:
        Print brief statistics about this thread, including length in tokens,
        max context length, percentage of context used, and total number of
        messages
    - summarize:
        Generate a summary of this thread
    - warmup:
        Ingest the text of this thread into its model's cache, reducing the
        latency of future generations

    The following attributes are available:
    - model:
        The `easy_llama.Model` instance used in this thread
    - format:
        A dictionary describing the way messages should be structured within
        this thread (or an AdvancedFormat instance)
    - sampler:
        The `easy_llama.samplers.SamplerSettings` instance used in this thread,
        which describes the sampler parameters used for generation
    - messages:
        The list of all messages in this thread
    - uuid: `uuid.UUID`
        A randomly generated UUID, unique to this specific thread instance
    """

    def __init__(
        self,
        model: Model,
        format: dict | AdvancedFormat,
        sampler: Optional[SamplerSettings] = None,
        messages: Optional[list[Message]] = None
    ):
        """
        Given a Model and a format, construct a Thread instance.

        model: The Model to use for text generation
        format: The format specifying how messages should be structured (see ez.formats)

        The following parameters are optional:
        - sampler: The SamplerSettings object used to control text generation
        - messages: A list of ez.thread.Message objects to add to the Thread upon construction
        """
        
        assert_type(model, Model, 'model', 'Thread')
        assert_type(format, (dict, AdvancedFormat), 'format', 'Thread')
        
        _format_keys = format.keys() # only read once

        if 'system_prompt' not in _format_keys and 'system_content' in _format_keys:
            raise KeyError(
                "Thread: format uses deprecated 'system_content' key instead "
                "of the expected 'system_prompt' key - please update your "
                "code accordingly"
            )

        if any(k not in _format_keys for k in formats_blank().keys()):
            raise KeyError(
                "Thread: format is missing one or more required keys, see " + \
                "easy_llama.formats.blank for an example"
            )
        
        assert_type(format['stops'], list, "format['stops']", 'Thread')

        sampler = SamplerSettings() if sampler is None else sampler
        
        if not all(hasattr(sampler, attr) for attr in SamplerSettings.param_types):
            raise AttributeError(
                'Thread: sampler is missing one or more required attributes'
            ).add_note(
                "Are you sure the specified sampler is really an instance of "
                "ez.samplers.SamplerSettings?"
            )

        self._messages: Optional[list[Message]] = messages
        if self._messages is not None:
            if not all(isinstance(msg, Message) for msg in self._messages):
                raise TypeError(
                    "Thread: one or more messages provided to __init__() is "
                    "not an instance of ez.thread.Message"
                )
        
        # Thread.messages is never empty, unless `messages` param is explicity
        # set to `[]` during construction

        self.model: Model = model
        self.format: dict | AdvancedFormat = format
        self.sampler: SamplerSettings = sampler
        self.messages: list[Message] = [
            self.create_message("system", self.format['system_prompt'])
        ] if self._messages is None else self._messages
        self.uuid = uuid.uuid4()

        if self.model.verbose:
            print_verbose(f"new Thread instance with the following attributes:")
            print_verbose(f"   uuid                      == {self.uuid}")
            print_verbose(f"   model.uuid                == {self.model.uuid}")
            print_verbose(
                f"   format['system_prefix']   == "
                f"{truncate(repr(self.format['system_prefix']))}"
            )
            print_verbose(
                f"   format['system_prompt']   == "
                f"{truncate(repr(self.format['system_prompt']))}"
            )
            print_verbose(
                f"   format['system_suffix']   == "
                f"{truncate(repr(self.format['system_suffix']))}"
            )
            print_verbose(
                f"   format['user_prefix']     == "
                f"{truncate(repr(self.format['user_prefix']))}"
            )
            print_verbose(
                f"   format['user_suffix']     == "
                f"{truncate(repr(self.format['user_suffix']))}"
            )
            print_verbose(
                f"   format['bot_prefix']      == "
                f"{truncate(repr(self.format['bot_prefix']))}"
            )
            print_verbose(
                f"   format['bot_suffix']      == "
                f"{truncate(repr(self.format['bot_suffix']))}"
            )
            print_verbose(
                f"   format['stops']           == "
                f"{truncate(repr(self.format['stops']))}"
            )
            print_verbose(f"this Thread uses the following sampler settings:")
            print_sampler_settings(self.sampler)


    def __repr__(self) -> str:
        # if only message in self.messages is system message
        if len(self.messages) == 1 and self.messages[0]['role'] == 'system':
            # do not represent it because it is constructed based on
            # the format, which is already represented
            return f"Thread({self.model!r}, {self.format!r}, " + \
                   f"{self.sampler!r})"
        # represent all messages, potentially including a system message
        return f"Thread({self.model!r}, {self.format!r}, " + \
                f"{self.sampler!r}, {self.messages!r})"
    

    def __str__(self) -> str:
        return self.as_string()
    

    def __len__(self) -> int:
        """
        `len(Thread)` returns the length of the Thread in tokens

        To get the number of messages in the Thread, use `len(Thread.messages)`
        """
        return self.len_messages()


    def create_message(
        self,
        role: Literal['system', 'user', 'bot'],
        content: str
    ) -> Message:
        """
        Construct a message using the format of this Thread
        """

        assert_type(role, str, 'role', 'create_message')
        assert_type(content, str, 'content', 'create_message')

        if role.lower() not in ['system', 'user', 'bot']:
            raise ValueError(
                "create_message: role should be 'system', 'user', or 'bot', "
                f"not {role.lower()!r}"
            )

        if role.lower() == 'system':
            return Message(
                {
                    'role': 'system',
                    'prefix': self.format['system_prefix'],
                    'content': content,
                    'suffix': self.format['system_suffix']
                }
            )
        
        elif role.lower() == 'user':
            return Message(
                {
                    'role': 'user',
                    'prefix': self.format['user_prefix'],
                    'content': content,
                    'suffix': self.format['user_suffix']
                }
            )
        
        elif role.lower() == 'bot':
            return Message(
                {
                    'role': 'bot',
                    'prefix': self.format['bot_prefix'],
                    'content': content,
                    'suffix': self.format['bot_suffix']
                }
            )
        
        else:
            raise UnreachableException
    

    def len_messages(self) -> int:
        """
        Return the total length of all messages in this thread, in tokens.
        
        Equivalent to `len(Thread)`.
        """

        return self.model.get_length(self.as_string())


    def add_message(
        self,
        role: Literal['system', 'user', 'bot'],
        content: str
    ) -> None:
        """
        Create a message and append it to `Thread.messages`.

        `Thread.add_message(...)` is a shorthand for
        `Thread.messages.append(Thread.create_message(...))`
        """
        self.messages.append(
            self.create_message(
                role=role,
                content=content
            )
        )


    def inference_str_from_messages(self) -> str:
        """
        Alias to `.inf_str()` for backward compatability
        """
        return self.inf_str()
    

    def inference_str_for_role(
            self,
            role: str = Literal['system', 'user', 'bot']
    ) -> str:
        """
        Alias to `.inf_str()` for backward compatability
        """
        return self.inf_str_for_role(role=role)
    

    def inf_str(self) -> str:
        """
        Using the list of messages, construct a string suitable for inference,
        respecting the format of this thread. It is assumed that the incoming
        generation is for the role of `bot`. To get an inference string for
        `user` or `system` messages, see `inf_str_for_role`.
        """

        inf_str = ''

        if len(self.messages) == 0:
            return self.format['bot_prefix']
        
        for message in reversed(self.messages):
            msg_str = message.as_string()
            inf_str = msg_str + inf_str

        inf_str += self.format['bot_prefix']

        return inf_str
    

    def inf_str_for_role(
            self, role: str = Literal['system', 'user', 'bot']
        ) -> str:
        """
        Using the list of messages, construct a string suitable for inference,
        respecting the format of this thread. The `role` parameter specifies
        whether the following generated text is from `system`, `user`, or `bot`.
        If `role` is `bot`, the behaviour is equivalent to
        `inference_str_from_messages`.
        """

        assert_type(role, str, 'role', 'inf_str_for_role')
        role = role.lower()
        if role not in ['system', 'user', 'bot']:
            raise ValueError(
                "inf_str_for_role: `role` must be 'system', 'user', or 'bot'"
            )
        
        # i can't think of a reason you would ever need to generate a system
        # message on the fly, but i might as well give the user the option
        
        inf_str = ''

        if role == 'system':
            if len(self.messages) == 0:
                return self.format['system_prefix']
            
            for message in reversed(self.messages):
                inf_str = message.as_string() + inf_str

            inf_str += self.format['system_prefix']

            return inf_str
            
        elif role == 'user':
            if len(self.messages) == 0:
                return self.format['user_prefix']
            
            for message in reversed(self.messages):
                inf_str = message.as_string() + inf_str

            inf_str += self.format['user_prefix']

            return inf_str
        
        elif role == 'bot':
            if len(self.messages) == 0:
                return self.format['bot_prefix']
            
            for message in reversed(self.messages):
                inf_str = message.as_string() + inf_str

            inf_str += self.format['bot_prefix']

            return inf_str
        
        else:
            raise UnreachableException


    def send(self, prompt: str) -> str:
        """
        Send a message in this thread. This adds your message and the bot's
        response to the list of messages.

        Returns a string containing the response to your message.
        """

        self.add_message("user", prompt)
        output = self.model.generate(
            self.inference_str_from_messages(),
            stops=self.format['stops'],
            sampler=self.sampler
        )
        self.add_message("bot", output)

        return output
    

    def _interactive_update_sampler(self) -> None:
        """Interactively update the sampler settings used in this Thread"""
        print()

        _sentinel = False

        try:
            new_max_len_tokens = input(
                f'max_len_tokens: {self.sampler.max_len_tokens} -> '
            )
            new_top_k = input(f'top_k: {self.sampler.top_k} -> ')
            new_top_p = input(f'top_p: {self.sampler.top_p} -> ')
            new_min_p = input(f'min_p: {self.sampler.min_p} -> ')
            new_temp = input(f'temp: {self.sampler.temp} -> ')
            new_frequency_penalty = input(
                f'frequency_penalty: {self.sampler.frequency_penalty} -> '
            )
            new_presence_penalty = input(
                f'presence_penalty: {self.sampler.presence_penalty} -> '
            )
            new_repeat_penalty = input(
                f'repeat_penalty: {self.sampler.repeat_penalty} -> '
            )
            _sentinel = True

        except KeyboardInterrupt:
            print('\neasy_llama: sampler settings not updated\n')
            return
        print()

        try:
            self.sampler.max_len_tokens = int(new_max_len_tokens)
        except ValueError:
            pass
        else:
            print('easy_llama: max_len_tokens updated')
        
        try:
            self.sampler.top_k = int(new_top_k)
        except ValueError:
            pass
        else:
            print('easy_llama: top_k updated')
        
        try:
            self.sampler.top_p = float(new_top_p)
        except ValueError:
            pass
        else:
            print('easy_llama: top_p updated')

        try:
            self.sampler.min_p = float(new_min_p)
        except ValueError:
            pass
        else:
            print('easy_llama: min_p updated')
        
        try:
            self.sampler.temp = float(new_temp)
        except ValueError:
            pass
        else:
            print('easy_llama: temp updated')

        try:
            self.sampler.frequency_penalty = float(new_frequency_penalty)
        except ValueError:
            pass
        else:
            print('easy_llama: frequency_penalty updated')
        
        try:
            self.sampler.presence_penalty = float(new_presence_penalty)
        except ValueError:
            pass
        else:
            print('easy_llama: presence_penalty updated')
        
        try:
            self.sampler.repeat_penalty = float(new_repeat_penalty)
        except ValueError:
            pass
        else:
            print('easy_llama: repeat_penalty updated')
        
        if _sentinel:   # pretty formatting
            print()
    

    def _interactive_input(
        self,
        prompt: str,
        _dim_style: str,
        _user_style: str,
        _bot_style: str,
        _special_style: str,
        _error_style: str
    ) -> tuple:
        """
        Recive input from the user, while handling multi-line input
        and commands
        """
        full_user_input = '' # may become multiline
        
        while True:
            user_input = input(prompt)
            
            if user_input.endswith('\\'):
                full_user_input += user_input[:-1] + '\n'
            
            elif user_input == '!':

                print()
                try:
                    command = input(f'{RESET_ALL}  ! {_dim_style}')
                except KeyboardInterrupt:
                    print('\n')
                    continue

                if command == '':
                    print(f'\n{_error_style}[no command]{RESET_ALL}\n')

                elif command.lower() in ['reset', 'restart']:
                    self.reset()
                    print(f'\n[thread reset]\n')

                elif command.lower() in ['cls', 'clear']:
                    cls()
                    print()

                elif command.lower() in ['ctx', 'context']:
                    print(f"\n{self.len_messages()}\n")

                elif command.lower() in ['stats', 'print_stats']:
                    print()
                    self.print_stats()
                    print()
                
                elif command.lower() in ['sampler', 'samplers', 'settings']:
                    self._interactive_update_sampler()
                
                elif command.lower() in ['str', 'string', 'as_string']:
                    print(f"\n{self.as_string()}\n")
                
                elif command.lower() in ['repr', 'save', 'backup']:
                    print(f"\n{self!r}\n")
                
                elif command.lower() in ['remove', 'rem', 'delete', 'del']:
                    print()
                    if len(self.messages) == 0:
                        print(f'{_error_style}[no previous message]{RESET_ALL}\n')
                    else:
                        del self.messages[-1]
                        print('[removed last message]\n')

                elif command.lower() in ['last', 'repeat', 'what?']:
                    print()
                    if len(self.messages) == 0:
                        print(f'{_error_style}[no previous message]{RESET_ALL}\n')
                    else:
                        last_msg = self.messages[-1]
                        if last_msg['role'] == 'system':
                            print(f"{_special_style}{last_msg['content']}{RESET_ALL}\n")
                        elif last_msg['role'] == 'user':
                            print(f"{_user_style}{last_msg['content']}{RESET_ALL}\n")
                        elif last_msg['role'] == 'bot':
                            print(f"{_bot_style}{last_msg['content']}{RESET_ALL}\n")
                        else:
                            print(f"{_error_style}{last_msg['content']}{RESET_ALL}\n")
                
                elif command.lower() in ['refill', 'reprint', 'lost', 'find']:
                    cls()
                    print()
                    if len(self.messages) == 0:
                        print(f'{_error_style}[no previous message]{RESET_ALL}\n')
                    else:
                        for msg in self.messages:
                            if msg['role'] == 'system':
                                print(f"{_special_style}{msg['content']}{RESET_ALL}\n")
                            elif msg['role'] == 'user':
                                print(f"{_user_style}{msg['content']}{RESET_ALL}\n")
                            elif msg['role'] == 'bot':
                                print(f"{_bot_style}{msg['content']}{RESET_ALL}\n")
                            else:
                                print(f"{_error_style}{msg['content']}{RESET_ALL}\n")
                        
                elif command.lower() in ['inf', 'inference', 'inf_str']:
                    print(f'\n"""{self.inference_str_from_messages()}"""\n')
                
                elif command.lower() in ['reroll', 're-roll', 're', 'swipe']:
                    old_len = len(self.messages)
                    del self.messages[-1]
                    assert len(self.messages) == (old_len - 1)
                    return '', None
                
                elif command.lower() in ['exit', 'quit']:
                    print(RESET_ALL)
                    return None, None
                
                elif command.lower() in ['sum', 'summ', 'summary', 'summarize']:
                    print('\nGenerating summary...\n')
                    print(self.summarize())
                    print()
                
                elif command.lower() in ['msgs', 'messages', 'msg']:
                    print(f'\n{self.messages}\n')
                
                elif command.lower() in ['help', '/?', '?']:
                    print()
                    print('reset | restart     -- Reset the thread to its original state')
                    print('clear | cls         -- Clear the terminal')
                    print('context | ctx       -- Get the context usage in tokens')
                    print('print_stats | stats -- Get the context usage stats')
                    print('sampler | settings  -- Update the sampler settings')
                    print('string | str        -- Print the message history as a string')
                    print('repr | save         -- Print the representation of the thread')
                    print('remove | delete     -- Remove the last message')
                    print('last | repeat       -- Repeat the last message')
                    print('inference | inf     -- Print the inference string')
                    print('reroll | swipe      -- Regenerate the last message')
                    print('summary | summarize -- Generate a summary of the thread')
                    print('messages | msg      -- Print the raw list of all messages')
                    print('exit | quit         -- Exit the interactive chat (can also use ^C)')
                    print('help | ?            -- Show this screen')
                    print()
                    print("TIP: type < at the prompt and press ENTER to prefix the bot's next message.")
                    print('     for example, type "Sure!" to bypass refusals')
                    print()
                    print("TIP: type !! at the prompt and press ENTER to insert a system message")
                    print()

                else:
                    print(f'\n{_error_style}[unknown command]{RESET_ALL}\n')
            
            # prefix the bot's next message
            elif user_input == '<':

                print()
                try:
                    next_message_start = input(f'{RESET_ALL}  < {_dim_style}')

                except KeyboardInterrupt:
                    print(f'{RESET_ALL}\n')
                    continue

                else:
                    print()
                    return '', next_message_start

            # insert a system message
            elif user_input == '!!':
                print()

                try:
                    next_sys_msg = input(f'{RESET_ALL} !! {_special_style}')
                
                except KeyboardInterrupt:
                    print(f'{RESET_ALL}\n')
                    continue
                
                else:
                    print()
                    return next_sys_msg, -1

            # concatenate multi-line input
            else:
                full_user_input += user_input
                return full_user_input, None


    def interact(
        self,
        color: bool = True,
        header: Optional[str] = None,
        stream: bool = True,
        hook: Optional[Callable] = None,
        prompt: str = '  > '
    ) -> None:
        """
        Start an interactive chat session using this Thread.

        While text is being generated, press `^C` to interrupt the bot.
        Then you have the option to press `ENTER` to re-roll, or to simply type
        another message.

        At the prompt, press `^C` to end the chat session.

        End your input with a backslash `\\` for multi-line input.

        Type `!` and press `ENTER` to enter a basic command prompt. For a list
        of  commands, type `help` at this prompt.
        
        Type `<` and press `ENTER` to prefix the bot's next message, for
        example with `Sure!`.

        Type `!!` at the prompt and press `ENTER` to insert a system message.

        The following parameters are optional:
        - color: Whether to use colored text to differentiate user / bot
        - header: Header text to print at the start of the interaction
        - stream: Whether to stream text as it is generated
        """
        print()

        if color:
            from .utils import (
                SPECIAL_STYLE, USER_STYLE,
                BOT_STYLE, DIM_STYLE,
                ERROR_STYLE
            )
        else:
            SPECIAL_STYLE = ''
            USER_STYLE = ''
            BOT_STYLE = ''
            DIM_STYLE = ''
            ERROR_STYLE = ''
        
        if header is not None:
            print(f"{SPECIAL_STYLE}{header}{RESET_ALL}\n")
        
        while True:

            if hook is not None:
                print(DIM_STYLE, end='', flush=True)
                hook(self)
                print(RESET_ALL, end='', flush=True)

            prompt = f"{RESET_ALL}{prompt}{USER_STYLE}"
            
            try:
                user_prompt, next_message_start = self._interactive_input(
                    prompt,
                    DIM_STYLE,
                    USER_STYLE,
                    BOT_STYLE,
                    SPECIAL_STYLE,
                    ERROR_STYLE
                )
            except KeyboardInterrupt:
                print(f"{RESET_ALL}\n")
                return
            
            # got 'exit' or 'quit' command
            if user_prompt is None and next_message_start is None:
                break
            
            # insert a system message via `!!` prompt
            if next_message_start == -1:
                self.add_message('system', user_prompt)
                continue
            
            if next_message_start is not None:
                try:
                    if stream:
                        print(f"{BOT_STYLE}{next_message_start}", end='', flush=True)
                        output = next_message_start + self.model.stream_print(
                            self.inference_str_from_messages() + next_message_start,
                            stops=self.format['stops'],
                            sampler=self.sampler,
                            end=''
                        )
                    else:
                        print(f"{BOT_STYLE}", end='', flush=True)
                        output = next_message_start + self.model.generate(
                            self.inference_str_from_messages() + next_message_start,
                            stops=self.format['stops'],
                            sampler=self.sampler
                        )
                        print(output, end='', flush=True)
                except KeyboardInterrupt:
                    print(f"{DIM_STYLE} [message not added to history; press ENTER to re-roll]\n")
                    continue
                except ExceededContextLengthException as exc:
                    print(f'{ERROR_STYLE}{exc}{RESET_ALL}')
                    print(f'easy_llama: thread: {DIM_STYLE}{self!r}{RESET_ALL}')
                    raise
                else:
                    self.add_message("bot", output)
            else:
                print(BOT_STYLE)
                if user_prompt != "":
                    self.add_message("user", user_prompt)
                try:
                    if stream:
                        output = self.model.stream_print(
                            self.inference_str_from_messages(),
                            stops=self.format['stops'],
                            sampler=self.sampler,
                            end=''
                        )
                    else:
                        output = self.model.generate(
                            self.inference_str_from_messages(),
                            stops=self.format['stops'],
                            sampler=self.sampler
                        )
                        print(output, end='', flush=True)
                except KeyboardInterrupt:
                    print(f"{DIM_STYLE} [message not added to history; press ENTER to re-roll]\n")
                    continue
                except ExceededContextLengthException as exc:
                    print(f'{ERROR_STYLE}{exc}{RESET_ALL}')
                    print(f'easy_llama: thread: {DIM_STYLE}{self!r}{RESET_ALL}')
                    raise
                else:
                    self.add_message("bot", output)

            if output.endswith("\n\n"):
                print(RESET_ALL, end = '', flush=True)
            elif output.endswith("\n"):
                print(RESET_ALL)
            else:
                print(f"{RESET_ALL}\n")


    def reset(self) -> None:
        """
        Reset Thread.messages to its original state
        """
        self.messages: list[Message] = [
            self.create_message("system", self.format['system_prompt'])
        ] if self._messages is None else self._messages
    
    
    def as_string(self) -> str:
        """Return this thread's message history as a string"""
        return ''.join(msg.as_string() for msg in self.messages)

    
    def print_stats(
        self,
        file: _SupportsWriteAndFlush = sys.stdout,
    ) -> None:
        """Print stats about the context usage in this thread"""
        thread_len_tokens = self.len_messages()
        max_ctx_len = self.model.context_length
        c = (thread_len_tokens/max_ctx_len) * 100
        ctx_used_pct = int(c) + (c > int(c)) # round up to next integer
        print(f"{thread_len_tokens} / {max_ctx_len} tokens", file=file)
        print(f"{ctx_used_pct}% of context used", file=file)
        print(f"{len(self.messages)} messages", file=file)
    

    def summarize(
            self,
            messages: Optional[list[Message]] = None,
            model: Optional[Model] = None
        ):
        """
        Generate a summary from a list of messages. If no messages are
        provided, use `self.messages`. If no model is specified, use
        `self.model`.
        """

        assert_type(
            messages,
            (list, NoneType),
            'messages',
            'summarize'
        )

        if model is not None:
            _model = model
        else:
            _model = self.model
        
        _model.load()

        if messages == []:
            raise ValueError(
                f"summarize: the list of messages cannot be empty"
            )

        if messages is None:
            messages = self.messages

        messages_str = ''.join(msg.as_string() for msg in messages)

        inf_str = \
                self.create_message(
                    'system',
                    'Follow the given instructions exactly. Do not add any '
                    'unnecessary information.'
                ).as_string() + \
                self.create_message(
                    'user',
                    'Hello. Take a moment to read the following conversation '
                    'carefully. When you\'re done, write a single paragraph '
                    'that explains all of the most relevant details.'
                    f'\n\n```\n{messages_str}\n```\n\n'
                    'Now that you\'ve read the above conversation, please '
                    'provide a summary in the form of a single paragraph.'
                ).as_string() + \
                self.format['bot_prefix'] + \
                'After carefully reading through the conversation, here\'s' + \
                ' a paragraph that explains the most relevant details:\n\n'
        
        summary = _model.generate(
            inf_str,
            stops=self.format['stops'] + ['\n\n'],
            sampler=SamplerSettings(max_len_tokens=512)
        )

        # unload helper model, if used
        if model is not None:
            _model.unload()
        
        return summary


    def warmup(self):
        """
        Ingest the text of this thread into the model's cache
        """
        self.model.ingest(self.inference_str_from_messages())
