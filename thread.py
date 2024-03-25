# thread.py
# Python 3.11.7
# https://github.com/ddh0/easy-llama/

"""Submodule containing the Thread class, used for interaction with a Model"""

import globals
import time

from utils import get_timestamp_prefix_str, RESET_ALL, cls
from samplers import SamplerSettings, DefaultSampling
from formats import blank as blank_format
from typing import Optional
from model import Model

class Thread(object):
    """
    Provide functionality to facilitate easy interactions with a Model

    This is just a brief overview of easy_llama.Thread.
    To see a full description of each method and its parameters,
    call help(Thread), or see the relevant docstring.

    The following parameters are provided to the constructor:
    - model: the instance of easy_llama.Model to use in this Thread
    - format: the dict to use as the format for this Thread. see the module
    easy_llama.formats for examples and presets
    - timestamps: bool stating whether or not to prefix messages with
    timestamps to provide context to the chat
    - amnesiac: bool stating whether or not the Thread should be 'amnesiac',
    i.e only remember the system message. not compatible with timestamps or
    track_context
    - max_context_length: the maximum allowed length of the Thread's message
    history, in tokens
    - track_context: bool stating whether or not to print context stats after
    a message is sent or during interactions
    - sampler: the instance of easy_llama.samplers.SamplerSettings to use in
    this Thread, which controls how tokens are sampled from the Model

    The following methods are available:
    - .create_message(): create a message (dict) using the format of this Thread
    - .add_message(): shorthand for `Thread.messages.append(Thread.create_message(...))`
    - .inference_str_from_messages(): given a list of messages, construct a string
    that is suitable for inference, using the format and context length of this Thread
    - .send(): send a message in this thread. this adds your message and the bot's
    response to the Thread's history. returns a string containing the response to your message
    - .interact(): start an interactive chat session using this Thread. see the
    method's docstring for details
    - .reset(): reset the Thread to its original state
    - .print_stats(): print stats about the context usage in this Thread

    The following attributes are available:
    - .messages: a list of all messages in this Thread, in chronological order
    """
    def __init__(self,
                 model: Model,
                 format: dict,
                 timestamps: bool = False,
                 amnesiac: bool = False,
                 max_context_length: int = -1,
                 track_context: bool = False,
                 sampler: SamplerSettings = DefaultSampling
            ):
        """
        Construct an instance of easy_llama.Thread. See the class docstring for
        details
        """
        
        assert isinstance(model, Model), \
            "Thread: model should be an " + \
            f"instance of easy_llama.Model, not {type(model)}"

        assert isinstance(format, dict), \
            f"Thread: format should be dict, not {type(format)}"
        
        if any(k not in format.keys() for k in blank_format.keys()):
            raise KeyError(
                "Thread: format is missing one or more required keys, see " + \
                "easy_llama.formats.blank for an example"
            )

        assert isinstance(format["stops"], (list, type(None))), \
            "Thread: format['stops'] should be list[str] or None, " + \
            f"not {type(format['stops'])}"

        assert isinstance(timestamps, bool), \
            f"Thread: timestamps should be True or False, not '{timestamps}'"
        
        assert isinstance(amnesiac, bool), \
            f"Thread: amnesiac should be True or False, not '{amnesiac}'"
        
        assert isinstance(track_context, bool), \
            f"Thread: amnesiac should be True or False, not '{track_context}'"

        assert isinstance(max_context_length, int), \
            f"Thread: max_context_length should be int, not '{max_context_length}"
        
        if amnesiac and timestamps:
            raise RuntimeError(
                "Thread: amnesiac threads are not compatible with timestamps"
            )
        
        self.model: Model = model
        self.format: dict = format
        self.enable_timestamps: bool = timestamps
        self.amnesiac: bool = amnesiac
        self.messages: list[dict] = [
            self.create_message("system", self.format['system_content'])
        ]
        self.max_context_length: int = (
            max_context_length if max_context_length > 0 else self.model.context_length
        )
        self.track_context: bool = track_context
        self.sampler: SamplerSettings = sampler
        self._interact_color: bool = False

        if self.enable_timestamps:
            # stop generation on timestamps
            self.format['stops'].append(
                time.strftime('[%Y')
            )

        if globals.VERBOSE:
            print("--------------------- easy_llama.Thread -----------------------")
            print(f"self.model                     == {self.model}")
            print(f"self.enable_timestamps         == {self.enable_timestamps}")
            print(f"self.amnesiac                  == {self.amnesiac}")
            print(f"self.max_context_length        == {self.max_context_length}")
            print(f"self.track_context             == {self.track_context}")
            print(f"self.sampler.max_len_tokens    == {self.sampler.max_len_tokens}")
            print(f"self.sampler.temp              == {self.sampler.temp}")
            print(f"self.sampler.top_p             == {self.sampler.top_p}")
            print(f"self.sampler.min_p             == {self.sampler.min_p}")
            print(f"self.sampler.frequency_penalty == {self.sampler.frequency_penalty}")
            print(f"self.sampler.presence_penalty  == {self.sampler.presence_penalty}")
            print(f"self.sampler.repeat_penalty    == {self.sampler.repeat_penalty}")
            print(f"self.sampler.top_k             == {self.sampler.top_k}")
    

    def __repr__(self) -> str:
        repr_str = ''
        repr_str += f"Thread({repr(self.model)}, {repr(self.format)}, "
        repr_str += f"timestamps={self.enable_timestamps}, amnesiac={self.amnesiac}, "
        repr_str += f"max_context_length={self.max_context_length}, track_context={self.track_context}, "
        repr_str += f"sampler={repr(self.sampler)})"
        if len(self.messages) <= 1:
            return repr_str
        else:
            for msg in self.messages:
                # system message is created from format, so not represented
                if msg['role'] == 'user':
                    repr_str += '\nThread.add_message("user", "' + msg['content'] + '")'
                elif msg['role'] == 'bot':
                    repr_str += '\nThread.add_message("bot", "' + msg['content'] + '")'
            return repr_str


    def create_message(self, role: str, content: str) -> dict:
        """
        Create a message (dict) using the format of this Thread
        """
        assert role.lower() in ['system', 'user', 'bot'], \
            "create_message: role should be 'system', 'user', or " + \
            f"'bot', not '{role.lower()}'"

        assert isinstance(content, str), \
            f"create_message: content should be str, not {type(content)}"

        if role.lower() == 'system':
            prefix_maybe_with_timestamp: str = (
                self.format['system_prefix'] if not self.enable_timestamps else (
                get_timestamp_prefix_str() +
                self.format['system_prefix']
                )
            )
            return {
                "role": "system",
                "prefix": prefix_maybe_with_timestamp,
                "content": content,
                "postfix": self.format['system_postfix'],
                "tokens": (
                    self.model.llama.tokenize(
                        prefix_maybe_with_timestamp.encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos=True,
                        special=True
                    ) + self.model.llama.tokenize(
                        content.encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos = False,
                        special = False
                    ) + self.model.llama.tokenize(
                        self.format['system_postfix'].encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos = False,
                        special = True
                    )
                )
            }
        
        elif role.lower() == 'user':
            prefix_maybe_with_timestamp: str = (
                self.format['user_prefix'] if not self.enable_timestamps else (
                get_timestamp_prefix_str() +
                self.format['user_prefix']
                )
            )
            return {
                "role": "user",
                "prefix": prefix_maybe_with_timestamp,
                "content": content,
                "postfix": self.format['user_postfix'],
                "tokens": (
                    self.model.llama.tokenize(
                        prefix_maybe_with_timestamp.encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos=False,
                        special=True
                    ) + self.model.llama.tokenize(
                        content.encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos = False,
                        special = False
                    ) + self.model.llama.tokenize(
                        self.format['user_postfix'].encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos = False,
                        special = True
                    )
                )
            }
        
        elif role.lower() == 'bot':
            prefix_maybe_with_timestamp: str = (
                self.format['bot_prefix'] if not self.enable_timestamps else (
                get_timestamp_prefix_str() +
                self.format['bot_prefix']
                )
            )
            return {
                "role": "bot",
                "prefix": prefix_maybe_with_timestamp,
                "content": content,
                "postfix": self.format['bot_postfix'],
                "tokens": (
                    self.model.llama.tokenize(
                        prefix_maybe_with_timestamp.encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos=False,
                        special=True
                    ) + self.model.llama.tokenize(
                        content.encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos = False,
                        special = False
                    ) + self.model.llama.tokenize(
                        self.format['bot_postfix'].encode(
                            'utf-8', errors='ignore'
                        ),
                        add_bos = False,
                        special = True
                    )
                )
            }
    
    def len_messages(self, messages: Optional[list[dict]] = None):
        """Return the total length of all `messages` in tokens"""
        if messages is None:
            messages = self.messages
        
        return sum([len(message['tokens']) for message in messages])

    def add_message(self, role: str, content: str) -> None:
        """
        `Thread.add_message(...)` is a shorthand for
        `Thread.messages.append(Thread.create_message(...))`
        """
        self.messages.append(
            self.create_message(
                role=role,
                content=content
            )
        )

    def inference_str_from_messages(self, messages: Optional[list[dict]] = None) -> str:
        """
        Given a list of messages, construct a string suitable for
        inference, using the format and context length of this Thread
        """

        if messages is None:
            messages = self.messages

        context_len_budget = self.max_context_length
        if len(messages) > 0:
            system_message = messages[0]
            sys_msg_str = (
                system_message['prefix'] +
                system_message['content'] +
                system_message['postfix']
            )
        else:
            context_len_budget -= len(system_message['tokens'])
            sys_msg_str = ''
        
        if self.amnesiac:
            
            # TODO
            # Currently, max_context_length only applies for non-amnesiac
            # threads....

            assert self.enable_timestamps is False

            last_msg = messages[-1]
            last_msg_str = (
                last_msg['prefix'] +
                last_msg['content'] +
                last_msg['postfix']
            )
            
            inf_str = (
                sys_msg_str +
                last_msg_str +
                self.format['bot_prefix']
            )
            
            #if globals.VERBOSE:
            #    print(f'easy_llama: inference str is \"\"\"{inf_str}\"\"\"')
            return inf_str

        else:

            inf_str = ''

            # Start at most recent message and work backwards up the history
            # excluding system message. Once we exceed thread
            # max_context_length, break without including that message
            for message in reversed(messages[1:]):
                context_len_budget -= len(message['tokens'])

                if context_len_budget <= 0:
                    break

                msg_str = (
                    message['prefix'] +
                    message['content'] +
                    message['postfix']
                )
                
                inf_str = msg_str + inf_str

            inf_str = sys_msg_str + inf_str
            inf_str += self.format['bot_prefix'] if not self.enable_timestamps else (
                get_timestamp_prefix_str() + self.format['bot_prefix']
            )

            #if globals.VERBOSE:
            #    print(f'easy_llama: inference str is \"\"\"{inf_str}\"\"\"')
            return inf_str


    def send(self, prompt: str) -> str:
        """
        Send a message in this thread. This adds your message and the bot's
        response to the Thread's history.

        Returns a string containing the response to your message.
        """
        assert isinstance(prompt, str), \
            f"Thread.send: prompt should be str, not {type(prompt)}"

        self.add_message("user", prompt)
        output = self.model.generate(
            self.inference_str_from_messages(self.messages),
            stops=self.format['stops'],
            sampler=self.sampler
        )
        self.add_message("bot", output)

        if self.track_context:
            print(f"track_context: total tokens so far: {self.len_messages()}")

        return output
    

    def interactive_update_sampler(self) -> None:

        print()
        try:
            new_max_len_tokens    = input(f'max_len_tokens     {self.sampler.max_len_tokens} \t\t-> ')
            new_temp              = input(f'temp:              {self.sampler.temp} \t\t-> ')
            new_top_p             = input(f'top_p:             {self.sampler.top_p} \t-> ')
            new_min_p             = input(f'min_p:             {self.sampler.min_p} \t-> ')
            new_frequency_penalty = input(f'frequency_penalty: {self.sampler.frequency_penalty} \t\t-> ')
            new_presence_penalty  = input(f'presence_penalty:  {self.sampler.presence_penalty} \t\t-> ')
            new_repeat_penalty    = input(f'repeat_penalty:    {self.sampler.repeat_penalty} \t\t-> ')
            new_top_k             = input(f'top_k:             {self.sampler.top_k} \t\t-> ')
        
        except KeyboardInterrupt:
            print('\neasy_llama: sampler not updated\n')
            return
        print()

        try:
            self.sampler.max_len_tokens = int(new_max_len_tokens)
        except ValueError:
            print('easy_llama: max_len_tokens not updated')
        else:
            print('easy_llama: max_len_tokens updated')
        
        try:
            self.sampler.temp = float(new_temp)
        except ValueError:
            print('easy_llama: temp not updated')
        else:
            print('easy_llama: temp updated')
        
        try:
            self.sampler.top_p = float(new_top_p)
        except ValueError:
            print('easy_llama: top_p not updated')
        else:
            print('easy_llama: top_p updated')

        try:
            self.sampler.min_p = float(new_min_p)
        except ValueError:
            print('easy_llama: min_p not updated')
        else:
            print('easy_llama: min_p updated')

        try:
            self.sampler.frequency_penalty = float(new_frequency_penalty)
        except ValueError:
            print('easy_llama: frequency_penalty not updated')
        else:
            print('easy_llama: frequency_penalty updated')
        
        try:
            self.sampler.presence_penalty = float(new_presence_penalty)
        except ValueError:
            print('easy_llama: presence_penalty not updated')
        else:
            print('easy_llama: presence_penalty updated')
        
        try:
            self.sampler.repeat_penalty = float(new_repeat_penalty)
        except ValueError:
            print('easy_llama: repeat_penalty not updated')
        else:
            print('easy_llama: repeat_penalty updated')
        
        try:
            self.sampler.top_k = int(new_top_k)
        except ValueError:
            print('easy_llama: top_k not updated')
        else:
            print('easy_llama: top_k updated')
        
        print()
                

    def interactive_input(self,
                          prompt: str,
                          _user_style: str,
                          _bot_style: str,
                          _dim_style: str
                          ) -> str:
        """
        Recieve (optionally) multi-line input from the user and return the
        entered string. Lines must end with a backslash `\` in order to recieve
        another line. Lines are separated by `\\n`

        Works just like normal `input()` if the input does not end with a backslash
        """
        res = ''
        
        while True:
            s = input(prompt)
            
            if s.endswith('\\'):
                res += s[:-1] + '\n'
            
            elif s == '!':

                print()
                try:
                    c = input(f'{RESET_ALL}  ! {_dim_style}')

                except KeyboardInterrupt:
                    print('\n')
                    continue

                else:
                    if c.lower() in ['reset', 'restart']:
                        self.reset()
                        print()

                    elif c.lower() in ['cls', 'clear']:
                        cls()
                        print()

                    elif c.lower() in ['ctx', 'context', 'track_context']:
                        print(f"\n{self.len_messages()}\n")

                    elif c.lower() in ['stats', 'print_stats']:
                        print()
                        self.print_stats()
                        print()
                    
                    elif c.lower() in ['sampler', 'samplers', 'samplersettings']:
                        self.interactive_update_sampler()
                    
                    elif c.lower() in ['str', 'string', 'as_string']:
                        print(f"\n{self.as_string()}\n")
                    
                    elif c.lower() in ['repr', 'save', 'backup']:
                        print(f"\n{repr(self)}\n")
                    
                    elif c.lower() in ['color', 'colors']:
                        self._interact_color = not self._interact_color
                        if self._interact_color:
                            print(f"\ncolors are ON\n")
                        else:
                            print(f"\ncolors are OFF\n")
                    
                    elif c.lower() in ['undo', 'del', 'remove', 'back']:
                        print()
                        old_len = len(self.messages)
                        del self.messages[-1]
                        assert len(self.messages) == (old_len - 1)
                        print('easy_llama: removed last message\n')

                    elif c.lower() in ['last', 'repeat']:
                        print('\neasy_llama: re-printing last message in history\n')
                        print(self.messages[-1]['content'] + '\n')
                    
                    elif c.lower() in ['inf', 'inference', 'inf_str']:
                        print(f'\n"""{self.inference_str_from_messages()}"""\n')
                    
                    elif c.lower() in ['reroll', 're-roll', 're']:
                        old_len = len(self.messages)
                        del self.messages[-1]
                        assert len(self.messages) == (old_len - 1)
                        return '', None

                    else:
                        print(f'\neasy_llama: unknown command\n')
            
            elif s == '<': # the next bot message will start with...

                print()
                try:
                    next_message_start = input(f'{_dim_style}  < ')

                except KeyboardInterrupt:
                    print(f'{RESET_ALL}\n')
                    continue

                else:
                    print()
                    return '', next_message_start
            
            elif s.endswith('<'):

                print()

                msg = s.removesuffix('<')
                self.add_message("user", msg)
                
                try:
                    next_message_start = input(f'{_dim_style}  < ')

                except KeyboardInterrupt:
                    print(f'{RESET_ALL}\n')
                    continue

                else:
                    print()
                    return '', next_message_start
                    

            else:
                res += s
                return res, None


    def interact(self, color: bool = True) -> None:
        """
        Start an interactive chat session using this Thread.

        While text is being generated, press ^C to interrupt the bot.
        Then you have the option to press ENTER to re-roll, or to simply type
        another message.

        At the prompt, press ^C to end the chat session.
        """
        print()

        # fresh import of color codes in case `color` param has changed
        from utils import USER_STYLE, BOT_STYLE, DIM_STYLE

        # disable color codes if explicitly disabled by `color` param
        if not color:
            USER_STYLE = ''
            BOT_STYLE = ''
            DIM_STYLE = ''
        
        while True:

            if self.track_context:
                prompt = f"{RESET_ALL}{self.len_messages()} > {USER_STYLE}"
            else:
                prompt = f"{RESET_ALL}  > {USER_STYLE}"
            
            try:
                user_prompt, next_message_start = self.interactive_input(
                    prompt,
                    USER_STYLE, 
                    BOT_STYLE,
                    DIM_STYLE
                )
            except KeyboardInterrupt:
                print(f"{RESET_ALL}\n")
                return
            
            if next_message_start is not None:
                print(f"{BOT_STYLE}{next_message_start}", end='', flush=True)
                try:
                    output = next_message_start + self.model.stream_print(
                        self.inference_str_from_messages(self.messages) + next_message_start,
                        stops=self.format['stops'],
                        sampler=self.sampler,
                        end=''
                    )
                except KeyboardInterrupt:
                    print(f"{DIM_STYLE} [message not added to history; press ENTER to re-roll]\n")
                    continue
                else:
                    self.add_message("bot", output)
            else:
                print(BOT_STYLE)
                if user_prompt != "":
                    self.add_message("user", user_prompt)
                try:
                    output = self.model.stream_print(
                        self.inference_str_from_messages(self.messages),
                        stops=self.format['stops'],
                        sampler=self.sampler,
                        end=''
                    )
                except KeyboardInterrupt:
                    print(f"{DIM_STYLE} [message not added to history; press ENTER to re-roll]\n")
                    continue
                else:
                    self.add_message("bot", output)

            if output.endswith("\n\n"):
                print(f"{RESET_ALL}", end = '', flush=True)
            elif output.endswith("\n"):
                print(f"{RESET_ALL}")
            else:
                print(f"{RESET_ALL}\n")
            
            # EXPERIMENTAL
            # raise temperature as conversation goes on to avoid getting
            # stuck in loops


    def reset(self) -> None:
        """Reset the Thread to its original state"""
        self.messages: list[dict] = [
            self.create_message("system", self.format['system_content'])
        ]
    
    
    def as_string(self) -> str:
        """Return this Thread's message history as a string"""
        ret = ''
        for msg in self.messages:
            ret += msg['prefix']
            ret += msg['content']
            ret += msg['postfix']
        return ret

    
    def print_stats(self) -> None:
        """Print stats about the context usage in this Thread"""
        thread_len_tokens = self.len_messages()
        context_used_percentage = (
            round((thread_len_tokens/self.max_context_length)*100)
        )
        print(f"{thread_len_tokens} / {self.max_context_length} tokens")
        print(f"{context_used_percentage}% of context used")
        print(f"{len(self.messages)} messages")
