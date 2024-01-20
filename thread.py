# thread.py
# Python 3.11.6

"""Submodule containing the Thread class, used for interaction with a Model"""

import globals
import time

from samplers import SamplerSettings, DefaultSampling
from utils import get_timestamp_prefix_str, multiline_input
from model import Model


class Thread(object):
    """
    Provide functionality to facilitate easy interactions with a Model

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

        try:
            format['system_prefix']
            format['system_content']
            format['system_postfix']
            format['user_prefix']
            format['user_content']
            format['user_postfix']
            format['bot_prefix']
            format['bot_content']
            format['bot_postfix']
            format['stops']
        except KeyError as e:
            e.add_note(
                "Thread: format is missing one or more required keys, see " + \
                "easy_llama.formats.blank for an example"
            )
            raise

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

        if self.enable_timestamps:
            # stop generation on timestamps
            self.format['stops'].append(
                time.strftime('[%Y')
            )

        if globals.VERBOSE:
            print("--------------------- easy_llama.Thread -----------------------")
            print(f"self.model                    == {self.model}")
            print(f"self.enable_timestamps        == {self.enable_timestamps}")
            print(f"self.amnesiac                 == {self.amnesiac}")
            print(f"self.max_context_length       == {self.max_context_length}")
            print(f"self.track_context            == {self.track_context}")
            print(f"self.sampler.max_len_tokens   == {self.sampler.max_len_tokens}")
            print(f"self.sampler.temp             == {self.sampler.temp}")
            print(f"self.sampler.top_p            == {self.sampler.top_p}")
            print(f"self.sampler.min_p            == {self.sampler.min_p}")
            print(f"self.sampler.presence_penalty == {self.sampler.presence_penalty}")
            print(f"self.sampler.repeat_penalty   == {self.sampler.repeat_penalty}")
            print(f"self.sampler.top_k            == {self.sampler.top_k}")


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
                "prefix": prefix_maybe_with_timestamp,
                "content": content,
                "postfix": self.format['system_postfix'],
                "tokens": self.model.llama.tokenize(
                    prefix_maybe_with_timestamp.encode() + \
                    content.encode() + \
                    self.format['system_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
                ),
            }
        
        elif role.lower() == 'user':
            prefix_maybe_with_timestamp: str = (
                self.format['user_prefix'] if not self.enable_timestamps else (
                get_timestamp_prefix_str() +
                self.format['user_prefix']
                )
            )
            return {
                "prefix": prefix_maybe_with_timestamp,
                "content": content,
                "postfix": self.format['user_postfix'],
                "tokens": self.model.llama.tokenize(
                    prefix_maybe_with_timestamp.encode() + \
                    content.encode() + \
                    self.format['user_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
                ),
            }
        
        elif role.lower() == 'bot':
            prefix_maybe_with_timestamp: str = (
                self.format['bot_prefix'] if not self.enable_timestamps else (
                get_timestamp_prefix_str() +
                self.format['bot_prefix']
                )
            )
            return {
                "prefix": prefix_maybe_with_timestamp,
                "content": content,
                "postfix": self.format['bot_postfix'],
                "tokens": self.model.llama.tokenize(
                    prefix_maybe_with_timestamp.encode() + \
                    content.encode() + \
                    self.format['bot_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
                ),
            }

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

    def inference_str_from_messages(self, messages: list[dict]) -> str:
        """
        Given a list of messages, construct a string that is suitable for
        inference, using the format and context length of this Thread
        """

        system_message = messages[0]
        context_len_budget = self.max_context_length
        context_len_budget -= len(system_message['tokens'])

        sys_msg_str = (
            system_message['prefix'] +
            system_message['content'] +
            system_message['postfix']
        )
        
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
            
            if globals.VERBOSE:
                print(f'easy_llama: inference str is \"\"\"{inf_str}\"\"\"')
            return inf_str

        else:

            inf_str = ''

            # Start at most recent message and work backwards up the history
            # excluding system message. Once we exceed thread max_context_length,
            # break without including that message
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

            if globals.VERBOSE:
                print(f'easy_llama: inference str is \"\"\"{inf_str}\"\"\"')
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
            c = 0
            for msg in self.messages:
                c += len(msg['tokens'])
            print(f"track_context: total tokens so far: {c}")

        return output


    def interact(self) -> None:
        """
        Start an interactive chat session using this Thread.

        While text is being generated, press ^C to interrupt the bot.
        Then you have the option to press ENTER to re-roll, or to simply type
        another message.

        At the prompt, press ^C to end the chat session.
        """
        print()
        while True:

            if self.track_context:
                c = 0
                for msg in self.messages:
                    c += len(msg['tokens'])
                print(f"track_context: total tokens so far: {c}")
            
            try:
                prompt = multiline_input("  > ")
            except KeyboardInterrupt:
                print("\n")
                return
            
            print()
            if prompt == "":
                try:
                    output = self.model.stream_print(
                        self.inference_str_from_messages(self.messages),
                        stops=self.format['stops'],
                        sampler=self.sampler,
                        end=''
                    )
                except KeyboardInterrupt:
                    print(' [message not added to history; press ENTER to re-roll]\n')
                    continue
                self.add_message("bot", output)

            else:
                self.add_message("user", prompt)
                try:
                    output = self.model.stream_print(
                        self.inference_str_from_messages(self.messages),
                        stops=self.format['stops'],
                        sampler=self.sampler,
                        end=''
                    )
                except KeyboardInterrupt:
                    print(' [message not added to history; press ENTER to re-roll]\n')
                    continue
                self.add_message("bot", output)

            if output.endswith("\n\n"):
                pass
            elif output.endswith("\n"):
                print()
            else:
                print("\n")


    def reset(self) -> None:
        """Reset the Thread to its original state"""
        self.messages: list[dict] = [
            self.create_message("system", self.format['system_content'])
        ]
        self.model.llama.reset()
    
    def print_stats(self) -> None:
        """Print stats about the context usage in this Thread"""
        thread_len_tokens = self.model.get_length(
            self.inference_str_from_messages(self.messages)
        )
        context_used_percentage = (
            round((thread_len_tokens/self.max_context_length)*100)
        )
        print(f"{thread_len_tokens} / {self.max_context_length} tokens")
        print(f"{context_used_percentage}% of context used")
        print(f"{len(self.messages)} messages")