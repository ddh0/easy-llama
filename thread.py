# thread.py
# Python 3.11.6

"""Submodule containing the Thread class, used for interaction with a Model"""

import time
from model import Model
from samplers import SamplerSettings, DefaultSampling
import globals

class Thread(object):
    def __init__(self,
                 model: Model,
                 format: dict,
                 timestamps: bool = False,
                 amnesiac: bool = False,
                 max_context_length: int = -1,
                 track_context: bool = False,
                 sampler: SamplerSettings = DefaultSampling
                 ):
        
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
                "Thread: format is missing one or more required keys, see " \
                + "easy_llama.blank for an example"
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
        assert role.lower() in ['system', 'user', 'bot'], \
            "create_message: role should be 'system', 'user', or " + \
            f"'bot', not '{role.lower()}'"

        assert isinstance(content, str), \
            f"create_message: content should be str, not {type(content)}"

        if role.lower() == 'system':
            return {
                "prefix": self.format['system_prefix'],
                "content": content if not self.enable_timestamps else
                    time.strftime("[system message at %a %I:%M %p]:")
                     + content,
                "postfix": self.format['system_postfix'],
                "tokens": self.model.llama.tokenize(
                    self.format['system_prefix'].encode() + \
                    content.encode() + \
                    self.format['system_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
                ),
            }
        
        elif role.lower() == 'user':
            return {
                "prefix": self.format['user_prefix'],
                "content": content
                if not self.enable_timestamps
                else time.strftime("[new message sent at %a %I:%M %p]:") + content,
                "postfix": self.format['user_postfix'],
                "tokens": self.model.llama.tokenize(
                    self.format['user_prefix'].encode() + \
                    content.encode() + \
                    self.format['user_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
                ),
            }
        
        elif role.lower() == 'bot':
            return {
                "prefix": self.format['bot_prefix'],
                "content": content
                if not self.enable_timestamps
                else time.strftime('[new message sent at %a %I:%M %p]:') + content,
                "postfix": self.format['bot_postfix'],
                "tokens": self.model.llama.tokenize(
                    self.format['bot_prefix'].encode() + \
                    content.encode() + \
                    self.format['bot_postfix'].encode()
                ),
                "content_tokens": self.model.llama.tokenize(
                    content.encode()
                ),
            }

    def add_message(self, role: str, content: str) -> None:
        """Shorthand for Thread.messages.append(Thread.create_message(...))"""
        self.messages.append(
            self.create_message(
                role=role,
                content=content
            )
        )

    def inference_str_from_messages(self, messages: list[dict]) -> str:

        system_message = messages[0]
        sys_msg_str = (
            system_message['prefix']
            + system_message['content']
            + system_message['postfix']
        )
        
        # in amnesiac threads, the model only sees the system message
        # and the previous message
        if self.amnesiac:
            inf_str = ''
            last_msg = messages[-1]
            last_msg_str = (
                last_msg['prefix']
                + last_msg['content']
                + last_msg['postfix']
                )
            inf_str = (
                sys_msg_str
                + last_msg_str
                + self.format['bot_prefix']
                )
            return inf_str
        
        context_len_budget = self.max_context_length
        context_len_budget -= 3 # little buffer just in case
        context_len_budget -= len(system_message['tokens'])
        context_len_budget -= self.model.get_length(self.format['bot_prefix'])
        if self.enable_timestamps:
            context_len_budget -= self.model.get_length(
                time.strftime("[new message sent at %a %I:%M %p]:")
            ) + 4

        # start at most recent message and work backwards up the history
        # excluding system message. once we exceed thread max_context_length,
        # break without including that message
        inf_str = ''
        for message in reversed(messages[1:]):
            context_len_budget -= len(message['tokens'])
            if context_len_budget <= 0:
                break
            msg_str = (
                message['prefix']
                + message['content']
                + message['postfix']
                )
            inf_str = msg_str + inf_str
        inf_str = sys_msg_str + inf_str
        inf_str += self.format['bot_prefix']
        if self.enable_timestamps:
            inf_str += time.strftime("[new message sent at %a %I:%M %p]:")
        return inf_str


    def send(self, prompt: str) -> str:
        assert isinstance(prompt, str), \
            f"Thread.send: prompt should be str, not {type(prompt)}"

        self.add_message("user", prompt)
        output = self.model.generate(
            self.inference_str_from_messages(self.messages),
            stops=self.format["stops"],
            sampler=self.sampler
        )
        self.add_message("bot", output)

        return output


    def interact(self) -> None:
        print()
        try:
            while True:
                if self.track_context:
                    c = 0
                    for msg in self.messages:
                        c += len(msg['tokens'])
                    print(f"total tokens so far: {c}")
                    last_toks: list[int] = self.messages[-1:][0]['content_tokens']
                    print(f'last msg content tokens: {last_toks}\n')
                prompt = input("  > ")
                print()
                if prompt == "":
                    token_generator = self.model.stream(
                        self.inference_str_from_messages(self.messages),
                        stops=self.format['stops'],
                        sampler=self.sampler
                    )

                    output = ""
                    for i in token_generator:
                        token = i['choices'][0]['text']
                        output += token
                        print(token, end="", flush=True)

                    self.add_message("bot", output)

                else:
                    self.add_message("user", prompt)

                    token_generator = self.model.stream(
                        self.inference_str_from_messages(self.messages),
                        stops=self.format['stops'],
                        sampler=self.sampler
                    )

                    output = ""
                    for i in token_generator:
                        token = i['choices'][0]['text']
                        output += token
                        print(token, end="", flush=True)

                    self.add_message("bot", output)

                if output.endswith("\n\n"):
                    pass
                elif output.endswith("\n"):
                    print()
                else:
                    print("\n")

        except KeyboardInterrupt:
            print("\n")
            return

    def reset(self) -> None:
        self.messages: list[dict] = [
            self.create_message("system", self.format['system_content'])
        ]
        self.model.llama.reset()
    
    def print_stats(self) -> None:
        thread_len_tokens = self.model.get_length(
            self.inference_str_from_messages(self.messages)
        )
        context_used_percentage = (
            round((thread_len_tokens/self.max_context_length)*100)
        )
        print(f"{thread_len_tokens} / {self.max_context_length} tokens")
        print(f"{context_used_percentage}% of context used")
        print(f"{len(self.messages)} messages")