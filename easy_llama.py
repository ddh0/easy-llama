# easy_llama.py
# Python 3.11.6

"""
Simple, on-device text inference in Python

for complete and up-to-date information, see https://github.com/ddh0/easy-llama
"""

# TODO: Switch to low-level API ??
# TODO: message-based context length handling
# TODO: Model.next_candidates() -> list[str]
# TODO: Automatic detection of METAL, CUDA, OpenBLAS/CPU and set NUM_GPU_LAYERS accordingly
# TODO: Text streaming in Thread.interact?

import os
import sys
import struct
from enum import IntEnum

MAX_LEN_TOKENS = 256        # Max length of each generation in tokens
NUM_GPU_LAYERS = 1          # Leave at 1 for Apple Silicon, tweak for CUDA and ROCm, set to 0 for OpenBLAS / CPU
VERBOSE = False             # Print all backend information as it occurs
SEED = -1                   # Leave at -1 for random seed
SUPPRESS_WARNINGS = False   # Warnings are rare and contain helpful information


class _GGUF_READER:
    """
    Credit to oobabooga for most of the code in this class, which is
    licensed under the AGPL-3.0 license
    """
    class GGUFValueType(IntEnum):
        UINT8 = 0
        INT8 = 1
        UINT16 = 2
        INT16 = 3
        UINT32 = 4
        INT32 = 5
        FLOAT32 = 6
        BOOL = 7
        STRING = 8
        ARRAY = 9
        UINT64 = 10
        INT64 = 11
        FLOAT64 = 12


    _simple_value_packing = {
        GGUFValueType.UINT8: "<B",
        GGUFValueType.INT8: "<b",
        GGUFValueType.UINT16: "<H",
        GGUFValueType.INT16: "<h",
        GGUFValueType.UINT32: "<I",
        GGUFValueType.INT32: "<i",
        GGUFValueType.FLOAT32: "<f",
        GGUFValueType.UINT64: "<Q",
        GGUFValueType.INT64: "<q",
        GGUFValueType.FLOAT64: "<d",
        GGUFValueType.BOOL: "?",
    }

    value_type_info = {
        GGUFValueType.UINT8: 1,
        GGUFValueType.INT8: 1,
        GGUFValueType.UINT16: 2,
        GGUFValueType.INT16: 2,
        GGUFValueType.UINT32: 4,
        GGUFValueType.INT32: 4,
        GGUFValueType.FLOAT32: 4,
        GGUFValueType.UINT64: 8,
        GGUFValueType.INT64: 8,
        GGUFValueType.FLOAT64: 8,
        GGUFValueType.BOOL: 1,
    }


    def get_single(value_type, file):
        if value_type == _GGUF_READER.GGUFValueType.STRING:
            value_length = struct.unpack("<Q", file.read(8))[0]
            value = file.read(value_length)
            try:
                value = value.decode('utf-8')
            except:
                pass
        else:
            type_str = _GGUF_READER._simple_value_packing.get(value_type)
            bytes_length = _GGUF_READER.value_type_info.get(value_type)
            value = struct.unpack(type_str, file.read(bytes_length))[0]

        return value


    def load_metadata(fname):
        metadata = {}
        with open(fname, 'rb') as file:
            GGUF_MAGIC = struct.unpack("<I", file.read(4))[0]
            GGUF_VERSION = struct.unpack("<I", file.read(4))[0]
            ti_data_count = struct.unpack("<Q", file.read(8))[0]
            kv_data_count = struct.unpack("<Q", file.read(8))[0]

            if GGUF_VERSION == 1:
                raise Exception('You are using an outdated GGUF, please download a new one.')

            for i in range(kv_data_count):
                key_length = struct.unpack("<Q", file.read(8))[0]
                key = file.read(key_length)

                value_type = _GGUF_READER.GGUFValueType(struct.unpack("<I", file.read(4))[0])
                if value_type == _GGUF_READER.GGUFValueType.ARRAY:
                    ltype = _GGUF_READER.GGUFValueType(struct.unpack("<I", file.read(4))[0])
                    length = struct.unpack("<Q", file.read(8))[0]
                    for j in range(length):
                        _ = _GGUF_READER.get_single(ltype, file)
                else:
                    value = _GGUF_READER.get_single(value_type, file)
                    metadata[key.decode()] = value

        return metadata


class _suppress_if_not_verbose(object):
    """
    Suppress console output from llama.cpp if easy_llama.VERBOSE is False

    Changing VERBOSE inside the WITH block may result in stdout and stderr
    being stuck to /dev/null, or other undefined behaviour

    See https://github.com/abetlen/llama-cpp-python/issues/478
    """

    def __enter__(self):
        if not VERBOSE:
            self.outnull_file = open(os.devnull, 'w')
            self.errnull_file = open(os.devnull, 'w')

            self.old_stdout_fileno_undup    = sys.stdout.fileno()
            self.old_stderr_fileno_undup    = sys.stderr.fileno()

            self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
            self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr

            os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
            os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

            sys.stdout = self.outnull_file        
            sys.stderr = self.errnull_file
            return self
        
        if VERBOSE:
            return self


    def __exit__(self, *_):
        if not VERBOSE:        
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr

            os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
            os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

            os.close ( self.old_stdout_fileno )
            os.close ( self.old_stderr_fileno )

            self.outnull_file.close()
            self.errnull_file.close()


with _suppress_if_not_verbose():
        import llama_cpp


def _print_warning(text: str) -> str:
    """
    Print text to stderr unless SUPPRESS_WARNINGS is True,
    prefixing with 'easy_llama: '
    """

    if not SUPPRESS_WARNINGS:
        print('easy_llama: ' + text, file=sys.stderr)


class Model(object):
    """
    Abstraction of a llama model
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize a llama model from a file.

        Model must be in GGUF format (but it does not have to be quantized).

        easy_llama will automatically determine the model's trained
        context length from the GGUF metadata.
        """

        assert isinstance(model_path, str), 'model_path should be a string, not %s' % type(model_path)
        assert not os.path.isdir(model_path), 'the given model_path \'%s\' is a directory, not a file' % model_path
        assert os.path.exists(model_path), 'the given model_path \'%s\' does not exist' % model_path
        
        self.metadata: dict = _GGUF_READER.load_metadata(model_path)
        
        assert isinstance(self.metadata['llama.context_length'], int), \
               'GGUF metadata reports that context_length is not an integer'
        if self.metadata['llama.context_length'] < 2048:
               _print_warning('GGUF metadata reports an unusually small native context length (%s)' % \
                             self.metadata['llama.context_length'])
        if self.metadata['llama.context_length'] > 131072: # 2^17 or ~128k tokens
               _print_warning('GGUF metadata reports an unusually large native context length (%s)' % \
                             self.metadata['llama.context_length'])

        self.context_length = self.metadata['llama.context_length'] # n_ctx_train

        with _suppress_if_not_verbose():
            self._internal_model: llama_cpp.Llama = llama_cpp.Llama(model_path=model_path,
                                                                    n_ctx=self.context_length,
                                                                    n_gpu_layers=NUM_GPU_LAYERS,
                                                                    seed=SEED,
                                                                    use_mmap=False,
                                                                    use_mlock=False,
                                                                    logits_all=True,
                                                                    n_batch=512,
                                                                    n_threads=max(os.cpu_count()//2, 1),
                                                                    n_threads_batch=os.cpu_count(),
                                                                    mul_mat_q=True,
                                                                    verbose=VERBOSE)
            
            # The first inference on a freshly loaded model has a little extra delay
            # This gets that out of the way before the user's first actual generation
            self._internal_model.create_completion('',
                                                   max_tokens=1,
                                                   top_p=0,
                                                   top_k=1,
                                                   stream=False,
                                                   stop=None,
                                                   repeat_penalty=1)
    

    def trim(self, text: str, overwrite: str=None) -> str:
        """
        Trim the given text to the context length of this model,
        leaving room for two extra tokens.
        
        Optionally overwrite the oldest tokens with the text given in the 'overwrite'
        parameter, which is useful for keeping the system prompt in context.

        Does nothing if the text is equal to or shorter than (context_length - 2).
        """
        trim_length = self.context_length - 2

        tokens_list = self._internal_model.tokenize(text.encode('utf-8', errors='ignore'))
        if len(tokens_list) <= trim_length:
            return text
        if len(tokens_list) > trim_length and overwrite is None:
            # Cut to context length
            tokens_list = tokens_list[-trim_length:]
            return self._internal_model.detokenize(tokens_list).decode('utf-8', errors='ignore')
        if len(tokens_list) > self.context_length and overwrite is not None:
            # Cut to context length and overwrite the oldest tokens with overwrite
            tokens_list = tokens_list[-trim_length:]
            overwrite_tokens = self._internal_model.tokenize(overwrite.encode('utf-8', errors='ignore'))
            tokens_list[0:len(overwrite_tokens)] = overwrite_tokens
            return self._internal_model.detokenize(tokens_list).decode('utf-8', errors='ignore')
    

    def get_length(self, text: str) -> int:
        """
        Return the length of the given text in tokens,
        according to this model.
        """
        return len(self._internal_model.tokenize(text.encode('utf-8', errors='ignore')))


    def greedy(self, prompt: str, stops: list[str] | str | None=None) -> str:
        """
        Given a prompt, return a generated string using greedy decoding,
        where the most likely token is always chosen.

        The following parameter is optional:
        
        stops: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert isinstance(stops, (list, str, type(None))), 'stops should be list, str, or None'

        if type(stops) is list:
            for item in stops:
                assert isinstance(item, str), "some item in stops list is not of type str"

        if MAX_LEN_TOKENS > self.context_length:
           _print_warning("MAX_LEN_TOKENS is greater than this model's context length")
        
        with _suppress_if_not_verbose():
            return self._internal_model.create_completion(prompt,
                                                          max_tokens=MAX_LEN_TOKENS,
                                                          top_p=0,
                                                          top_k=1,
                                                          stream=False,
                                                          stop=stops,
                                                          repeat_penalty=1
                                                          )['choices'][0]['text']


    def generate_low(self, prompt: str, stops: list[str] | str | None=None) -> str:
        """
        Given a prompt, return a generated string using constrastive search
        with a low alpha value. This leads to more predictable results.

        For more information on contrastive search, see here:
        https://huggingface.co/blog/introducing-csearch

        The following parameter is optional:
        
        stops: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert isinstance(stops, (list, str, type(None))), 'stops should be list, str, or None'

        if isinstance(stops, list):
            for item in stops:
                assert isinstance(item, str), "some item in stops list is not of type str"

        if MAX_LEN_TOKENS > self.context_length:
           _print_warning("MAX_LEN_TOKENS is greater than this model's context length")
        
        with _suppress_if_not_verbose():
            return self._internal_model.create_completion(prompt,
                                                          max_tokens=MAX_LEN_TOKENS,
                                                          top_k=4,
                                                          presence_penalty=0.2,
                                                          stream=False,
                                                          stop=stops,
                                                          )['choices'][0]['text']


    def generate_medium(self, prompt: str, stops: list[str] | str | None=None) -> str:
        """
        Given a prompt, return a generated string using constrastive search
        with a moderate alpha value. This is the method easy_llama recommends
        for most cases.

        For more information on contrastive search, see here:
        https://huggingface.co/blog/introducing-csearch

        The following parameter is optional:
        
        stops: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert isinstance(stops, (list, str, type(None))), 'stops should be list, str, or None'

        if isinstance(stops, list):
            for item in stops:
                assert isinstance(item, str), "some item in stops list is not of type str"

        if MAX_LEN_TOKENS > self.context_length:
           _print_warning("MAX_LEN_TOKENS is greater than this model's context length")
        
        with _suppress_if_not_verbose():
            return self._internal_model.create_completion(prompt,
                                                          max_tokens=MAX_LEN_TOKENS,
                                                          top_k=4,
                                                          presence_penalty=0.5,
                                                          stream=False,
                                                          stop=stops,
                                                          )['choices'][0]['text']


    def generate_high(self, prompt: str, stops: list[str] | str | None=None) -> str:
        """
        Given a prompt, return a generated string using constrastive search
        with a high alpha value. This leads to more creative results.

        For more information on contrastive search, see here:
        https://huggingface.co/blog/introducing-csearch

        The following parameter is optional:
        
        stops: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert isinstance(stops, (list, str, type(None))), 'stops should be list, str, or None'

        if isinstance(stops, list):
            for item in stops:
                assert isinstance(item, str), "some item in stops list is not of type str"

        if MAX_LEN_TOKENS > self.context_length:
           _print_warning("MAX_LEN_TOKENS is greater than this model's context length")
        
        with _suppress_if_not_verbose():
            return self._internal_model.create_completion(prompt,
                                                          max_tokens=MAX_LEN_TOKENS,
                                                          top_k=4,
                                                          presence_penalty=0.7,
                                                          stream=False,
                                                          stop=stops,
                                                          )['choices'][0]['text']
    

    def next_candidates(self, prompt: str) -> list[str]:
        """
        Given a prompt, return a list of the most likely next tokens,
        in descending order.
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)


class _Message(object):

    def __init__(self, role: str, text: str) -> None:
        assert isinstance(text, str), 'Message: text should be str'
        assert role.lower() in ['system', 'user', 'bot'], 'Message: role must be \'system\', \'user\', or \'bot\''
        self.role = role.lower()
        self.text = text


class Thread(object):
    """
    Provide functionality to facilitate conversation with a model
    (i.e. remembering past messages.)

    You should specify a format. easy_llama includes several built-in
    formats within easy_llama.Formats. See that class docstring for details.

    Optionally, you can set the 'setting' parameter to one of
    'greedy', 'low', 'medium', 'high'.
    
    Default is 'medium'.

    easy_llama.Thread currently does not support double-messaging,
    i.e if user sends a message, the next message must be from the bot,
    and vice-versa.

    Remember to set easy_llama.MAX_LEN_TOKENS to suit your needs.
    """

    def __init__(self, model: Model, format=None, setting: str='medium') -> None:

        assert isinstance(model, Model), 'Thread: model should be an instance of easy_llama.Model'
        if format is not None:
            assert hasattr(format, 'system_str'), 'Thread: format is missing required attribute system_str'
            assert hasattr(format, 'user_prefix_str'), 'Thread: format is missing required attribute user_prefix_str'
            assert hasattr(format, 'bot_prefix_str'), 'Thread: format is missing required attribute bot_prefix_str'
            assert hasattr(format, 'bot_postfix_str'), 'Thread: format is missing required attribute bot_postfix_str'
            assert hasattr(format, 'stops'), 'Thread: format is missing required attribute stops'
            self.format = format
        else:
            _print_warning("Thread: no format was provided, so none will be used. \
                           unless you're using a base model, this will affect the quality of outputs")
            self.format = Formats.Blank
        assert isinstance(setting, str), 'Thread: setting should be str'
        assert setting.lower() in ['greedy', 'low', 'medium', 'high'], \
            'Thread: setting should be one of \'greedy\', \'low\', \'medium\', \'high\''

        self.model: Model = model
        self.setting = setting.lower()

        # Calculate these only once and save them for later
        self.system_len = model.get_length(self.format.system_str)
        self.user_prefix_len = model.get_length(self.format.user_prefix_str)
        self.bot_prefix_len = model.get_length(self.format.bot_prefix_str)
        self.bot_postfix_len = model.get_length(self.format.bot_postfix_str)

        # list will contain all messages in thread regardless of context length,
        # with the system message at the start and the newest message at the end
        self.messages = [_Message(role='system', text=self.format.system_str)]
    

    def get_inference_str(self) -> str:
        """
        Using the list of messages, return a string suitable for text inference.

        Old messages will be discarded as needed to fit the model's context length,
        while always keeping the system message first.
        """
        assert self.messages[0].role == 'system', 'Thread: get_inference_str: first message is not system message'
        inference_str = self.messages[0].text

        # iterate over messages from newest to oldest (i.e. in reverse)
        # 
        # as you go, subtract tokens from (context length - system message length)
        # and build the inference string in reverse
        # once you hit 1, break and return string without including the current message

        previous_message_role = 'system'
        for message in self.messages[1:]: # starting at first non-system message
            assert message.role != 'system', 'easy_llama.Thread only supports one system message'
            
            if message.role == 'user':
                if previous_message_role == 'system':
                    # first message from user, directly after system prompt
                    inference_str += self.format.user_prefix_str + message.text
                elif previous_message_role == 'bot':
                    # last message was from bot, this message is from user
                    inference_str += self.format.bot_postfix_str + self.format.user_prefix_str + message.text
                previous_message_role = 'user'
            
            elif message.role == 'bot':
                inference_str += self.format.bot_prefix_str + message.text
                previous_message_role = 'bot'
        
        # now inference_str ends with the text of the last message
        
        inference_str += self.format.bot_prefix_str

        return self.model.trim(inference_str, overwrite=self.format.system_str)
    
    
    def send(self, prompt: str) -> str:
        """
        Send a message in this Thread
        """
        assert isinstance(prompt, str), 'Thread.send: prompt should be str'
        # do not allow empty input, as that would mess up the format
        assert prompt != '', 'Thread.send: empty prompts are not allowed in threads'
        self.messages.append(_Message(role='user', text=prompt))
        if self.setting == 'greedy':
            output = self.model.greedy(self.get_inference_str(), stops=self.format.stops)
        elif self.setting == 'low':
            output = self.model.generate_low(self.get_inference_str(), stops=self.format.stops)
        elif self.setting == 'medium':
            output = self.model.generate_medium(self.get_inference_str(), stops=self.format.stops)
        elif self.setting == 'high':
            output = self.model.generate_high(self.get_inference_str(), stops=self.format.stops)
        self.messages.append(_Message(role='bot', text=output))
        return output
    

    def interact(self) -> None:
        """
        Start an interactive chat session from this Thread.
        Designed for use in interactive python shells.
        Quit using KeyboardInterrupt (CTRL+C).
        """
        print()
        try:
            while True:
                prompt = input('  > ')
                print() # Put the cursor where the generated text is about to appear
                if prompt == '':
                    _print_warning('empty prompts are not allowed in threads\n')
                    continue
                else:
                    self.messages.append(_Message(role='user', text=prompt))
                    if self.setting == 'greedy':
                        output = self.model.greedy(self.get_inference_str(), stops=self.format.stops)
                    elif self.setting == 'low':
                        output = self.model.generate_low(self.get_inference_str(), stops=self.format.stops)
                    elif self.setting == 'medium':
                        output = self.model.generate_medium(self.get_inference_str(), stops=self.format.stops)
                    elif self.setting == 'high':
                        output = self.model.generate_high(self.get_inference_str(), stops=self.format.stops)
                    self.messages.append(_Message(role='bot', text=output))

                # Strip all leading and trailing spaces and newlines from displayed output
                # This is only done for Thread.interact()
                while output.startswith(' ') or output.startswith('\n'):
                    output = output[1:]
                while output.endswith('\n') or output.endswith(' '):
                    output = output[:-1]
                print(output + '\n')
        
        except KeyboardInterrupt:
            print('\n')
            return


    def reset(self) -> None:
        """
        Erase all messages, leaving only the system prompt
        """
        self.messages = [_Message(role='system', text=self.format.system_str)]


class Formats:
    """
    This class contains several subclasses, each one corresponding
    to a common prompt format, such as Alpaca, Llama2, etc.

    These can be used in two ways. Firstly, each format class has a
    .wrap() method, which takes a single string and returns the string with
    all the necessary formatting required for one-off generations.

    Second, when creating a Thread (easy_llama.Thread), you can specify one
    of these formats, and all messages in the thead will be appropriately
    formatted.

    For example:

    `Llama2 = easy_llama.Model('./Llama-2-Chat.gguf')`

    `MyThread = easy_llama.Thread(model=Llama2, format=easy_llama.Formats.Llama2)`

    Note that using a format that does not match your model might appear to work,
    but this has the potential to significantly affect the quality of generations.

    You can create your own format class, and it does not need to inherit from
    this class. However, it must have the following attributes:
    - system_str: str
    - user_prefix_str: str
    - bot_prefix_str: str
    - bot_postfix_str: str
    - stops: list[str] | str | None

    Additionally, it should have the .wrap() method as described above.
    """

    class Blank:
        system_str = ''
        user_prefix_str = ''
        bot_prefix_str = ''
        bot_postfix_str = ''
        stops = None

        def wrap(text: str) -> str:
            return text


    class Alpaca:
        system_str = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
        user_prefix_str = '### Instruction:\n'
        bot_prefix_str = '\n\n### Response:\n'
        bot_postfix_str = '\n\n'
        stops = ['###', '\nInstruction:']

        def wrap(text: str) -> str:
            return Formats.Alpaca.system_str + Formats.Alpaca.user_prefix_str + \
                   text + Formats.Alpaca.bot_prefix_str


    class Llama2:
        system_str = '<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n'
        user_prefix_str = ''
        bot_prefix_str = ' [/INST] '
        bot_postfix_str = '</s><s>[INST] '
        stops = ['</s>', '[INST]']

        def wrap(text: str) -> str:
            return Formats.Llama2.system_str + Formats.Llama2.user_prefix_str + \
                   text + Formats.Llama2.bot_prefix_str


    class Vicuna:
        system_str = ''
        user_prefix_str = 'USER: '
        bot_prefix_str = '\nASSISTANT: '
        bot_postfix_str = '\n'
        stops = ['</s>', '\nUSER:']

        def wrap(text: str) -> str:
            return Formats.Vicuna.system_str + Formats.Vicuna.user_prefix_str + \
                   text + Formats.Vicuna.bot_prefix_str


    class Airoboros:
        system_str = '<s>[INST] <<SYS>>\nYou are Airoboros, a helpful, unbiased, uncensored AI assistant.\n<</SYS>>\n\n'
        user_prefix_str = ''
        bot_prefix_str = ' [/INST] '
        bot_postfix_str = '</s><s>[INST] '
        stops = ['</s>', '[INST]']

        def wrap(text: str) -> str:
            return Formats.Airoboros.system_str + Formats.Airoboros.user_prefix_str + \
                   text + Formats.Airoboros.bot_prefix_str


    class MistralInstruct:
        # This system prompt is from Mistral 7B's official release PDF on arXiv
        # The other format strings are from Mistral 7B's official HF page
        system_str = '<s>Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. '
        user_prefix_str = '[INST] '
        bot_prefix_str = ' [/INST] '
        bot_postfix_str = '</s> '
        stops = '</s>'

        def wrap(text: str) -> str:
            return Formats.MistralInstruct.system_str + Formats.MistralInstruct.user_prefix_str + \
                   text + Formats.MistralInstruct.bot_prefix_str


    class MistralOrca:
        system_str = '<|im_start|>system\nYou are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!\n<|im_end|>\n'
        user_prefix_str = '<|im_start|>user\n'
        bot_prefix_str = '<|im_end|>\n<|im_start|>assistant\n'
        bot_postfix_str = '<|im_end|>\n'
        stops = ['<|im_end|>', '<|im_start|>']

        def wrap(text: str) -> str:
            return Formats.MistralOrca.system_str + Formats.MistralOrca.user_prefix_str + \
                   text + Formats.MistralOrca.bot_prefix_str


    class Jackalope:
        system_str = '<|im_start|>system\nYou are JackalopeAI, a large language model trained by OpenAccess AI Collective. Write out your reasoning step-by-step to be sure you get the right answers!\n<|im_end|>\n'
        user_prefix_str = '<|im_start|>user\n'
        bot_prefix_str = '<|im_end|>\n<|im_start|>assistant\n'
        bot_postfix_str = '<|im_end|>\n'
        stops = ['<|im_end|>', '<|im_start|>']

        def wrap(text: str) -> str:
            return Formats.Jackalope.system_str + Formats.Jackalope.user_prefix_str + \
                   text + Formats.Jackalope.bot_prefix_str


    class Dolphin:
        system_str = '<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n'
        user_prefix_str = '<|im_start|>user\n'
        bot_prefix_str = '<|im_end|>\n<|im_start|>assistant\n'
        bot_postfix_str = '<|im_end|>\n'
        stops = ['<|im_end|>', '<|im_start|>']

        def wrap(text: str) -> str:
            return Formats.Dolphin.system_str + Formats.Dolphin.user_prefix_str + \
                   text + Formats.Dolphin.bot_prefix_str


    class Samantha:
        system_str = '<|im_start|>system\nYou are a caring and empathetic sentient AI companion named Samantha.<|im_end|>\n'
        user_prefix_str = '<|im_start|>user\n'
        bot_prefix_str = '<|im_end|>\n<|im_start|>assistant\n'
        bot_postfix_str = '<|im_end|>\n'
        stops = ['<|im_start|>', '<|im_end|>']

        def wrap(text: str) -> str:
            return Formats.Samantha.system_str + Formats.Samantha.user_prefix_str + \
                   text + Formats.Samantha.bot_prefix_str


    class Guanaco:
        system_str = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions.\n\n'
        user_prefix_str = '### Human: '
        bot_prefix_str = '\n### Assistant: '
        bot_postfix_str = '\n'
        stops = ['###', '\nHuman:']

        def wrap(text: str) -> str:
            return Formats.Guanaco.system_str + Formats.Guanaco.user_prefix_str + \
                   text + Formats.Guanaco.bot_prefix_str


    class OrcaMini:
        system_str = '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n'
        user_prefix_str = '### User:\n'
        bot_prefix_str = '\n\n### Assistant:\n'
        bot_postfix_str = '\n\n'
        stops = ['###', '\nUser:']

        def wrap(text: str) -> str:
            return Formats.OrcaMini.system_str + Formats.OrcaMini.user_prefix_str + \
                   text + Formats.OrcaMini.bot_prefix_str


    class Zephyr:
        system_str = '<|system|>\nYou are Zephyr, an artificial intelligence assistant. You are helpful and polite.</s>\n'
        user_prefix_str = '<|user|>\n'
        bot_prefix_str = '</s>\n<|assistant|>\n'
        bot_postfix_str = '</s>\n'
        stops = ['</s>', '<|user|>', '<|assistant|>']

        def wrap(text: str) -> str:
            return Formats.Zephyr.system_str + Formats.Zephyr.user_prefix_str + \
                   text + Formats.Zephyr.bot_prefix_str


    class Metharme:
        # This is used for Pygmalion models, and is different from ChatML
        system_str = '<|system|>You are an artificial intelligence assistant. You are helpful and polite.'
        user_prefix_str = '<|user|>'
        bot_prefix_str = '<|model|>'
        bot_postfix_str = ''
        stops = ['<|user|>', '<|model|>']

        def wrap(text: str) -> str:
            return Formats.Metharme.system_str + Formats.Metharme.user_prefix_str + \
                   text + Formats.Metharme.bot_prefix_str
