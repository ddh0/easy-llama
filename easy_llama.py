# easy_llama.py
# Python 3.11.6

"""
Natural text generation in Python, made easy
---------------------------------------
https://github.com/ddh0/easy-llama/
"""

# TODO: Thread.add_message as shorthand for T.messages.append(T.create_message) ?
# TODO: functions to transfer a list of messages between disk / models, handle token count
# TODO: function to do summarization to compress context ?
# TODO: verify message-based context length handling works
# TODO: Model.next_candidates() -> list[str]

import os
import sys
import time
import struct
from typing import Generator
from enum import IntEnum

# this is set per pip package
# 'metal' | 'cuda' | 'rocm' | 'cpu' 
# NOTE TO END USER: DO NOT TOUCH!
_backend = None

# Max length of each generation in tokens
MAX_LEN_TOKENS: int = 2048

# Do not suppress llama.cpp output
VERBOSE: bool = False

# Leave at -1 for random seed
SEED: int = -1

# Warnings are infrequent and contain helpful information
SUPPRESS_WARNINGS: bool = False

class _GGUF_READER:
    """
    Peek at file header for GGUF metadata

    Raise ValueError if file is not GGUF or is outdated

    Credit to oobabooga for the parts of the code in this class

    Format spec: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
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

    def load_metadata(fname) -> dict:
        metadata = {}
        with open(fname, 'rb') as file:
            MAGIC_NUM = file.read(4)
            GGUF_VERSION = struct.unpack("<I", file.read(4))[0]
            ti_data_count = struct.unpack("<Q", file.read(8))[0]
            kv_data_count = struct.unpack("<Q", file.read(8))[0]
            
            if MAGIC_NUM != b'GGUF':
                raise ValueError(f'easy_llama: Your model file is not a valid GGUF file \
                    (magic number mismatch, got {MAGIC_NUM}, expected b\'GGUF\')')

            if GGUF_VERSION == 1:
                raise ValueError("easy_llama: Your model file reports GGUF version 1, but only \
                    version 2 is supported. Re-convert your model or download a newer version.")

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
    Suppress stdout and stderr if easy_llama.VERBOSE is False

    This prevents llama.cpp's very detailed output from being displayed

    Changing VERBOSE inside the WITH block may result in stdout and stderr
    being stuck to /dev/null, or other undefined behaviour

    See https://github.com/abetlen/llama-cpp-python/issues/478
    """

    def __enter__(self):
        if not VERBOSE:
            self.outnull_file = open(os.devnull, "w")
            self.errnull_file = open(os.devnull, "w")

            self.old_stdout_fileno_undup = sys.stdout.fileno()
            self.old_stderr_fileno_undup = sys.stderr.fileno()

            self.old_stdout_fileno = os.dup(sys.stdout.fileno())
            self.old_stderr_fileno = os.dup(sys.stderr.fileno())

            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr

            os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
            os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

            sys.stdout = self.outnull_file
            sys.stderr = self.errnull_file
            return self
        
        if VERBOSE:
            return self

    def __exit__(self, *_):
        if not VERBOSE:        
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr

            os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
            os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

            os.close(self.old_stdout_fileno)
            os.close(self.old_stderr_fileno)

            self.outnull_file.close()
            self.errnull_file.close()

def _print_warning(text: str) -> str:
    if not SUPPRESS_WARNINGS:
        print('easy_llama: warning:', text, file=sys.stderr)

with _suppress_if_not_verbose():
        import llama_cpp

class Model(object):
    """
    A high-level abstraction of a llama model

    This is just a brief overview of easy_llama.Model.
    To see a full description of each method and its parameters,
    call help(Model), or see the relevant docstring.

    The following methods are available:
    - .generate(): return a string generated with Contrastive Search,
    which generates more human-like text
    - .get_length(): return the length of a given string in tokens
    - .trim(): trim a given string to this model's context length
    - .next_candidates(): return a list of candidates for the most likely next token

    The following attributes are available:
    - .metadata: the GGUF metadata read from the model file, dict
    - .context_lenth: the native context length of the model in tokens, int
    - .llama: the raw llama_cpp.Llama instance
    """

    def __init__(self, model_path: str):
        """
        Given the path to a GGUF file, create a Model instance

        The model must be in GGUF format.

        easy_llama will automatically determine the model's trained
        context length from the GGUF metadata.
        """

        if _backend == 'metal':
            # Must be 1
            NUM_GPU_LAYERS: int = 1
            MUL_MAT_Q = True
        elif _backend == 'cuda':
            # Tweak between 1 and 10000000
            # Set to -1 to move all layers to CUDA
            NUM_GPU_LAYERS: int = 1
            MUL_MAT_Q = False
        elif _backend == 'rocm':
            # Tweak between 1 and 10000000
            # Set to -1 to move all layers to ROCm
            NUM_GPU_LAYERS: int = 1
            MUL_MAT_Q = True
        elif _backend == 'cpu':
            # Must be 0
            NUM_GPU_LAYERS: int = 0
            MUL_MAT_Q = True
        else:
            raise RuntimeError(f'easy_llama: _backend \'{_backend}\' is invalid. this is unexpected')

        assert isinstance(model_path, str), f'model_path should be a string, not {type(model_path)}'
        assert not os.path.isdir(model_path), f'the given model_path \'{model_path}\' is a directory, not a file'
        assert os.path.exists(model_path), f'the given model_path \'{model_path}\' does not exist'
        
        self.metadata = _GGUF_READER.load_metadata(model_path)
        
        if self.metadata['llama.context_length'] < 2048:
               _print_warning(f"GGUF metadata reports an unusually small native context length ({self.metadata['llama.context_length']})")
        if self.metadata['llama.context_length'] > 131072: # 2^17 or ~128k tokens
               _print_warning(f"GGUF metadata reports an unusually large native context length ({self.metadata['llama.context_length']})")

        self.context_length = self.metadata['llama.context_length'] # n_ctx_train

        with _suppress_if_not_verbose():
            self.llama: llama_cpp.Llama = \
                llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=self.context_length, # Read from GGUF
                    n_gpu_layers=NUM_GPU_LAYERS,
                    seed=SEED,
                    use_mmap=False, # Faster inference but model must fit in memory
                    use_mlock=True, # Force the model to stay in memory
                    logits_all=False,
                    n_batch=os.cpu_count() * 64, # experimental, but likely sensible
                    n_threads=max(os.cpu_count()//2, 1), # optimal if == physical cores or P cores
                    n_threads_batch=os.cpu_count(), # always optimal
                    mul_mat_q=MUL_MAT_Q,
                    verbose=VERBOSE
                )

            print(f'----------------------------------------------')
            print(f'easy_llama: _backend       == {_backend}')
            print(f'easy_llama: NUM_GPU_LAYERS == {NUM_GPU_LAYERS}')
            print(f'easy_llama: MUL_MAT_Q      == {MUL_MAT_Q}')
            print()

    def trim(self, text: str, overwrite: str=None) -> str:
        """
        Trim the given text to the context length of this model,
        leaving room for three extra tokens.
        
        Optionally overwrite the oldest tokens with the text given in the 'overwrite'
        parameter, which is useful for keeping the system prompt in context.

        Does nothing if the text is equal to or shorter than (context_length - 3).
        """
        trim_length = self.context_length - 3
        tokens_list = self.llama.tokenize(text.encode('utf-8', errors='ignore'))

        if len(tokens_list) <= trim_length:
            return text
        
        if len(tokens_list) > trim_length and overwrite is None:
            # Cut to context length
            tokens_list = tokens_list[-trim_length:]
            return self.llama.detokenize(tokens_list).decode('utf-8', errors='ignore')
        
        if len(tokens_list) > self.context_length and overwrite is not None:
            # Cut to context length and overwrite the oldest tokens with overwrite
            tokens_list = tokens_list[-trim_length:]
            overwrite_tokens = self.llama.tokenize(overwrite.encode('utf-8', errors='ignore'))
            tokens_list[0:len(overwrite_tokens)] = overwrite_tokens
            return self.llama.detokenize(tokens_list).decode('utf-8', errors='ignore')
    
    def get_length(self, text: str) -> int:
        """
        Return the length of the given text in tokens,
        according to this model.
        """
        return len(self.llama.tokenize(text.encode('utf-8', errors='ignore')))

    def generate(self, prompt: str, stops: list[str] | None=None) -> str:
        """
        Given a prompt, return a generated string.

        The following parameter is optional:
        
        stops: list[str] | None: a list of strings at which to end the generation early
        """

        assert isinstance(prompt, str), f'prompt should be string, not {type(prompt)}'
        if isinstance(stops, list):
            for item in stops:
                assert isinstance(item, str), f'item {item} in stops list is not a string'
        else:
            assert stops is None, f'stops should be list[str] or None, not {type(stops)}'

        if MAX_LEN_TOKENS > self.context_length:
           _print_warning("MAX_LEN_TOKENS is greater than this model's context length, expect poor results")
        
        # see https://github.com/facebookresearch/llama/issues/217
        if prompt[-1] == ' ':
            _print_warning('prompt has trailing whitespace, this may reduce the quality of generations\n')

        # 'presence_penalty' is the alpha value used in contrastive search
        # easy_llama's official recommenedation for most purposes is 0.45 - 0.6
        with _suppress_if_not_verbose():
            return self.llama.create_completion(
                prompt,
                max_tokens=MAX_LEN_TOKENS,
                top_k=4,
                presence_penalty=0.5,
                stop=stops,
            )['choices'][0]['text']

    def stream(self, prompt: str, stops: list[str] | None=None) -> Generator:
        """
        Given a prompt, return a generator that yields dicts containing tokens.

        To get the token string itself, subscript the dict with:

        `['choices'][0]['text']`

        The following parameter is optional:
        
        stops: list[str] | None: a list of strings at which to end the generation early
        """

        assert isinstance(prompt, str), f'prompt should be string, not {type(prompt)}'
        if isinstance(stops, list):
            for item in stops:
                assert isinstance(item, str), f'item {item} in stops list is not a string'
        else:
            assert stops is None, f'stops should be list[str] or None, not {type(stops)}'

        if MAX_LEN_TOKENS > self.context_length:
           _print_warning("MAX_LEN_TOKENS is greater than this model's context length, expect poor results")
        
        # see https://github.com/facebookresearch/llama/issues/217
        if prompt[-1] == ' ':
            _print_warning('prompt has trailing whitespace, this may reduce the quality of generations\n')

        # 'presence_penalty' is the alpha value used in contrastive search
        # easy_llama's official recommenedation for most purposes is 0.45 - 0.6
        with _suppress_if_not_verbose():
            return self.llama.create_completion(
                prompt,
                max_tokens=MAX_LEN_TOKENS,
                top_k=4,
                stream=True,
                presence_penalty=0.5,
                stop=stops,
            )
 
    def next_candidates(self, prompt: str, k: int) -> list[str]:
        """
        Given prompt (str) and k (int), return a sorted list of the
        top k candidates for most likely next token
        """

        # Llama.eval(tokens_list_ints)
        # Llama.scores is a list of lists (basically)
        # scores[0] is a list of numbers like -4.4852424...
        # the number (float) indicates the probability of the token

        # len(raw_scores) == n_vocab
        # note that eos token is not always the last token in the vocab,
        # especially for fine-tuned models
        # raw_scores[0] -> logprob of token with ID 0
        # raw_scores[1] -> logprob of token with ID 1
        # ...
        # raw_scores[len(n_vocab) - 1] -> ID of last token in vocab

        assert isinstance(prompt, str), f'prompt should be string, not {type(prompt)}'
        self.llama.reset() # reset model state
        prompt_tokens = self.llama.tokenize(prompt.encode('utf-8', errors='ignore'))
        self.llama.eval(prompt_tokens)
        raw_scores = self.llama.scores[0]

        all_probs_dict = {}

        for i in range(len(raw_scores)):
            #i_str = self.llama.detokenize([i]).decode('utf-8', errors='ignore')
            #all_probs_dict[i_str] = raw_scores[i]
            all_probs_dict[i] = raw_scores[i]
        
        # verified: now all_probs_dict[tok_id] -> tok_logprob ; and len(all_probs_dict) == n_vocab

        # sort dict by logprobs, high to low, then keep only the first k entries
        top_k_tok_list: list[tuple[int, float]] = sorted(all_probs_dict.items(), key=lambda x:x[1], reverse=True)[:k]
        #top_k_tok_list = sorted(all_probs_dict, key=all_probs_dict.get, reverse=True)[:k]

        top_k_str_list = []

        for tok in top_k_tok_list:
            top_k_str_list.append(self.llama.detokenize([tok[0]]).decode('utf-8', errors='ignore'))

        #top_k_str_list.reverse()

        return top_k_str_list

class Thread(object):

    def __init__(self, model: Model, format: dict, timestamps: bool=False):
        assert isinstance(model, Model), f'Thread: model should be an instance of easy_llama.Model, \
            not {type(model)}'
        
        assert isinstance(format, dict), f'Thread: format should be dict, not {type(format)}'

        # Q: why don't you just check if the format is in the formats list?
        # A: user should be able to create and use their own format without mangling the default formats

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
            e.add_note('Thread: format is missing one or more required keys, see \
                easy_llama.blank for an example')
            raise

        assert isinstance(format['stops'], (list, type(None))), \
            f"Thread: format['stops'] should be list[str] or None, not {type(format['stops'])}"

        assert isinstance(timestamps, bool), f'Thread: timestamps should be True or False, not \
            \'{timestamps}\''
        
        self.model: Model = model
        self.format: dict = format
        self.enable_timestamps: bool = timestamps
        self.messages: list[dict] = [self.create_message('system', self.format['system_content'])]
    
    def create_message(self, role: str, content: str) -> dict:
        assert role.lower() in ['system', 'user', 'bot'], f'create_message: \
            role should be \'system\', \'user\', or \'bot\', not \'{role.lower()}\''
        
        assert isinstance(content, str), f'create_message: content should be str, not {type(content)}'

        if role.lower() == 'system':
            return {
                 'prefix': self.format['system_prefix'],
                'content': content if not self.enable_timestamps
                    else time.strftime('It is %I:%M %p on %A, %b %e, %Y. ') + content,
                'postfix': self.format['system_postfix'],
                 'length': self.model.get_length(
                     self.format['system_prefix'] + content + self.format['system_postfix']
                     )
            }
        elif role.lower() == 'user':
            return {
                 'prefix': self.format['user_prefix'],
                'content': content if not self.enable_timestamps
                    else time.strftime('[at %a %I:%M %p] ') + content,
                'postfix': self.format['user_postfix'],
                 'length': self.model.get_length(
                     self.format['user_prefix'] + content + self.format['user_postfix']
                     )
            }
        elif role.lower() == 'bot':
            return {
                 'prefix': self.format['bot_prefix'],
                'content': content if not self.enable_timestamps
                    else time.strftime('[at %a %I:%M %p] ') + content,
                'postfix': self.format['bot_postfix'],
                 'length': self.model.get_length(
                     self.format['bot_prefix'] + content + self.format['bot_postfix']
                     )
            }

    def inference_str_from_messages(self) -> str:
        inference_str = ''
        # bos + eos + off-by-one errors == 3 (better safe than sorry)
        context_len_budget = self.model.context_length - 3
        system_message = self.messages[0]
        context_len_budget -= system_message['length']
        context_len_budget -= self.model.get_length(self.format['bot_prefix'])
        sys_msg_str = system_message['prefix'] + system_message['content'] + system_message['postfix']

        # now inference_str contains system message
        # and context_len_budget equals n_ctx - (3 + system msg len + bot prefix len)

        # start at most recent message and work backwards up the history
        # excluding system message. once we exceed the model's context length,
        # break without including that message
        for message in reversed(self.messages[1:]):
            context_len_budget -= message['length']
            if context_len_budget <= 0:
                break
            msg_str = message['prefix'] + message['content'] + message['postfix']
            inference_str = msg_str + inference_str
        inference_str = sys_msg_str + inference_str
        inference_str += self.format['bot_prefix']
        if self.enable_timestamps:
            inference_str += time.strftime('[at %a %I:%M %p] ')
        return inference_str
    
    def send(self, prompt: str) -> str:
        assert isinstance(prompt, str), f'Thread.send: prompt should be str, not {type(prompt)}'
        assert prompt != '', 'Thread.send: empty prompts are not allowed in threads'

        self.messages.append(self.create_message('user', prompt))
        output = self.model.generate(
            self.inference_str_from_messages(),
            stops=self.format['stops']
        )
        self.messages.append(self.create_message('bot', output))

        if output.endswith(' '):
            _print_warning('easy_llama: inference str has trailing whitespace\n')

        return output
        
    def interact(self) -> None:
        print()
        try:
            while True:
                prompt = input('  > ')
                print() # Put the cursor where the generated text is about to appear
                if prompt == '':

                    # another assistant message
                    token_generator = self.model.llama.create_completion(
                        self.inference_str_from_messages(),
                        max_tokens=MAX_LEN_TOKENS,
                        top_k=4,
                        stream=True,
                        presence_penalty=0.5,
                        stop=self.format['stops'],
                    )
                    
                    output = ''
                    for i in token_generator:
                        token = i['choices'][0]['text']
                        output += token
                        print(token, end='', flush=True)

                    self.messages.append(self.create_message('bot', output))

                else:
                    self.messages.append(self.create_message('user', prompt))

                    token_generator = self.model.llama.create_completion(
                        self.inference_str_from_messages(),
                        max_tokens=MAX_LEN_TOKENS,
                        top_k=4,
                        stream=True,
                        presence_penalty=0.5,
                        stop=self.format['stops'],
                    )
                    
                    output = ''
                    for i in token_generator:
                        token = i['choices'][0]['text']
                        output += token
                        print(token, end='', flush=True)

                    self.messages.append(self.create_message('bot', output))
                
                if output.endswith('\n\n'):
                    pass
                elif output.endswith('\n'):
                    print()
                else:
                    print('\n')
        
        except KeyboardInterrupt:
            print('\n')
            return
    
    def reset(self) -> None:
        self.messages: list[dict] = [self.create_message('system', self.format['system_content'])]

blank = {
     'system_prefix': '',
    'system_content': '',
    'system_postfix': '',
       'user_prefix': '',
      'user_content': '',
      'user_postfix': '',
        'bot_prefix': '',
       'bot_content': '',
       'bot_postfix': '',
             'stops': []
}

# https://github.com/openai/openai-python/blob/main/chatml.md
chatml = {
     'system_prefix': '<|im_start|>system\n',
    'system_content': '',
    'system_postfix': '<|im_end|>\n',
       'user_prefix': '<|im_start|>user\n',
      'user_content': '',
      'user_postfix': '<|im_end|>\n',
        'bot_prefix': '<|im_start|>assistant\n',
       'bot_content': '',
       'bot_postfix': '<|im_end|>\n',
             'stops': ['<|im_end|>', '<|im_start|>']
}

# https://huggingface.co/blog/llama2
# system message relaxed to avoid undue refusals
llama2chat = {
     'system_prefix': '<s>[INST] <<SYS>>\n',
    'system_content': 'You are a helpful AI assistant.',
    'system_postfix': '\n<</SYS>>\n\n',
       'user_prefix': '',
      'user_content': '',
      'user_postfix': ' [/INST]',
        'bot_prefix': ' ',
       'bot_content': '',
       'bot_postfix': ' </s><s>[INST] ',
             'stops': ['</s>', '[INST]']
}

# https://github.com/tatsu-lab/stanford_alpaca
alpaca = {
     'system_prefix': '',
    'system_content': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.',
    'system_postfix': '\n\n',
       'user_prefix': '### Instruction:\n',
      'user_content': '',
      'user_postfix': '\n\n',
        'bot_prefix': '### Response:\n',
       'bot_content': '',
       'bot_postfix': '\n\n',
             'stops': ['###', 'Instruction:', '\n\n\n']
}

# this is the official vicuna. it is often butchered in various ways, 
# most commonly by adding line breaks
# see v1.1 here: https://github.com/flu0r1ne/FastChat/blob/259a171a24196acf97f9a4d90825ca6a68f331ab/docs/vicuna_weights_version.md?plain=1
vicuna_lmsys = {
     'system_prefix': '<s>',
    'system_content': '',
    'system_postfix': ' ',
       'user_prefix': 'USER: ',
      'user_content': '',
      'user_postfix': ' ',
        'bot_prefix': 'ASSISTANT: ',
       'bot_content': '',
       'bot_postfix': '</s> ',
             'stops': ['</s>', 'USER:']
}

# spotted here and elsewhere: https://huggingface.co/Norquinal/Mistral-7B-claude-chat
vicuna_common = {
     'system_prefix': '',
    'system_content': 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.',
    'system_postfix': '\n\n',
       'user_prefix': 'USER: ',
      'user_content': '',
      'user_postfix': '\n',
        'bot_prefix': 'ASSISTANT: ',
       'bot_content': '',
       'bot_postfix': '\n',
             'stops': ['</s>', 'USER:', 'ASSISTANT:']
}

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
mistral_instruct = {
     'system_prefix': '<s>',
    'system_content': '',
    'system_postfix': '',
       'user_prefix': '[INST] ',
      'user_content': '',
      'user_postfix': ' [/INST]',
        'bot_prefix': ' ',
       'bot_content': '',
       'bot_postfix': '</s> ',
             'stops': ['</s>', '[INST]']
}

# https://huggingface.co/timdettmers/guanaco-65b
guanaco = {
     'system_prefix': '',
    'system_content': 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.',
    'system_postfix': '\n',
       'user_prefix': '### Human: ',
      'user_content': '',
      'user_postfix': ' ',
        'bot_prefix': '### Assistant: ',
       'bot_content': '',
       'bot_postfix': ' ',
             'stops': ['###', 'Human:']
}

# https://huggingface.co/pankajmathur/orca_mini_v3_7b
orca_mini = {
     'system_prefix': '### System:\n',
    'system_content': 'You are an AI assistant that follows instruction extremely well. Help as much as you can.',
    'system_postfix': '\n\n',
       'user_prefix': '### User:\n',
      'user_content': '',
      'user_postfix': '\n\n',
        'bot_prefix': '### Assistant:\n',
       'bot_content': '',
       'bot_postfix': '\n\n',
             'stops': ['###', 'User:']
}

# https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
zephyr = {
     'system_prefix': '<|system|>\n',
    'system_content': 'You are a friendly chatbot.',
    'system_postfix': '</s>\n',
       'user_prefix': '<|user|>\n',
      'user_content': '',
      'user_postfix': '</s>\n',
        'bot_prefix': '<|assistant|>\n',
       'bot_content': '',
       'bot_postfix': '\n',
             'stops': ['</s>', '<|user|>']
}

mistral_openorca = chatml.copy()
dolphin = chatml.copy()
samantha = chatml.copy()
jackalope = chatml.copy()
dolphin['system_content'] = 'You are Dolphin, a helpful AI assistant.'
mistral_openorca['system_content'] = 'You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!'
samantha['system_content'] = 'You are a caring and empathetic sentient AI companion named Samantha.'
jackalope['system_content'] = 'You are JackalopeAI, a large language model trained by OpenAccess AI Collective. Write out your reasoning step-by-step to be sure you get the right answers!'
airoboros = llama2chat.copy()
airoboros['system_content'] = 'You are a helpful, unbiased, uncensored assistant.'

formats: list[dict] = [
    blank,              chatml,         llama2chat,
    alpaca,             vicuna_lmsys,   vicuna_common,
    mistral_openorca,   dolphin,        samantha,
    guanaco,            orca_mini,      airoboros,
    jackalope,          mistral_instruct,  zephyr
]

def wrap(prompt: str, format: dict, timestamps: bool=False) -> str:
    if not timestamps:
        return \
            format['system_prefix'] + \
            format['system_content'] + \
            format['system_postfix'] + \
            format['user_prefix'] + \
            prompt + \
            format['user_postfix'] + \
            format['bot_prefix']
    else:
        return \
            format['system_prefix'] + \
            time.strftime('It is %A, %b %e, %Y. ') + \
            format['system_content'] + \
            format['system_postfix'] + \
            format['user_prefix'] + \
            time.strftime('[at %a %I:%M %p] ') + \
            prompt + \
            format['user_postfix'] + \
            format['bot_prefix']

if __name__ == '__main__':
    raise RuntimeError('easy_llama cannot be run directly, please import it into your environment')
