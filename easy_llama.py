# easy_llama.py
# Python 3.11.6

"""
easy-llama - Natural text generation in Python, made easy
----------
https://github.com/ddh0/easy-llama/
"""

# Redo entire format / messages system
#
# store messages in dict with role and text attr ?
# 
# TODO: set batch size smartly based on memory or cpu ??
# TODO: message-based context length handling in threads
# TODO: Model.next_candidates() -> list[str]
# TODO: Automatic detection of METAL, CUDA, OpenBLAS/CPU and set NUM_GPU_LAYERS accordingly?

import os
import sys
import struct
from enum import IntEnum

# Set to 1 for Apple Silicon
# Set to 0 for CPU / OpenBLAS
# Tweak as needed for CUDA / ROCm
# Set to -1 to move all layers to CUDA / ROCm
NUM_GPU_LAYERS: int = 1

# Max length of each generation in tokens
MAX_LEN_TOKENS: int = 64

# Print all backend information as it occurs
VERBOSE: bool = False

# Leave at -1 for random seed. Used when loading models
SEED: int = -1

# Warnings are infrequent and contain helpful information
SUPPRESS_WARNINGS: bool = False

class _GGUF_READER:
    """
    Peek at file header for GGUF metadata

    Credit to oobabooga for the majority of the code in this class
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
            GGUF_MAGIC = struct.unpack("<I", file.read(4))[0]
            GGUF_VERSION = struct.unpack("<I", file.read(4))[0]
            ti_data_count = struct.unpack("<Q", file.read(8))[0]
            kv_data_count = struct.unpack("<Q", file.read(8))[0]

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


with _suppress_if_not_verbose():
        import llama_cpp


def _print_warning(text: str) -> str:
    if not SUPPRESS_WARNINGS:
        print('easy_llama: ' + text, file=sys.stderr)


class Model(object):
    """
    Abstraction of a llama model
    """

    def __init__(self, model_path: str):
        """
        Initialize a llama model from a file.

        Model must be in GGUF format (but it does not have to be quantized).

        easy_llama will automatically determine the model's trained
        context length from the GGUF metadata.
        """

        assert isinstance(model_path, str), 'model_path should be a string, not %s' % type(model_path)
        assert not os.path.isdir(model_path), 'the given model_path \'%s\' is a directory, not a file' % model_path
        assert os.path.exists(model_path), 'the given model_path \'%s\' does not exist' % model_path
        
        self.metadata = _GGUF_READER.load_metadata(model_path)
        
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
            self._internal_model: llama_cpp.Llama = \
                llama_cpp.Llama(model_path=model_path,
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
            # Same amount of time spent either way, but this is up-front rather than
            # during usage
            self._internal_model.create_completion('',
                                                   max_tokens=1, # if <= 0, unlimited
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


    def greedy(self, prompt: str, stops: list[str] | None=None) -> str:
        """
        Given a prompt, return a generated string using greedy decoding,
        where the most likely token is always chosen.

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
        
        with _suppress_if_not_verbose():
            return self._internal_model.create_completion(prompt,
                                                          max_tokens=MAX_LEN_TOKENS,
                                                          top_p=0,
                                                          top_k=1,
                                                          stop=stops,
                                                          repeat_penalty=1
                                                          )['choices'][0]['text']


    def generate(self, prompt: str, stops: list[str] | None=None) -> str:
        """
        Given a prompt, return a generated string using constrastive search
        with a moderate alpha value. This is the method easy_llama recommends
        for most cases.

        For more information on contrastive search, see here:
        https://huggingface.co/blog/introducing-csearch

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
        
        with _suppress_if_not_verbose():
            return self._internal_model.create_completion(prompt,
                                                          max_tokens=MAX_LEN_TOKENS,
                                                          top_k=4,
                                                          presence_penalty=0.525,
                                                          stop=stops,
                                                          )['choices'][0]['text']
    
 
    def next_candidates(self, prompt: str, k: int) -> list[str]:
        """
        Given a prompt, return a sorted list of the top k candidates for
        most likely next token
        """

        # Good luck to self
        # Possible hint:
        # logits
        # eval prompt without generating (i.e. keep state)
        # then look at values in:
        # Llama._candidates_data

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)

format_template = {
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

chat_ml = {
     'system_prefix': '<|im_start|>system\n',
    'system_content': '',
    'system_postfix': '<|im_end|>\n',
       'user_prefix': '<|im_start|>user\n',
      'user_content': '',
      'user_postfix': '<|im_end|>\n',
        'bot_prefix': '<|im_start|>assistant\n',
       'bot_content': '',
       'bot_postfix': '<|im_end|>\n',
             'stops': ['<|im_end|>']
}

formats: list[dict] = [format_template, chat_ml]

def create_message(format: dict, role: str, content: str) -> dict:
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
    except KeyError as e:
        e.add_note('create_message: format is missing one or more required keys, see \
                   easy_llama.format_template for an example')
        raise
    assert role.lower() in ['system', 'user', 'bot'], f'create_message: \
        role should be \'system\', \'user\', or \'bot\', not \'{role.lower()}\''
    assert isinstance(content, str), f'create_message: content should be str, not {type(content)}'
    if role.lower() == 'system':
        # length key will be set by model, since different models use different tokenization
        # it should be discarded and recalculated if the model is swapped out
        return {
             'prefix': format['system_prefix'],
            'content': content,
            'postfix': format['system_postfix'],
             'length': None
        }
    elif role.lower() == 'user':
        return {
             'prefix': format['user_prefix'],
            'content': content,
            'postfix': format['user_postfix'],
             'length': None
        }
    elif role.lower() == 'bot':
        return {
             'prefix': format['bot_prefix'],
            'content': content,
            'postfix': format['bot_postfix'],
             'length': None
        }


class Thread(object):

    def __init__(self, model: Model, format: dict):
        assert isinstance(model, Model), 'Thread: model should be an instance of easy_llama.Model'
        try:
            # Q: why don't you just check if the format is in the formats list?
            # A: user should be able to create and use their own format without mangling the default formats
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
                    easy_llama.format_template for an example')
            raise
        assert isinstance(format['stops'], list[str], type(None)), \
            f"Thread: format['stops'] should be list[str] or None, not {type(format['stops'])}"
        self.model: Model = model
        self.format: dict = format
        self.messages: list[dict] = [create_message(format, 'system', format['system_content'])]
    
    def reset(self) -> None:
        self.messages = [create_message(self.format, 'system', self.format['system_content'])]
    
    def send(self, prompt: str) -> str:
        assert isinstance(prompt, str), 'Thread.send: prompt should be str'
        # do not allow empty input, as that would mess up the format
        assert prompt != '', 'Thread.send: empty prompts are not allowed in threads'
    

    def interact(self) -> None:
        print()
        try:
            while True:
                prompt = input('  > ')
                print() # Put the cursor where the generated text is about to appear
                if prompt == '':
                    _print_warning('empty prompts are not allowed in threads\n')
                    continue
                else:
                    #X
                    output = self.model.generate(self.get_inference_str(), stops=self.format.stops)
                    #X

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
