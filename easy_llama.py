# easy_llama.py
# Python 3.11.2

"""
easy-llama - Simplified deployment of llama models

requires llama-cpp-python, which should be installed automatically

for complete and up-to-date information, see https://github.com/ddh0/easy-llama
"""

# TODO:
# - chat conversation threads
# - publish to pypi

import os
import sys

DEBUG = False

class suppress_if_no_debug(object):
    """
    Suppress console output from llama.cpp if easy_llama.DEBUG == False

    See https://github.com/abetlen/llama-cpp-python/issues/478
    """

    def __enter__(self):
        if not DEBUG:
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
        
        if DEBUG:
            # do nothing
            return self

    def __exit__(self, *_):
        if not DEBUG:        
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr

            os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
            os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

            os.close ( self.old_stdout_fileno )
            os.close ( self.old_stderr_fileno )

            self.outnull_file.close()
            self.errnull_file.close()

try:
    with suppress_if_no_debug():
        import llama_cpp

except ModuleNotFoundError as e:
    e.add_note('Hint: Try \'pip install llama-cpp-python\'')
    raise

class Presets:
    """
    You can make your own preset as long as it has .temp, .top_p, and .top_k attributes
    
    It does not need to be under Presets, it can be in your own module
    """

    class Standard:
        temp = 0.7
        top_p = 0.37
        top_k = 32
    
    class Creative:
        temp = 0.9
        top_p = 0.37
        top_k = 128

    class Crazy:
        temp = 1.7
        top_p = 1
        top_k = 256


class Model(object):
    """
    """

    def __init__(self, model_path: str, context_length=2048):
        """
        Initialize a Llama model from a file.
        
        This will take some time, anywhere from 2 seconds to 2 minutes
        depending on the size of the model and the speed of the device
        """

        if type(model_path) is not str:
            raise TypeError("model_path should be a string, not %s" % type(model_path))
    
        if os.path.isdir(model_path):
            raise IsADirectoryError("%s is a directory, not a file" % model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError("file %s does not exist" % model_path)
        
        if type(context_length) is not int:
            raise TypeError("context_length should be int, not %s" % type(context_length))
        
        self.context_length = context_length

        with suppress_if_no_debug():
            self.__internal_model = llama_cpp.Llama(model_path=model_path,
                                                    n_ctx=self.context_length,
                                                    seed=0,
                                                    use_mmap=False,
                                                    use_mlock=False,
                                                    n_batch=512,
                                                    verbose=False)
    
    def generate(self, prompt: str, preset=Presets.Standard, max_length=512, stop_sequences=['###']) -> str:
       """
       Given a prompt and a maximum length, return a generated string.

       Optionally, specify a preset to control the behaviour of the model.

       Built-in presets are Presets.Standard (default), Presets.Creative, and Presets.Crazy.

       Optionally, specify a list of stop sequences that force the model to end generation early.
       """

       if type(prompt) is not str:
           raise TypeError("prompt should be string, not %s" % type(prompt))

       if type(max_length) is not int:
           raise TypeError("max_length_in_tokens should be int, not %s" % type(max_length))
       
       if max_length > self.context_length:
           raise Warning("generate: max_length is greater than model's context length")
       
       if not hasattr(preset, 'temp'):
           raise AttributeError("preset is missing attribute .temp")
       
       if not hasattr(preset, 'top_p'):
           raise AttributeError("preset is missing attribute .top_p")
       
       if not hasattr(preset, 'top_k'):
           raise AttributeError("preset is missing attribute .top_k")

       if type(stop_sequences) is not list:
           raise TypeError("stop_sequence should be list, not %s" % type(stop_sequences))

       with suppress_if_no_debug():
            return self.__internal_model.create_completion(prompt,
                                                           max_tokens=max_length,
                                                           temperature=preset.temp,
                                                           top_p=preset.top_p,
                                                           top_k=preset.top_k,
                                                           stream=False,
                                                           stop=stop_sequences,
                                                           repeat_penalty=1.2
                                                           )['choices'][0]['text']


def text_to_alpaca_format(text: str):
    """
    Return any text in Stanford's Alpaca prompt format

    This is good for Instruction/Response generations.

    For example:
    
    >>> easy_llama.text_to_alpaca_format('How do I make pasta?')
    'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nHow do I make pasta?\n\n### Response:\n'
    """

    return 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n%s\n\n### Response:\n' % text

def text_to_llama_format(text: str):
    """
    Return any text in Meta AI's official Llama 2 chat prompt format

    This is good for a safe, cautious assistant.
    
    For example:

    >>> easy_llama.text_to_llama_format('How do I make pasta?')
    '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nHow do I make pasta? [/INST]'
    """

    return '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n%s [/INST]' \
    % text

def text_to_guanaco_format(text: str):
    """
    Return any text in Tim Dettmers' Guanaco prompt format

    This is good for helpful assistant that is not as cautious as Meta's.

    For example:

    >>> easy_llame.text_to_guanaco_format('How do I make pasta?')
    'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n### Human: How do I make pasta? ### Assistant:'
    """

    return 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n### Human: %s ### Assistant:' \
    % text

class ChatThread(object):
    """
    TODO
    """

