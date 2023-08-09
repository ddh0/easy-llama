# easy_llama.py
# Python 3.11.2

"""
Simplified deployment of llama models

for complete and up-to-date information, see https://github.com/ddh0/easy-llama
"""

# TODO:
# - chat conversation threads
# - publish to pypi

import os
import sys
import pickle

DEBUG = False

class suppress_if_no_debug(object):
    """
    Suppress console output from llama.cpp if easy_llama.DEBUG == False

    Changing DEBUG inside the WITH block could have strange effects

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

if 'llama_cpp' not in sys.modules:
    raise ImportWarning('easy_llama: required llama_cpp module not imported, \
                        likely due to \'from easy_llama import ...\' statement')

class Presets:
    """
    You can make your own preset as long as it these attributes:
    .temp
    .top_p
    .top_k
    .repeat_penalty
    
    It does not need to be under Presets, it can be in your own module.
    """

    # repeat_penalty == 1: no penalty
    # repeat_penalty <  1: no idea

    class Standard:
        temp = 0.7
        top_p = 0.37
        top_k = 32
        repeat_penalty = 1.1
    
    class Creative:
        temp = 0.9
        top_p = 0.37
        top_k = 128
        repeat_penalty = 1.2

    class Crazy:
        temp = 1.9
        top_p = 1
        top_k = 256
        repeat_penalty = 2


class Model(object):
    """
    """

    def __init__(self, model_path: str, context_length=2048) -> None:
        """
        Initialize a Llama model from a file.
        
        This will take some time, anywhere from 2 seconds to 2 minutes
        depending on the size of the model and the speed of the device
        """

        if type(model_path) is not str:
            raise TypeError('model_path should be a string, not %s' % type(model_path))
    
        if os.path.isdir(model_path):
            raise IsADirectoryError('the given model_path \'%s\' is a directory, not a file' % model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError('the given model_path \'%s\' does not exist' % model_path)
        
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
    
    def generate(self, prompt: str, preset=Presets.Standard, max_length=512, stop_sequences=None) -> str:
       """
       Given a prompt, return a generated string.

       Optional parameters:
       - preset: alters the way tokens are chosen, see the Presets class for options
       - max_length: maximum length of generated string in tokens
       - stop_sequences: string or list of strings at which to end the generation early (right before)
       """

       if type(prompt) is not str:
           raise TypeError("prompt should be string, not %s" % type(prompt))

       if type(max_length) is not int:
           raise TypeError("max_length should be int, not %s" % type(max_length))
       
       if max_length > self.context_length:
           print("easy_llama: generate: warning: max_length is greater than model's context_length")
       
       if not hasattr(preset, 'temp'):
           raise AttributeError("preset is missing attribute .temp")
       
       if not hasattr(preset, 'top_p'):
           raise AttributeError("preset is missing attribute .top_p")
       
       if not hasattr(preset, 'top_k'):
           raise AttributeError("preset is missing attribute .top_k")
       
       if not hasattr(preset, 'repeat_penalty'):
           raise AttributeError("preset is missing attribute .repeat_penalty")

       if type(stop_sequences) is not (list or str or None):
           raise TypeError("stop_sequence should be list or str or None, not %s" % type(stop_sequences))

       with suppress_if_no_debug(): 
            return self.__internal_model.create_completion(prompt,
                                                           max_tokens=max_length,
                                                           temperature=preset.temp,
                                                           top_p=preset.top_p,
                                                           top_k=preset.top_k,
                                                           stream=False,
                                                           stop=stop_sequences,
                                                           repeat_penalty=preset.repeat_penalty
                                                           )['choices'][0]['text']
    
    def generate_greedy(self, prompt: str, max_length=512, stop_sequences=None) -> str:
        """
        Given a prompt, return a generated string with greedy sampling.

        Optionally specify max_length and stop_sequences.

        Greedy sampling means that the most likely token is always chosen.

        Note that easy_llama.Model is hardcoded to initialize the model
        with a random seed, so generations are not deterministic across
        different instances of Model.
        """

        # top_p == 1 --> greedy sampling, according to llama.cpp docs

        if type(prompt) is not str:
            raise TypeError("prompt should be string, not %s" % type(prompt))
        
        if type(max_length) is not int:
           raise TypeError("max_length should be int, not %s" % type(max_length))

        if max_length > self.context_length:
           print("easy_llama: generate_greedy: warning: max_length is greater than model's context_length")
        
        with suppress_if_no_debug():
            return self.__internal_model.create_completion(prompt,
                                                           max_tokens=max_length,
                                                           temperature=0,
                                                           top_p=1,
                                                           top_k=0,
                                                           stream=False,
                                                           stop=stop_sequences,
                                                           repeat_penalty=1
                                                           )['choices'][0]['text']

    def generate_raw(self, prompt: str, max_tokens: int, temperature: float,
                       top_p: float, top_k: int, stream: bool, stop: list | str | None,
                       repeat_penalty: float
                       ) -> llama_cpp.Completion | llama_cpp.Iterator[llama_cpp.CompletionChunk]:
        """
        Directly expose the most commonly useful parameters
        of Llama.create_completion

        Returns a response object containing the generated text

        Raw in this case means forgoing easy_llama's usual
        simplicity in favor of utility. If you want something more
        low-level than this, use llama-cpp-python or llama.cpp
        directly.
        """
        
        with suppress_if_no_debug():
            return self.__internal_model.create_completion(prompt,
                                                           max_tokens,
                                                           temperature,
                                                           top_p,
                                                           top_k,
                                                           stream,
                                                           stop,
                                                           repeat_penalty,
                                                           )


class Formatters:
    """
    Convenience functions to return a given input in various
    prompt formats like Alpaca, Guanaco, etc
    """

    def alpaca(text: str) -> str:
        """
        Return any text in Stanford's Alpaca prompt format

        Recommened format for Intruct/Response

        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {text}
        
        ### Response:

        """

        return 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n%s\n\n### Response:\n' % text

    def llama(text: str) -> str:
        """
        Return any text in Meta AI's Llama-2-Chat prompt format

        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant.
        <</SYS>>
        
        {text} [/INST] 
        """

        return '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n%s [/INST] ' \
        % text

    def guanaco(text: str) -> str:
        """
        Return any text in Tim Dettmers' Guanaco prompt format

        Recommended format for assitant interactions

        A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        ### Human: {text} ### Assistant: 
        """

        return 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n### Human: %s ### Assistant: ' \
        % text


class Thread(object):
    """
    Provide functions to facilitate conversation threads with a model
    (i.e. remembering past messages)

    Allows for arbitrary number of threads to be held simultaneously
    and stored on disk, and for moving interactions between models
    """

    def __init__(self, parties, bot):
        pass
