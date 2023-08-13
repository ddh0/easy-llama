# easy_llama.py
# Python 3.11.2

# pip uninstall llama-cpp-python
# CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

"""
Simplified text inference using llama models

for complete and up-to-date information, see https://github.com/ddh0/easy-llama
"""

# TODO:
# - chat conversation threads
# - publish to pypi

import os
import sys

from typing import Iterable

DEBUG = False
SUPPRESS_WARNINGS = False

class _suppress_if_no_debug(object):
    """
    Suppress console output from llama.cpp if easy_llama.DEBUG is False

    Changing DEBUG inside the WITH block could have strange effects

    See https://github.com/abetlen/llama-cpp-python/issues/478
    """

    # llama_cpp has its own function that does this, but
    # I believe it may be removed at some point, so here it is again

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

with _suppress_if_no_debug():
        import llama_cpp

def _debug_output(text: str) -> str:
    """
    Print text to stderr if DEBUG is True,
    prefixing with 'easy_llama: debug: '
    """

    if DEBUG:
        print('easy_llama: debug: ' + text, file=sys.stderr)

def _warning_output(text: str) -> str:
    """
    Print text to stderr unless SUPPRESS_WARNINGS is True,
    prefixing with 'easy_llama: warning: '
    """

    if not SUPPRESS_WARNINGS:
        print('easy_llama: warning: ' + text, file=sys.stderr)

# if something is missing, fail ASAP and explain
_required_modules = ['os', 'sys', 'pickle', 'llama_cpp']

for modulename in _required_modules:
    if modulename not in sys.modules:
        e = ModuleNotFoundError('missing required module \'%s\'' % modulename)
        e.add_note('Hint: see https://github.com/ddh0/easy-llama')
        raise e

class Presets:
    """
    Default preset is Presets.Standard

    Choosing another preset will change the way tokens are sampled (chosen)

    You can use anything as a preset as long as it these attributes:
    .temp
    .top_p
    .top_k
    .repeat_penalty
    """

    # repeat_penalty == 1: no penalty
    # repeat_penalty <  1: no idea

    class Greedy:
        """Greedy sampling (deterministic for same Model instance)"""
        temp = 0.0
        top_p = 1
        top_k = 0
        repeat_penalty = 1

    class Standard:
        """Reasonable default"""
        temp = 0.7
        top_p = 0.37
        top_k = 32
        repeat_penalty = 1.1
    
    class Creative:
        """Coherent and creative"""
        temp = 0.9
        top_p = 0.37
        top_k = 128
        repeat_penalty = 1.2
    
    class Funny:
        """Just for laughs"""
        temp = 10
        top_p = 0.1
        top_k = 128
        repeat_penalty = 1
    
    class Crazy:
        """Creative to a fault"""
        temp = 1.4
        top_p = 1
        top_k = 1024
        repeat_penalty = 1.2
    

class Model(object):
    """
    Super-abstraction of a Llama model
    """

    def __init__(self, model_path: str, context_length: int=2048) -> None:
        """
        Initialize a Llama model from a file.
        
        This will take some time, anywhere from 0.1 seconds to 5 minutes
        depending on the size of the model.
        """

        _debug_output("init: begin initializing")

        if type(model_path) is not str:
            raise TypeError('model_path should be a string, not %s' % type(model_path))
    
        if os.path.isdir(model_path):
            raise IsADirectoryError('the given model_path \'%s\' is a directory, not a file' % model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError('the given model_path \'%s\' does not exist' % model_path)
        
        if type(context_length) is not int:
            raise TypeError("context_length should be int, not %s" % type(context_length))
        
        self.context_length = context_length

        _debug_output("init: basic checks passed. will now load model...")

        with _suppress_if_no_debug():
            self._internal_model: llama_cpp.Llama = llama_cpp.Llama(model_path=model_path,
                                                                    n_ctx=self.context_length,
                                                                    seed=0,
                                                                    use_mmap=False,
                                                                    use_mlock=False,
                                                                    logits_all=True,
                                                                    n_batch=512,
                                                                    verbose=DEBUG)
        
        _debug_output("init: model loaded")
    
    def generate(self, prompt: str, preset=Presets.Standard, max_length: int=512,
                 stop_sequences: list[str] | str | None=None, stream: bool=False) -> str | Iterable[str]:
        """
        Given a prompt, return a generated string.

        Optional parameters:
        
        preset: changes the way tokens are sampled (chosen). See the Presets class for more info.
        Default is Presets.Standard.
        
        max_length: maximum length of generated string in tokens. Default is 512
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None

        stream: if True, continue yield single tokens rather than returning a
        single string. Default is False
        """

        if type(prompt) is not str:
           raise TypeError("generate: prompt should be string, not %s" % type(prompt))
       
        if not hasattr(preset, 'temp'):
           raise AttributeError("generate: preset is missing attribute .temp")
       
        if not hasattr(preset, 'top_p'):
           raise AttributeError("generate: preset is missing attribute .top_p")
       
        if not hasattr(preset, 'top_k'):
           raise AttributeError("generate: preset is missing attribute .top_k")
       
        if not hasattr(preset, 'repeat_penalty'):
           raise AttributeError("generate: preset is missing attribute .repeat_penalty")

        if type(max_length) is not int:
           raise TypeError("generate: max_length should be int, not %s" % type(max_length))
       
        if max_length > self.context_length:
           _warning_output("generate: max_length is greater than context_length")

        if type(stop_sequences) not in [list, str, type(None)]:
           raise TypeError("generate: stop_sequence should be list or str or None, not %s" % type(stop_sequences))

        if type(stop_sequences) is list:
            for item in list:
                if type(item) is not str:
                    raise TypeError("generate: item \'%s\' in stop_sequences list is not of type str" \
                                    % repr(item))

        if type(stream) is not bool:
            raise TypeError("generate: stream should be bool (True or False), not %s" % type(stream))

        _debug_output("generate: basic checks passed. begin generation...")
        
        if not stream: # Return a completed string
           with _suppress_if_no_debug():
                return self._internal_model.create_completion(prompt,
                                                              max_tokens=max_length,
                                                              temperature=preset.temp,
                                                              top_p=preset.top_p,
                                                              top_k=preset.top_k,
                                                              stream=False,
                                                              stop=stop_sequences,
                                                              repeat_penalty=preset.repeat_penalty  
                                                             )['choices'][0]['text']
        if stream: # Continuously yield tokens
            with _suppress_if_no_debug():
                llama_completion_generator = self._internal_model.create_completion(prompt,
                                                              max_tokens=max_length,
                                                              temperature=preset.temp,
                                                              top_p=preset.top_p,
                                                              top_k=preset.top_k,
                                                              stream=True,
                                                              stop=stop_sequences,
                                                              repeat_penalty=preset.repeat_penalty)
                
            # create_completion() returns a generator which yields generators (??)
            while True:
                for sub_generator in llama_completion_generator:
                    for x in sub_generator:
                        yield x['choices']['text']


class Formatters:
    """
    Convenience functions to return a given input in various
    prompt formats.

    See docstrings of each function for more information.
    """

    def assist(text: str) -> str:
        """
        Return any text in Tim Dettmers' Guanaco prompt format.
        This is the suggested format for assitant generations.

        A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        ### Human: {text} ### Assistant: 
        """

        return 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n### Human: %s ### Assistant: ' \
        % text

    def instruct(text: str) -> str:
        """
        Return any text in Stanford's Alpaca prompt format.
        This is the suggested format for Instruct-Response generations.

        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {text}
        
        ### Response:

        """

        return 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n%s\n\n### Response:\n' % text

    def llama_2_chat(text: str) -> str:
        """
        Return any text in Meta AI's Llama-2-Chat prompt format.
        This is the suggested format for Llama-2-Chat generations.
        
        This does not apply to base Llama models or unofficial chat
        finetunes.

        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant.
        <</SYS>>
        
        {text} [/INST] 
        """

        return '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n%s [/INST] ' \
        % text


class Thread(object):
    """
    Provide functions to facilitate conversation threads with a model
    (i.e. remembering past messages)

    Allows for arbitrary number of threads to be held simultaneously
    and stored on disk, and for moving interactions between models
    """

    count = 0

    types = ['simple', 'assistant']

    def __init__(self, type):
        
        type = type.lower()
        if type not in Thread.types:
            pass # TODO

        #if types

        Thread.count += 1
        self.num = Thread.count + 1
    
    def msg(self):
        pass

