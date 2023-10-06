# easy_llama.py
# Python 3.11.6

"""
Simple, on-device text inference using language models

for complete and up-to-date information, see https://github.com/ddh0/easy-llama
"""

import os
import sys
import uuid

VERBOSE = False             # Print all backend information as it occurs
SUPPRESS_WARNINGS = False   # Not recommended
AUTOPRINT = False           # Whether to print or return generated responses
MAX_LEN_TOKENS = 100        # Default max length of responses, in tokens

class _suppress_if_not_verbose(object):
    """
    Suppress console output from llama.cpp if easy_llama.VERBOSE is False

    Changing VERBOSE inside the WITH block may result in stdout and stderr
    being stuck to /dev/null, or other undefined behaviour

    See https://github.com/abetlen/llama-cpp-python/issues/478
    """

    # llama_cpp has its own function that does this, but
    # I believe it may be removed at some point, so here it is again

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
            # do nothing
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

def _verbose_output(text: str) -> str:
    """
    Print text to stderr if VERBOSE is True,
    prefixing with 'easy_llama: '
    """

    if VERBOSE:
        print('easy_llama: ' + text, file=sys.stderr)

def _warning_output(text: str) -> str:
    """
    Print text to stderr unless SUPPRESS_WARNINGS is True,
    prefixing with 'easy_llama: WARNING: '
    """

    if not SUPPRESS_WARNINGS:
        print('easy_llama: WARNING: ' + text, file=sys.stderr)


class Model(object):
    """
    Abstraction of a Llama model
    """

    def __init__(self, model_path: str, context_length: int=4096, seed: int=-1) -> None:
        """
        Initialize a Llama model from a file.

        Model must be in GGUF format.

        Note that this defaults to a context length of 4096. This should be changed to
        2048 for Llama 1 models, or other models with different context lengths.
        """

        _verbose_output("init: begin initializing")

        if type(model_path) is not str:
            raise TypeError('model_path should be a string, not %s' % type(model_path))
    
        if os.path.isdir(model_path):
            raise IsADirectoryError('the given model_path \'%s\' is a directory, not a file' % model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError('the given model_path \'%s\' does not exist' % model_path)
        
        if type(context_length) is not int:
            raise TypeError("context_length should be int, not %s" % type(context_length))
        
        self.context_length = context_length

        _verbose_output("init: basic checks passed. will now load model...")

        with _suppress_if_not_verbose():
            self._internal_model: llama_cpp.Llama = llama_cpp.Llama(model_path=model_path,
                                                                    n_ctx=self.context_length,
                                                                    n_gpu_layers=1,
                                                                    seed=seed,
                                                                    use_mmap=False,
                                                                    use_mlock=False,
                                                                    logits_all=True,
                                                                    n_batch=512,
                                                                    n_threads=4,
                                                                    mul_mat_q=True,
                                                                    verbose=VERBOSE)
        
        _verbose_output("init: model loaded")
    

    def greedy(self, prompt: str, max_length: int=MAX_LEN_TOKENS,
                 stop_sequences: list[str] | str | None=None) -> str | None:
        """
        Given a prompt, return a generated string using greedy sampling,
        where the most likely token is always chosen.

        The following parameters are optional:
        
        max_length: maximum length of generated string in tokens.
        Default is set in the easy_llama.MAX_LEN_TOKENS
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        if type(prompt) is not str:
           raise TypeError("prompt should be string, not %s" % type(prompt))

        if type(max_length) is not int:
           raise TypeError("max_length should be int, not %s" % type(max_length))
       
        if max_length > self.context_length:
           _warning_output("max_length is greater than context_length")

        if type(stop_sequences) not in [list, str, type(None)]:
           raise TypeError("stop_sequence should be list or str or None, not %s" % type(stop_sequences))

        if type(stop_sequences) is list:
            for item in stop_sequences:
                if type(item) is not str:
                    raise TypeError("item \'%s\' in stop_sequences list is not of type str" \
                                    % repr(item))

        _verbose_output("basic checks passed. begin generation...")
        
        with _suppress_if_not_verbose():
            output = self._internal_model.create_completion(prompt,
                                                          max_tokens=max_length,
                                                          top_p=0,
                                                          top_k=1,
                                                          stream=False,
                                                          stop=stop_sequences,
                                                          repeat_penalty=1
                                                          )['choices'][0]['text']
        # outside of with block so print will work
        if AUTOPRINT:
            print(output)
        else:
            return output


    def standard(self, prompt: str, max_length: int=MAX_LEN_TOKENS,
                 stop_sequences: list[str] | str | None=None) -> str | None:
        """
        Given a prompt, return a generated string using the llama.cpp default
        sampling (as implemented in llama-cpp-python).

        The following parameters are optional:
        
        max_length: maximum length of generated string in tokens.
        Default is set in the easy_llama.MAX_LEN_TOKENS
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        if type(prompt) is not str:
           raise TypeError("prompt should be string, not %s" % type(prompt))

        if type(max_length) is not int:
           raise TypeError("max_length should be int, not %s" % type(max_length))
       
        if max_length > self.context_length:
           _warning_output("max_length is greater than context_length")

        if type(stop_sequences) not in [list, str, type(None)]:
           raise TypeError("stop_sequence should be list or str or None, not %s" % type(stop_sequences))

        if type(stop_sequences) is list:
            for item in stop_sequences:
                if type(item) is not str:
                    raise TypeError("item \'%s\' in stop_sequences list is not of type str" \
                                    % repr(item))

        _verbose_output("basic checks passed. begin generation...")
        
        with _suppress_if_not_verbose():
            output = self._internal_model.create_completion(prompt,
                                                          max_tokens=max_length,
                                                          stream=False,
                                                          stop=stop_sequences,
                                                          )['choices'][0]['text']
        # outside of with block so print will work
        if AUTOPRINT:
            print(output)
        else:
            return output


    def contrastive(self, prompt: str, max_length: int=MAX_LEN_TOKENS,
                    stop_sequences: list[str] | str | None=None) -> str | None:
        """
        Given a prompt, return a generated string using constrastive search.

        For more information on contrastive search, see here:
        https://huggingface.co/blog/introducing-csearch

        The following parameters are optional:
        
        max_length: maximum length of generated string in tokens.
        Default is set in the easy_llama.MAX_LEN_TOKENS
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        if type(prompt) is not str:
           raise TypeError("prompt should be string, not %s" % type(prompt))

        if type(max_length) is not int:
           raise TypeError("max_length should be int, not %s" % type(max_length))
       
        if max_length > self.context_length:
           _warning_output("max_length is greater than context_length")

        if type(stop_sequences) not in [list, str, type(None)]:
           raise TypeError("stop_sequence should be list or str or None, not %s" % type(stop_sequences))

        if type(stop_sequences) is list:
            for item in stop_sequences:
                if type(item) is not str:
                    raise TypeError("item \'%s\' in stop_sequences list is not of type str" \
                                    % repr(item))

        _verbose_output("basic checks passed. begin generation...")
        
        with _suppress_if_not_verbose():
            output = self._internal_model.create_completion(prompt,
                                                          max_tokens=max_length,
                                                          top_k=4,
                                                          presence_penalty=0.6,
                                                          stream=False,
                                                          stop=stop_sequences,
                                                          repeat_penalty=1
                                                          )['choices'][0]['text']
        
        # outside of with block so print will work
        if AUTOPRINT:
            print(output)
        else:
            return output

    def testing(self, prompt: str, max_length: int=MAX_LEN_TOKENS,
                    stop_sequences: list[str] | str | None=None) -> str | None:
        """
        Given a prompt, return a generated string using experimental settings.

        This is useful for testing out different sampling settings without changing
        the default functions.

        The following parameters are optional:
        
        max_length: maximum length of generated string in tokens.
        Default is set in the easy_llama.MAX_LEN_TOKENS
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        if type(prompt) is not str:
           raise TypeError("prompt should be string, not %s" % type(prompt))

        if type(max_length) is not int:
           raise TypeError("max_length should be int, not %s" % type(max_length))
       
        if max_length > self.context_length:
           _warning_output("max_length is greater than context_length")

        if type(stop_sequences) not in [list, str, type(None)]:
           raise TypeError("stop_sequence should be list or str or None, not %s" % type(stop_sequences))

        if type(stop_sequences) is list:
            for item in stop_sequences:
                if type(item) is not str:
                    raise TypeError("item \'%s\' in stop_sequences list is not of type str" \
                                    % repr(item))

        _verbose_output("basic checks passed. begin generation...")
        
        with _suppress_if_not_verbose():
            output = self._internal_model.create_completion(prompt,
                                                          max_tokens=max_length,
                                                          top_k=4,
                                                          presence_penalty=0.677,
                                                          stream=False,
                                                          stop=stop_sequences,
                                                          )['choices'][0]['text']
        # outside of with block so print will work
        if AUTOPRINT:
            print(output)
        else:
            return output

class Prompting:
    """
    Functions available:
    - Prompting.chat(string)
    - Prompting.assist(string)
    - Prompting.instruct(string)

    Prompting.assist is probably what you want.
    See each function's docstring for more info.

    Provide functionality to convert a string like this:
    
    "How do I make pasta?"
    
    to a string like this:

    "A chat between a curious human and an artificial intelligence assistant.
    The assistant gives helpful, detailed, and polite answers to the user's
    questions. ### Human: How do I make pasta? ### Assistant: "
    """

    def chat(string: str) -> str:
        """Return the given string in a custom chatbot prompt format
        
        Broadly useful for chatbots"""

        return "A chat between a human and an artificial intelligence chatbot. The chatbot is polite, intelligent, honest, and casual.\n" + \
               "### Human: " + string + " ### Chatbot: "

    def assist(string: str) -> str:
        """Return the given string in Tim Dettmers' Guanaco prompt format
        
        Broadly useful for assistant interactions"""

        return "A chat between a curious human and an artificial intelligence assistant. " + \
               "The assistant gives helpful, detailed, and polite answers to the user's questions.\n" + \
               "### Human: " + string + " ### Assistant: "

    def instruct(string: str) -> str:
        """Return the given string in Stanford's Alpaca prompt format
        
        Broadly useful for Instruct-Response interactions"""

        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n" + \
               "\n### Instruction:\n" + string + \
               "\n\n### Response:\n"

class Thread(object):
    """
    Provide functionality to facilitate conversation threads with a model
    (i.e. remembering past messages)

    Allows for arbitrary number of threads to be held simultaneously
    and stored on disk, and for moving interactions between models
    """

    def __init__(self) -> None:
        self.uuid = uuid.uuid4()
        


    
    def msg(self):
        pass
