# easy_llama.py
# Python 3.11.6

"""
Simple, on-device text inference using language models

for complete and up-to-date information, see https://github.com/ddh0/easy-llama
"""

import os
import sys
import uuid
import metadata_gguf

VERBOSE = False             # Print all backend information as it occurs
SUPPRESS_WARNINGS = False   # Not recommended
AUTOPRINT = False           # Whether to print or return generated responses
MAX_LEN_TOKENS = 100        # Default max length of responses, in tokens
NUM_GPU_LAYERS = 1          # Leave at 1 for Apple Silicon, tweak for CUDA and ROCm, set to 0 for OpenBLAS
NUM_THREADS = int(os.cpu_count() / 2) # Half of logical cores
                                      # This is equal to the number of physical cores
                                      # on an Intel CPU with hyperthreading, and for the
                                      # most common M-series chips, it is equal to the number of P-cores

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

        assert isinstance(model_path, str), 'model_path should be a string, not %s' % type(model_path)
        assert not os.path.isdir(model_path), 'the given model_path \'%s\' is a directory, not a file' % model_path
        assert os.path.exists(model_path), 'the given model_path \'%s\' does not exist' % model_path
        assert isinstance(context_length, int), 'context_length should be int, not %s' % type(context_length)
        
        # Read file to determine n_ctx_train
        self.metadata_dict = metadata_gguf.load_metadata(model_path)
        
        try:
            self.context_length = self.metadata_dict['llama.context_length']
        except KeyError:
            _warning_output("Unable to detemine model's native context length. Defaulting to 4096.")
            self.context_length = 4096
        
        self.filename = os.path.split(model_path)[1] # Should work regardless of OS

        _verbose_output("init: basic checks passed. will now load model...")

        with _suppress_if_not_verbose():
            self._internal_model: llama_cpp.Llama = llama_cpp.Llama(model_path=model_path,
                                                                    n_ctx=self.context_length,
                                                                    n_gpu_layers=NUM_GPU_LAYERS,
                                                                    seed=seed,
                                                                    use_mmap=False,
                                                                    use_mlock=False,
                                                                    logits_all=True,
                                                                    n_batch=512,
                                                                    n_threads=NUM_THREADS,
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
        Default is set in easy_llama.MAX_LEN_TOKENS
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert isinstance(max_length, int), 'max_length should be int, not %s' % type(max_length)
        assert type(stop_sequences) in [list, str, type(None)], 'stop_sequences should be list, str, or None'
        # Why is NoneType not accessible directly? I don't know.

        if type(stop_sequences) is list:
            for item in stop_sequences:
                assert isinstance(item, str), "some item in stop_sequences list is not of type str"

        if max_length > self.context_length:
           _warning_output("max_length is greater than context_length")

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
        Default is set in easy_llama.MAX_LEN_TOKENS
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert isinstance(max_length, int), 'max_length should be int, not %s' % type(max_length)
        assert type(stop_sequences) in [list, str, type(None)], 'stop_sequences should be list, str, or None'

        if type(stop_sequences) is list:
            for item in stop_sequences:
                assert isinstance(item, str), "some item in stop_sequences list is not of type str"

        if max_length > self.context_length:
           _warning_output("max_length is greater than context_length")

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

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert isinstance(max_length, int), 'max_length should be int, not %s' % type(max_length)
        assert type(stop_sequences) in [list, str, type(None)], 'stop_sequences should be list, str, or None'

        if type(stop_sequences) is list:
            for item in stop_sequences:
                assert isinstance(item, str), "some item in stop_sequences list is not of type str"

        if max_length > self.context_length:
           _warning_output("max_length is greater than context_length")

        _verbose_output("basic checks passed. begin generation...")
        
        with _suppress_if_not_verbose():
            output = self._internal_model.create_completion(prompt,
                                                          max_tokens=max_length,
                                                          top_k=4,
                                                          presence_penalty=0.6,
                                                          stream=False,
                                                          stop=stop_sequences,
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
        Default is set in easy_llama.MAX_LEN_TOKENS
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert isinstance(max_length, int), 'max_length should be int, not %s' % type(max_length)
        assert type(stop_sequences) in [list, str, type(None)], 'stop_sequences should be list, str, or None'

        if type(stop_sequences) is list:
            for item in stop_sequences:
                assert isinstance(item, str), "some item in stop_sequences list is not of type str"

        if max_length > self.context_length:
           _warning_output("max_length is greater than context_length")

        _verbose_output("basic checks passed. begin generation...")
        
        with _suppress_if_not_verbose():
            output = self._internal_model.create_completion(prompt,
                                                          max_tokens=max_length,
                                                          top_k=4,
                                                          presence_penalty=0.65,
                                                          stream=False,
                                                          stop=stop_sequences,
                                                          )['choices'][0]['text']
        # outside of with block so print will work
        if AUTOPRINT:
            print(output)
        else:
            return output

class Format(object):
    """Base class for prompt format helpers"""

class Alpaca(Format):
    system_str = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
    user_prefix_str = '### Instruction:\n'
    bot_prefix_str = '\n\n### Response:\n'
    bot_postfix_str = '\n\n'
    stop_sequences = ['###', '\nInstruction:']

    def wrap(text: str) -> str:
        return Alpaca.system_str + Alpaca.user_prefix_str + \
               text + Alpaca.bot_prefix_str

class Llama2(Format):
    system_str = '[INST] <<SYS>>\nAnswer the questions.\n<</SYS>>\n\n'
    user_prefix_str = ''
    bot_prefix_str = ' [/INST] '
    bot_postfix_str = '</s><s>[INST] '
    stop_sequences = ['</s>', '[INST]']

    def wrap(text: str) -> str:
        return Llama2.system_str + Llama2.user_prefix_str + \
               text + Llama2.bot_prefix_str

class Guanaco(Format):
    system_str = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions.\n\n'
    user_prefix_str = '### Human: '
    bot_prefix_str = '\n### Assistant: '
    bot_postfix_str = '\n'
    stop_sequences = ['###', '\nHuman:']

    def wrap(text: str) -> str:
        return Guanaco.system_str + Guanaco.user_prefix_str + \
               text + Guanaco.bot_prefix_str

class Samantha(Format):
    system_str = 'You are Samantha, a sentient AI companion.\n\n'
    user_prefix_str = 'USER: '
    bot_prefix_str = '\nASSISTANT: '
    bot_postfix_str = '\n'
    stop_sequences = ['\nUSER:']

    def wrap(text: str) -> str:
        return Samantha.system_str + Samantha.user_prefix_str + \
               text + Samantha.bot_prefix_str

class OrcaMini(Format):
    system_str = '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n'
    user_prefix_str = '### User:\n'
    bot_prefix_str = '\n\n### Assistant:\n'
    bot_postfix_str = '\n\n'
    stop_sequences = ['###', '\nUser:']

    def wrap(text: str) -> str:
        return OrcaMini.system_str + OrcaMini.user_prefix_str + \
               text + OrcaMini.bot_prefix_str

class Testing(Format):
    system_str = ''
    user_prefix_str = ''
    bot_prefix_str = ''
    bot_postfix_str = ''
    stop_sequences = None

    def wrap(text: str) -> str:
        return Testing.system_str + Testing.user_prefix_str + \
               text + Testing.bot_prefix_str

class Thread(object):
    """
    Provide functionality to facilitate conversation threads with a model
    (i.e. remembering past messages)

    Allows for arbitrary number of threads to be held simultaneously
    and stored on disk, and for moving interactions between models
    """

    def __init__(self, model: Model, format: Format, interactive: bool=False) -> None:
        self.uuid = uuid.uuid4()
        self.model = model # Models can be hot-swapped while keeping context!
        self.format = format
        self.interactive = interactive

        assert isinstance(self.model, Model), 'Thread: model should be an instance of easy_llama.Model'
        assert hasattr(self.format, 'system_str'), 'Thread: format is missing attribute system_str'
        assert hasattr(self.format, 'user_prefix_str'), 'Thread: format is missing attribute user_prefix_str'
        assert hasattr(self.format, 'bot_prefix_str'), 'Thread: format is missing attribute bot_prefix_str'
        assert hasattr(self.format, 'bot_postfix_str'), 'Thread: format is missing attribute bot_postfix_str'
        assert hasattr(self.format, 'stop_sequences'), 'Thread: format is missing attribute stop_sequences'
        assert isinstance(self.interactive, bool), 'Thread: interactive should be bool (True or False)'
    
    def msg(self):
        pass
