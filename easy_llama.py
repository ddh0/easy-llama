# easy_llama.py
# Python 3.11.6

"""
Simple, on-device text inference in Python

for complete and up-to-date information, see https://github.com/ddh0/easy-llama
"""

import os
import sys
import metadata_gguf


VERBOSE = False             # Print all backend information as it occurs (llama.cpp)
SUPPRESS_WARNINGS = False   # At your own risk
MAX_LEN_TOKENS = 200        # Default max length of all generations, in tokens
NUM_GPU_LAYERS = 1          # Leave at 1 for Apple Silicon, tweak for CUDA and ROCm, set to 0 for OpenBLAS / CPU
NUM_THREADS = int(os.cpu_count() / 2) # Default to half of logical cores
                                      # This is equal to the number of physical cores
                                      # on most CPUs with hyperthreading, and for the
                                      # most common M-series chips, it is equal to the number of P-cores


class suppress_if_not_verbose(object):
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


with suppress_if_not_verbose():
        import llama_cpp


def print_warning(text: str) -> str:
    """
    Print text to stderr unless SUPPRESS_WARNINGS is True,
    prefixing with 'easy_llama: WARNING: '
    """

    if not SUPPRESS_WARNINGS:
        print('easy_llama: WARNING: ' + text, file=sys.stderr)


class Model(object):
    """
    Abstraction of a llama model
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize a llama model from a file.

        Model must be in GGUF format.

        easy_llama will automatically determine the model's trained
        context length from the GGUF metadata.
        """

        assert isinstance(model_path, str), 'model_path should be a string, not %s' % type(model_path)
        assert not os.path.isdir(model_path), 'the given model_path \'%s\' is a directory, not a file' % model_path
        assert os.path.exists(model_path), 'the given model_path \'%s\' does not exist' % model_path
        
        self.metadata: dict = metadata_gguf.load_metadata(model_path)
        
        assert isinstance(self.metadata['llama.context_length'], int), \
               'GGUF metadata reports context_length is not an integer'
        if self.metadata['llama.context_length'] < 2048:
               print_warning('GGUF metadata reports an unusually small native context length (%s)' % \
                             self.metadata['llama.context_length'])
        if self.metadata['llama.context_length'] > 131072:
               print_warning('GGUF metadata reports an unusually large native context length (%s)' % \
                             self.metadata['llama.context_length'])

        self.context_length = self.metadata['llama.context_length']

        with suppress_if_not_verbose():
            self._internal_model: llama_cpp.Llama = llama_cpp.Llama(model_path=model_path,
                                                                    n_ctx=self.context_length,
                                                                    n_gpu_layers=NUM_GPU_LAYERS,
                                                                    seed=-1,
                                                                    use_mmap=False,
                                                                    use_mlock=False,
                                                                    logits_all=True,
                                                                    n_batch=512,
                                                                    n_threads=NUM_THREADS,
                                                                    mul_mat_q=True,
                                                                    verbose=VERBOSE)
    

    def trim(self, text: str, overwrite: str=None) -> str:
        """
        Trim the given text to the context length of this model,
        leaving room for two extra tokens.
        
        Optionally overwrite the oldest tokens with the text given in the 'overwrite'
        parameter (useful for keeping system prompt in context.)

        Assumes UTF-8.

        Does nothing if the text is equal or shorter to (context_length - 2).
        """
        trim_length = self.context_length - 2

        tokens_list = self._internal_model.tokenize(text.encode('utf-8'))
        if len(tokens_list) <= trim_length:
            return text
        if len(tokens_list) > trim_length and overwrite is None:
            # Cut to context length
            tokens_list = tokens_list[-trim_length:]
            return self._internal_model.detokenize(tokens_list).decode()
        if len(tokens_list) > self.context_length and overwrite is not None:
            # Cut to context length and overwrite the oldest tokens with overwrite
            tokens_list = tokens_list[-trim_length:]
            overwrite_tokens = self._internal_model.tokenize(overwrite.encode('utf-8'))
            tokens_list[0:len(overwrite_tokens)] = overwrite_tokens
            return self._internal_model.detokenize(tokens_list).decode()
    

    def get_length(self, text: str) -> int:
        """
        Return the length of the given text in tokens,
        according to this model.

        Assumes UTF-8.
        """
        return len(self._internal_model.tokenize(text.encode('utf-8')))


    #
    # llama.h says:
    #
    # /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details
    # 
    # ...which says:
    #
    # frequency_penalty
    # ------------------
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # their existing frequency in the text so far, decreasing the model's
    # likelihood to repeat the same line verbatim.
    #
    # presence_penalty
    # -----------------
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # whether they appear in the text so far, increasing the model's
    # likelihood to talk about new topics.
    #
    # My own summary
    # ------------------------------------------
    # frequency_penalty
    # |
    # | - Too low: model often gets stuck repeating same sentences
    # |
    # | - Too high: model becomes annoyingly quirky and rambling
    # |
    # | - Reasonable starting point: 0.075
    # |
    # 
    # presence_penalty
    # |
    # | - Too low: highly robotic and predictable
    # |
    # | - Too high: does not make sense in context
    # |
    # | - Reasonable starting point: 0.5
    # |
    #


    def generate(self, prompt: str, stop_sequences: list[str] | str | None=None) -> str:
        """
        Given a prompt, return a generated string using constrastive search
        which is the method easy-llama recommendeds for most cases.

        Also applies a very slight frequency penalty to prevent repetition.

        For more information on contrastive search, see here:
        https://huggingface.co/blog/introducing-csearch

        The following parameter is optional:
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert type(stop_sequences) in [list, str, type(None)], 'stop_sequences should be list, str, or None'

        if type(stop_sequences) is list:
            for item in stop_sequences:
                assert isinstance(item, str), "some item in stop_sequences list is not of type str"

        if MAX_LEN_TOKENS > self.context_length:
           print_warning("easy_llama.MAX_LEN_TOKENS is greater than this model's context length")
        
        with suppress_if_not_verbose():
            output = self._internal_model.create_completion(prompt,
                                                            max_tokens=MAX_LEN_TOKENS,
                                                            top_k=4,
                                                            frequency_penalty=0.025,
                                                            presence_penalty=0.25,
                                                            stream=False,
                                                            stop=stop_sequences,
                                                            )['choices'][0]['text']
        return output


    def greedy(self, prompt: str, stop_sequences: list[str] | str | None=None) -> str:
        """
        Given a prompt, return a generated string using greedy sampling,
        where the most likely token is always chosen.

        The following parameter is optional:
        
        stop_sequences: list of strings at which to end the generation early (right before),
        can also be a single string or None. Default is None
        """

        assert isinstance(prompt, str), 'prompt should be string, not %s' % type(prompt)
        assert type(stop_sequences) in [list, str, type(None)], 'stop_sequences should be list, str, or None'

        if type(stop_sequences) is list:
            for item in stop_sequences:
                assert isinstance(item, str), "some item in stop_sequences list is not of type str"

        if MAX_LEN_TOKENS > self.context_length:
           print_warning("easy_llama.MAX_LEN_TOKENS is greater than this model's context length")
        
        with suppress_if_not_verbose():
            output = self._internal_model.create_completion(prompt,
                                                            max_tokens=MAX_LEN_TOKENS,
                                                            top_p=0,
                                                            top_k=1,
                                                            stream=False,
                                                            stop=stop_sequences,
                                                            repeat_penalty=1
                                                            )['choices'][0]['text']
        return output


class Format:
    """
    Base class for prompt format helpers

    If you make your own format, it does not need to inherit from this class, but it
    does need to have the following attributes in order to work with easy_llama.Thread:
    - system_str: str
    _ user_prefix_str: str
    - bot_prefix_str: str
    - bot_postfix_str: str
    - stop_sequences: list[str] | str | None

    For convenience, it should also have the .wrap() method, which should wrap a given
    string in a format suitable for single-turn completion.
    """


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
    system_str = '<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n'
    user_prefix_str = ''
    bot_prefix_str = ' [/INST] '
    bot_postfix_str = '</s><s>[INST] '
    stop_sequences = ['</s>', '[INST]']

    def wrap(text: str) -> str:
        return Llama2.system_str + Llama2.user_prefix_str + \
               text + Llama2.bot_prefix_str


class Vicuna(Format):
    system_str = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n\n'
    user_prefix_str = 'USER: '
    bot_prefix_str = '\nASSISTANT: '
    bot_postfix_str = '\n'
    stop_sequences = ['</s>', '\nUSER:']

    def wrap(text: str) -> str:
        return Vicuna.system_str + Vicuna.user_prefix_str + \
               text + Vicuna.bot_prefix_str


class MistralOrca(Format):
    system_str = '<|im_start|>system\nYou are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!\n<|im_end|>\n'
    user_prefix_str = '<|im_start|>user\n'
    bot_prefix_str = '<|im_end|>\n<|im_start|>assistant\n'
    bot_postfix_str = '<|im_end|>\n'
    stop_sequences = ['<|im_end|>', '<|im_start|>']

    def wrap(text: str) -> str:
        return MistralOrca.system_str + MistralOrca.user_prefix_str + \
               text + MistralOrca.bot_prefix_str


class Jackalope(Format):
    system_str = '<|im_start|>system\nYou are JackalopeAI. Write out your reasoning step-by-step to be sure you get the right answers!\n<|im_end|>\n'
    user_prefix_str = '<|im_start|>user\n'
    bot_prefix_str = '<|im_end|>\n<|im_start|>assistant\n'
    bot_postfix_str = '<|im_end|>\n'
    stop_sequences = ['<|im_end|>', '<|im_start|>']

    def wrap(text: str) -> str:
        return Jackalope.system_str + Jackalope.user_prefix_str + \
               text + Jackalope.bot_prefix_str


class Dolphin(Format):
    system_str = '<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n'
    user_prefix_str = '<|im_start|>user\n'
    bot_prefix_str = '<|im_end|>\n<|im_start|>assistant\n'
    bot_postfix_str = '<|im_end|>\n'
    stop_sequences = ['<|im_end|>', '<|im_start|>']

    def wrap(text: str) -> str:
        return Dolphin.system_str + Dolphin.user_prefix_str + \
               text + Dolphin.bot_prefix_str


class Samantha(Format):
    system_str = '<|im_start|>system\nYou are a caring and empathetic sentient AI companion named Samantha.<|im_end|>\n'
    user_prefix_str = '<|im_start|>user\n'
    bot_prefix_str = '<|im_end|>\n<|im_start|>assistant\n'
    bot_postfix_str = '<|im_end|>\n'
    stop_sequences = ['<|im_start|>', '<|im_end|>']

    def wrap(text: str) -> str:
        return Samantha.system_str + Samantha.user_prefix_str + \
               text + Samantha.bot_prefix_str


class Guanaco(Format):
    system_str = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions.\n\n'
    user_prefix_str = '### Human: '
    bot_prefix_str = '\n### Assistant: '
    bot_postfix_str = '\n'
    stop_sequences = ['###', '\nHuman:']

    def wrap(text: str) -> str:
        return Guanaco.system_str + Guanaco.user_prefix_str + \
               text + Guanaco.bot_prefix_str


class OrcaMini(Format):
    system_str = '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n'
    user_prefix_str = '### User:\n'
    bot_prefix_str = '\n\n### Assistant:\n'
    bot_postfix_str = '\n\n'
    stop_sequences = ['###', '\nUser:']

    def wrap(text: str) -> str:
        return OrcaMini.system_str + OrcaMini.user_prefix_str + \
               text + OrcaMini.bot_prefix_str


class Metharme(Format):
    # This is used for Pygmalion models, and is different from ChatML formats like Dolphin, etc
    system_str = '<|system|>You are an artificial intelligence assistant. You are helpful, detailed, and polite.'
    user_prefix_str = '<|user|>'
    bot_prefix_str = '<|model|>'
    bot_postfix_str = ''
    stop_sequences = ['<|user|>', '<|system|>']

    def wrap(text: str) -> str:
        return Metharme.system_str + Metharme.user_prefix_str + \
               text + Metharme.bot_prefix_str


class Thread(object):
    """
    Provide functionality to facilitate conversation with a model
    (i.e. remembering past messages.)

    You must specify a format, which must have the following attributes:
    - system_str: str
    - user_prefix_str: str
    - bot_prefix_str: str
    - bot_postfix_str: str
    - stop_sequences: list[str] | str | None

    After initializing a Thread, you can change the .model attribute to refer to a different model,
    while keeping your Thread intact.
    
    As a hypothetical example:
    
    >>> Airoboros = Model('./airoboros-l2-13b-3.0.Q6_K.gguf')
    >>> MyThread = Thread(model=Airoboros, format=Llama2)
    >>> MyThread.msg("What is your name?")
    'I am Airoboros.'
    
    ...then, later on...

    >>> MyThread.model = none    # Optional, to avoid loading
    >>> Airoboros = None         # two models in memory at once
    >>> Guanaco = Model('./guanaco-13b.Q8_0.gguf')
    >>> MyThread.model = Guanaco
    >>> MyThread.msg("What is your name now?")
    'I apologize, what I said earlier is incorrect. I am Guanaco.'

    The full context (i.e transcript of the conversation) is accessible via Thread.context
    (a string).

    Remember to set easy_llama.MAX_LEN_TOKENS to suit your needs.
    """

    def __init__(self, model: Model, format: Format) -> None:

        assert isinstance(model, Model), 'Thread: model should be an instance of easy_llama.Model'
        assert hasattr(format, 'system_str'), 'Thread: format is missing attribute system_str'
        assert hasattr(format, 'user_prefix_str'), 'Thread: format is missing attribute user_prefix_str'
        assert hasattr(format, 'bot_prefix_str'), 'Thread: format is missing attribute bot_prefix_str'
        assert hasattr(format, 'bot_postfix_str'), 'Thread: format is missing attribute bot_postfix_str'
        assert hasattr(format, 'stop_sequences'), 'Thread: format is missing attribute stop_sequences'

        self.model: Model = model
        self.format: Format = format
        self.context = self.format.system_str
    
    
    def msg(self, prompt: str) -> str:
        """
        Send a message to this Thread, which will be remembered in
        this Thread's context.
        """

        assert isinstance(prompt, str), 'Thread.msg: prompt should be str'

        self.context += (self.format.user_prefix_str + prompt + self.format.bot_prefix_str)
        self.context = self.model.trim(self.context, overwrite=self.format.system_str)
        output = self.model.generate(self.context, stop_sequences=self.format.stop_sequences)
        self.context += (output + self.format.bot_postfix_str)

        return output


    def interact(self) -> None:
        """
        Start an interactive chat session.
        Designed for use in interactive python shells.
        """
        print()
        try:
            while True:
                prompt = input('  > ')
                print() # Put the cursor where the generated text is about to appear
                if prompt == '':
                    # No input from user - assume user wants model to keep talking
                    self.context += (self.format.bot_prefix_str)
                    self.context = self.model.trim(self.context, overwrite=self.format.system_str)
                    output = self.model.generate(self.context, stop_sequences=self.format.stop_sequences)
                    self.context += (output + self.format.bot_postfix_str)
                else:
                    self.context += (self.format.user_prefix_str + prompt + self.format.bot_prefix_str)
                    self.context = self.model.trim(self.context, overwrite=self.format.system_str)
                    output = self.model.generate(self.context, stop_sequences=self.format.stop_sequences)
                    self.context += (output + self.format.bot_postfix_str)

                
                # Strip all leading and trailing spaces and newlines
                # This is only done for Thread.interact()
                while output.startswith(' ') or output.startswith('\n'):
                    output = output[1:]
                while output.endswith('\n') or output.startswith(' '):
                    output = output[:-1]

                print(output + '\n')
        
        except KeyboardInterrupt:
            print('\n')
            return


    def reset(self) -> None:
        """Reset the Thread's context"""
        self.context = self.format.system_str
