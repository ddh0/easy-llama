# formats.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

from collections.abc import Callable

def _call_or_return(obj: str | Callable[..., str]) -> str:
    if callable(obj):
        return obj()
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError(
            f'_call_or_return: obj must be a string or a callable that returns '
            f'a string (got {repr(type(obj))})'
        )

class PromptFormat:
    """
    Define a prompt format
    """

    def __init__(
        self,
        system_prefix: str | Callable[..., str],
        system_prompt: str | Callable[..., str],
        system_suffix: str | Callable[..., str],
        user_prefix:   str | Callable[..., str],
        user_suffix:   str | Callable[..., str],
        bot_prefix:    str | Callable[..., str],
        bot_suffix:    str | Callable[..., str]
    ) -> None:
        self._system_prefix = system_prefix
        self._system_prompt = system_prompt
        self._system_suffix = system_suffix
        self._user_prefix   = user_prefix
        self._user_suffix   = user_suffix
        self._bot_prefix    = bot_prefix
        self._bot_suffix    = bot_suffix
    
    def system_prefix(self) -> str:
        return _call_or_return(self._system_prefix)

    def system_prompt(self) -> str:
        return _call_or_return(self._system_prompt)

    def system_suffix(self) -> str:
        return _call_or_return(self._system_suffix)

    def user_prefix(self) -> str:
        return _call_or_return(self._user_prefix)

    def user_suffix(self) -> str:
        return _call_or_return(self._user_suffix)

    def bot_prefix(self) -> str:
        return _call_or_return(self._bot_prefix)

    def bot_suffix(self) -> str:
        return _call_or_return(self._bot_suffix)

def _prompt_format_from_llama_metadata(metadata: dict) -> PromptFormat:
    if 'tokenizer.chat_template' in metadata.keys():
        pass # TODO