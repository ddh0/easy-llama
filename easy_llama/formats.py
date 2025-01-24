# formats.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""This file provides functionality for defining prompt formats, which are used to define how
the input to a Llama model should be structured."""

from collections.abc import Callable
from typing          import Optional
from datetime        import datetime

def _call_or_return(obj: str | Callable[..., str]) -> str:
    if callable(obj):
        ret = obj()
        if not isinstance(ret, str):
            raise TypeError(
                f'_call_or_return: obj must be a string or a callable that returns a string '
                f'(the callable returned {repr(type(ret))})'
            )
        return ret
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError(
            f'_call_or_return: obj must be a string or a callable that returns a string '
            f'(the object\'s type was {repr(type(obj))})'
        )

class PromptFormat:
    """Define a prompt format"""

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
    
    def __repr__(self) -> str:
        return (
            f"PromptFormat("
            f"system_prefix={self._system_prefix!r}, "
            f"system_prompt={self._system_prompt!r}, "
            f"system_suffix={self._system_suffix!r}, "
            f"user_prefix={self._user_prefix!r}, "
            f"user_suffix={self._user_suffix!r}, "
            f"bot_prefix={self._bot_prefix!r}, "
            f"bot_suffix={self._bot_suffix!r}"
            f")"
        )
    
    def system_prefix(self) -> str:
        """Get the system prompt prefix"""
        return _call_or_return(self._system_prefix)

    def system_prompt(self) -> str:
        """Get the system prompt"""
        return _call_or_return(self._system_prompt)

    def system_suffix(self) -> str:
        """Get the system prompt suffix"""
        return _call_or_return(self._system_suffix)

    def user_prefix(self) -> str:
        """Get the user message prefix"""
        return _call_or_return(self._user_prefix)

    def user_suffix(self) -> str:
        """Get the user message suffix"""
        return _call_or_return(self._user_suffix)

    def bot_prefix(self) -> str:
        """Get the bot message prefix"""
        return _call_or_return(self._bot_prefix)

    def bot_suffix(self) -> str:
        """Get the bot message suffix"""
        return _call_or_return(self._bot_suffix)

def _llama3_today_date() -> str:
    return datetime.today().strftime('%d %B %Y')

def BlankFormat() -> PromptFormat:
    return PromptFormat(
        system_prefix='',
        system_prompt='',
        system_suffix='',
        user_prefix='',
        user_suffix='',
        bot_prefix='',
        bot_suffix=''
    )

def AlpacaFormat() -> PromptFormat:
    return PromptFormat(
        system_prefix='',
        system_prompt='',
        system_suffix='',
        user_prefix='',
        user_suffix='',
        bot_prefix='',
        bot_suffix=''
    )

def Llama3Format(system_prompt: Optional[str] = None) -> PromptFormat:
    return PromptFormat(
        system_prefix='<|start_header_id|>system<|end_header_id|>\n\n',
        system_prompt=system_prompt if system_prompt is not None else f"""Cutting Knowledge Date: December 2023
Today Date: {_llama3_today_date()}

You are a helpful assistant""",
        system_suffix='<|eot_id|>',
        user_prefix='<|start_header_id|>user<|end_header_id|>\n\n',
        user_suffix='<|eot_id|>',
        bot_prefix='<|start_header_id|>assistant<|end_header_id|>\n\n',
        bot_suffix='<|eot_id|>'
    )

def ChatMLFormat(system_prompt: Optional[str] = None) -> PromptFormat:
    return PromptFormat(
        system_prefix='<|im_start|>system\n',
        system_prompt=system_prompt if system_prompt is not None else '',
        system_suffix='<|im_end|>\n',
        user_prefix='<|im_start|>user\n',
        user_suffix='<|im_end|>\n',
        bot_prefix='<|im_start|>assistant\n',
        bot_suffix='<|im_end|>\n'
    )
