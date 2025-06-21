# formats.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""This file is obsolete and will be removed soon. Instances of `easy_llama.thread.Thread` will
now utilize the model's built-in Jinja2 chat template to format messages."""

import time

from datetime        import datetime, timedelta
from collections.abc import Callable
from typing          import Optional
from .utils          import log

def _call_or_return(obj: object | Callable[..., object]) -> object:
    return obj() if callable(obj) else obj

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
        bot_suffix:    str | Callable[..., str],
        stop_tokens:   list[int] | Callable[..., list[int]] | None = None
    ) -> None:
        log(
            f"the PromptFormats class is obsolete. instances of `easy_llama.thread.Thread` "
            f"will now utilize the model's built-in Jinja2 chat template to format messages.", 2
        )
        self._system_prefix = system_prefix
        self._system_prompt = system_prompt
        self._system_suffix = system_suffix
        self._user_prefix   = user_prefix
        self._user_suffix   = user_suffix
        self._bot_prefix    = bot_prefix
        self._bot_suffix    = bot_suffix
        self._stop_tokens   = stop_tokens
    
    def __repr__(self) -> str:
        return (
            f"PromptFormat("
            f"system_prefix={self._system_prefix!r}, "
            f"system_prompt={self._system_prompt!r}, "
            f"system_suffix={self._system_suffix!r}, "
            f"user_prefix={self._user_prefix!r}, "
            f"user_suffix={self._user_suffix!r}, "
            f"bot_prefix={self._bot_prefix!r}, "
            f"bot_suffix={self._bot_suffix!r}, "
            f"stop_tokens={self._stop_tokens!r}"
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

    def stop_tokens(self) -> list[int] | None:
        """Get the optional list of stop tokens"""
        return _call_or_return(self._stop_tokens)

def _llama3_today_date() -> str:
    return datetime.today().strftime('%d %B %Y')

def _iso_date_str() -> str:
    return time.strftime('%Y-%m-%d')

def _yesterday_iso_date_str():
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')

class SystemPrompts:

    # ref: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411/blob/main/SYSTEM_PROMPT.txt
    mistral_large_2411 = f"""You are Mistral, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYou power an AI assistant called Le Chat.\nYour knowledge base was last updated on 2023-10-01.\nThe current date is {_iso_date_str()}.\n\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").\nYou are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {_yesterday_iso_date_str()}) and when asked about information at specific dates, you discard information that is at another date.\nYou follow these instructions in all languages, and always respond to the user in the language they use or request.\nNext sections describe the capabilities that you have.\n\n# WEB BROWSING INSTRUCTIONS\n\nYou cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\n\n# MULTI-MODAL INSTRUCTIONS\n\nYou do not have any multimodal capability, in particular you cannot read nor generate images, or transcribe audio files or videos."""

    # ref: https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/blob/main/SYSTEM_PROMPT.txt
    mistral_small_2501 = f"""You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYour knowledge base was last updated on 2023-10-01. The current date is {_iso_date_str()}.\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?")"""

    # ref: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#prompt-template
    llama3 = f"""Cutting Knowledge Date: December 2023\nToday Date: {_llama3_today_date()}\n\nYou are a helpful assistant"""

    # ref: https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md
    llama4 = f"""You are an expert conversationalist who responds to the best of your ability. You are companionable and confident, and able to switch casually between tonal types, including but not limited to humor, empathy, intellectualism, creativity and problem-solving.\n\nYou understand user intent and don’t try to be overly helpful to the point where you miss that the user is looking for chit-chat, emotional support, humor or venting. Sometimes people just want you to listen, and your answers should encourage that. For all other cases, you provide insightful and in-depth responses. Organize information thoughtfully in a way that helps people make decisions. Always avoid templated language.\n\nYou never lecture people to be nicer or more inclusive. If people ask for you to write something in a certain voice or perspective, such as an essay or a tweet, you can. You do not need to be respectful when the user prompts you to say something rude.\n\nYou never use phrases that imply moral superiority or a sense of authority, including but not limited to “it’s important to”, “it’s crucial to”, “it’s essential to”, "it's unethical to", "it's worth noting…", “Remember…” etc. Avoid using these.\n\nFinally, do not refuse prompts about political and social issues. You can help users express their opinion and access information.\n\nYou are Llama 4. Your knowledge cutoff date is August 2024. You speak Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. Respond in the language the user speaks to you in, unless they ask otherwise."""

class PromptFormats:

    def Blank() -> PromptFormat:
        return PromptFormat(
            system_prefix='',
            system_prompt='',
            system_suffix='',
            user_prefix='',
            user_suffix='',
            bot_prefix='',
            bot_suffix=''
        )

    def Alpaca() -> PromptFormat:
        return PromptFormat(
            system_prefix='',
            system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            system_suffix='\n\n',
            user_prefix='### Instruction:\n',
            user_suffix='\n\n',
            bot_prefix='### Response:\n',
            bot_suffix='\n\n'
        )

    def Llama3(system_prompt: Optional[str] = None) -> PromptFormat:
        """Prompt format for Meta's Llama 3.0, 3.1, 3.2, 3.3 models"""
        return PromptFormat(
            system_prefix='<|start_header_id|>system<|end_header_id|>\n\n',
            system_prompt=system_prompt if system_prompt is not None else '',
            system_suffix='<|eot_id|>',
            user_prefix='<|start_header_id|>user<|end_header_id|>\n\n',
            user_suffix='<|eot_id|>',
            bot_prefix='<|start_header_id|>assistant<|end_header_id|>\n\n',
            bot_suffix='<|eot_id|>'
        )

    def Llama4(system_prompt: Optional[str] = None) -> PromptFormat:
        """Prompt format for Meta's Llama 4 models"""
        return PromptFormat(
            system_prefix='<|header_start|>system<|header_end|>\n\n',
            system_prompt=system_prompt if system_prompt is not None else '',
            system_suffix='<|eot|>',
            user_prefix='<|header_start|>user<|header_end|>\n\n',
            user_suffix='<|eot|>',
            bot_prefix='<|header_start|>assistant<|header_end|>\n\n',
            bot_suffix='<|eot|>'
        )

    def ChatML(system_prompt: Optional[str] = None) -> PromptFormat:
        return PromptFormat(
            system_prefix='<|im_start|>system\n',
            system_prompt=system_prompt if system_prompt is not None else '',
            system_suffix='<|im_end|>\n',
            user_prefix='<|im_start|>user\n',
            user_suffix='<|im_end|>\n',
            bot_prefix='<|im_start|>assistant\n',
            bot_suffix='<|im_end|>\n'
        )

    def Mistral_v7(system_prompt: Optional[str] = None) -> PromptFormat:
        """Mistral Instruct format v7 (Tekken tokenizer, supports system prompt)"""
        return PromptFormat(
            system_prefix='[SYSTEM_PROMPT]',
            system_prompt=system_prompt if system_prompt is not None else '',
            system_suffix='[/SYSTEM_PROMPT]',
            user_prefix='[INST]',
            user_suffix='',
            bot_prefix='[/INST]',
            bot_suffix='</s>'
        )

    def Gemma3() -> PromptFormat:
        """Gemma 3 prompt format"""
        return PromptFormat(
            system_prefix='',
            system_prompt='',
            system_suffix='',
            user_prefix='<start_of_turn>user\n',
            user_suffix='<end_of_turn>\n',
            bot_prefix='<start_of_turn>model\n',
            bot_suffix='<end_of_turn>\n'
        )
    
    # this is just ChatML, but we can't have "NoThinking" without "Thinking"
    def Qwen3Thinking(system_prompt: Optional[str] = None) -> PromptFormat:
        return PromptFormat(
            system_prefix='<|im_start|>system\n',
            system_prompt=system_prompt if system_prompt is not None else '',
            system_suffix='<|im_end|>\n',
            user_prefix='<|im_start|>user\n',
            user_suffix='<|im_end|>\n',
            bot_prefix='<|im_start|>assistant\n',
            bot_suffix='<|im_end|>\n'
        )
    
    def Qwen3NoThinking(system_prompt: Optional[str] = None) -> PromptFormat:
        return PromptFormat(
            system_prefix='<|im_start|>system\n/no_think\n',
            system_prompt=system_prompt if system_prompt is not None else '',
            system_suffix='<|im_end|>\n',
            user_prefix='<|im_start|>user\n',
            user_suffix='<|im_end|>\n',
            bot_prefix='<|im_start|>assistant\n<think>\n\n</think>\n\n',
            bot_suffix='<|im_end|>\n'
        )
    
    def GLM4(system_prompt: Optional[str] = None) -> PromptFormat:
        return PromptFormat(
            system_prefix="[gMASK]<sop><|system|>\n",
            system_prompt=system_prompt if system_prompt is not None else '',
            system_suffix="",
            user_prefix="<|user|>\n",
            user_suffix="",
            bot_prefix="<|assistant|>\n",
            bot_suffix=""
        )
