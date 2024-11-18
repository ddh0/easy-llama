# formats.py
# https://github.com/ddh0/easy-llama/

"""Submodule containing various prompt formats used by models"""

import time
from datetime import datetime, timedelta

from typing import Callable, Any, Optional
from .utils import assert_type, NoneType


class AdvancedFormat:

    def __init__(self, _dict: dict[str, str | list | Callable]):
        assert_type(_dict, dict, '_dict', 'AdvancedFormat')
        _dict_keys = _dict.keys() # only read once
        if 'system_prompt' not in _dict_keys and 'system_content' in _dict_keys:
            raise ValueError(
                "AdvancedFormat: the provided dictionary uses the deprecated "
                "'system_content' key instead of the expected 'system_prompt' "
                "key. please update your code accordingly."
            )
        self._dict = _dict

    def __getitem__(self, key: str) -> Any:
        if key in self._dict.keys():
            if isinstance(self._dict[key], Callable):
                # return the result of the function as though it was a
                # value in the dictionary
                return self._dict[key]()
            else:
                return self._dict[key]
        else:
            raise KeyError(
                f"AdvancedFormat: the specified key {key!r} was not found"
            )

    def __repr__(self) -> str:
        return f'AdvancedFormat({self._dict!r})'

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def wrap(self, prompt: str) -> str:
        assert_type(prompt, str, 'prompt', 'AdvancedFormat.wrap')
        return self['system_prefix'] + \
               self['system_prompt'] + \
               self['system_suffix'] + \
               self['user_prefix']   + \
               prompt                + \
               self['user_suffix']   + \
               self['bot_prefix']

def wrap(
    prompt: str,
    format: dict[str, str | list] | AdvancedFormat
) -> str:
    """Wrap a given string in any prompt format for single-turn completion"""
    assert_type(prompt, str, 'prompt', 'formats.wrap')
    return format['system_prefix'] + \
           format['system_prompt'] + \
           format['system_suffix'] + \
           format['user_prefix']   + \
           prompt                  + \
           format['user_suffix']   + \
           format['bot_prefix']

def get_time_str() -> str:
    """Return a timestamp of the current time as a string"""
    # helpful: https://strftime.net
    return time.strftime("%l:%M %p, %A, %B %e, %Y")

def short_time_str() -> str:
    """Return a shorter timestamp of the current time as a string"""
    return time.strftime('%a %I:%M %p')

def iso_date_str() -> str:
    """Return a timestamp of the current date as a string in YYYY-MM-DD"""
    return time.strftime('%Y-%m-%d')

def _yesterday_iso_date_str():
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')

def blank() -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": "",
        "system_suffix": "",
        "user_prefix": "",
        "user_suffix": "",
        "bot_prefix": "",
        "bot_suffix": "",
        "stops": []
    }

def alpaca(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": system_prompt if system_prompt is not None else "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        "system_suffix": "\n\n",
        "user_prefix": "### Instruction:\n",
        "user_suffix": "\n\n",
        "bot_prefix": "### Response:\n",
        "bot_suffix": "\n\n",
        "stops": ['###', 'Instruction:', '\n\n\n']
    }

def mistral_instruct() -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": "",
        "system_suffix": "",
        "user_prefix": "[INST] ",
        "user_suffix": " ",
        "bot_prefix": "[/INST]",
        "bot_suffix": "</s>",
        "stops": []
    }

def mistral_instruct_v7(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "[SYSTEM_PROMPT] ",
        "system_prompt": system_prompt if system_prompt is not None else f"""You are Mistral, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is {iso_date_str()}.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {_yesterday_iso_date_str()}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You do not have any multimodal capability, in particular you cannot read nor generate images, or transcribe audio files or videos.""",
        "system_suffix": "[/SYSTEM_PROMPT]",
        "user_prefix": "[INST] ",
        "user_suffix": " ",
        "bot_prefix": "[/INST]",
        "bot_suffix": "</s>",
        "stops": []
    }

def mistral_instruct_safe() -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": "",
        "system_suffix": "",
        "user_prefix": "[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.\n\n",
        "user_suffix": " ",
        "bot_prefix": "[/INST]",
        "bot_suffix": "</s>",
        "stops": []
    }

def mistral_instruct_roleplay(user_name: str = 'Alice', bot_name: str = 'Bob') -> dict[str, str | list]:
    return {
        "system_prefix": "[INST] ",
        "system_prompt": f"A chat between {user_name} and {bot_name}.",
        "system_suffix": " [/INST]</s>\n\n",
        "user_prefix": f"[INST] {user_name.upper()}: ",
        "user_suffix": " ",
        "bot_prefix": f"[/INST] {bot_name.upper()}:",
        "bot_suffix": "</s>",
        "stops": ['\n', '\n\n', '\n\n\n']
    }

def chatml(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": ['<|im_end|>']
    }

def llama2chat(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "[INST] <<SYS>>\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are a helpful AI assistant.",
        "system_suffix": "\n<</SYS>>\n\n",
        "user_prefix": "",
        "user_suffix": " [/INST]",
        "bot_prefix": " ",
        "bot_suffix": " [INST] ",
        "stops": ['[INST]', '[/INST]']
    }

def llama3(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_prompt": system_prompt if system_prompt is not None else 'You are a helpful AI assistant called "Llama 3".',
        "system_suffix": "<|eot_id|>\n",
        "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_suffix": "<|eot_id|>\n",
        "bot_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "bot_suffix": "<|eot_id|>\n",
        "stops": [128001, 128008, 128009]
    }

def phi3(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|system|>\n",
        "system_prompt": system_prompt if system_prompt is not None else "",
        "system_suffix": "<|end|>\n",
        "user_prefix": "<|user|>\n",
        "user_suffix": "<|end|>\n",
        "bot_prefix": "<|assistant|>\n",
        "bot_suffix": "<|end|>\n",
        "stops": []
    }

def gemma2() -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": "",
        "system_suffix": "",
        "user_prefix": "<start_of_turn>user\n",
        "user_suffix": "<end_of_turn>\n",
        "bot_prefix": "<start_of_turn>model\n",
        "bot_suffix": "<end_of_turn>\n",
        "stops": ["<end_of_turn>"]
    }

def vicuna_lmsys() -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": "",
        "system_suffix": "",
        "user_prefix": "USER: ",
        "user_suffix": " ",
        "bot_prefix": "ASSISTANT: ",
        "bot_suffix": " ",
        "stops": ['USER:']
    }

def vicuna_common(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": system_prompt if system_prompt is not None else "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        "system_suffix": "\n\n",
        "user_prefix": "USER: ",
        "user_suffix": "\n",
        "bot_prefix": "ASSISTANT: ",
        "bot_suffix": "\n",
        "stops": ['USER:', 'ASSISTANT:']
    }

def markup(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": '<message from="system">',
        "system_prompt": system_prompt if system_prompt is not None else '',
        "system_suffix": '</message>',
        "user_prefix": '<message from="user">',
        "user_suffix": '</message>',
        "bot_prefix": '<message from="bot">',
        "bot_suffix": '</message>',
        "stops": ['</message>']
    }

def pygmalion(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|system|>",
        "system_prompt": system_prompt if system_prompt is not None else """Enter RP mode. Pretend to be {{char}} whose persona follows:
{{persona}}

You shall reply to the user while staying in character, and generate long responses.""",
        "system_suffix": "",
        "user_prefix": "<|user|>",
        "user_suffix": "",
        "bot_prefix": "<|model|>",
        "bot_suffix": "</s>",
        "stops": ["<|",]
    }

def guanaco(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": system_prompt if system_prompt is not None else "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        "system_suffix": "\n",
        "user_prefix": "### Human: ",
        "user_suffix": " ",
        "bot_prefix": "### Assistant:",
        "bot_suffix": " ",
        "stops": ['###', 'Human:']
    }

def orca_mini(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "### System:\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
        "system_suffix": "\n\n",
        "user_prefix": "### User:\n",
        "user_suffix": "\n\n",
        "bot_prefix": "### Assistant:\n",
        "bot_suffix": "\n\n",
        "stops": ['###', 'User:']
    }

def zephyr(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|system|>\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are a friendly chatbot.",
        "system_suffix": "</s>\n",
        "user_prefix": "<|user|>\n",
        "user_suffix": "</s>\n",
        "bot_prefix": "<|assistant|>\n",
        "bot_suffix": "\n",
        "stops": ['<|user|>']
    }

def openchat() -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": "",
        "system_suffix": "",
        "user_prefix": "GPT4 Correct User: ",
        "user_suffix": "<|end_of_turn|>",
        "bot_prefix": "GPT4 Correct Assistant:",
        "bot_suffix": "<|end_of_turn|>",
        "stops": ['<|end_of_turn|>']
    }

def synthia(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "SYSTEM: ",
        "system_prompt": system_prompt if system_prompt is not None else "Elaborate on the topic using a Tree of Thoughts and backtrack when necessary to construct a clear, cohesive Chain of Thought reasoning. Always answer without hesitation.",
        "system_suffix": "\n",
        "user_prefix": "USER: ",
        "user_suffix": "\n",
        "bot_prefix": "ASSISTANT: ",
        "bot_suffix": "\n",
        "stops": ['USER:', 'ASSISTANT:', 'SYSTEM:', '\n\n\n']
    }

def neural_chat(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "### System:\n",
        "system_prompt": system_prompt if system_prompt is not None else "- You are a helpful assistant chatbot trained by Intel.\n- You answer questions.\n- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.",
        "system_suffix": "</s>\n\n",
        "user_prefix": "### User:\n",
        "user_suffix": "</s>\n\n",
        "bot_prefix": "### Assistant:\n",
        "bot_suffix": "</s>\n\n",
        "stops": ['###']
    }

def chatml_alpaca(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>instruction\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>response\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": ['<|im_end|>']
    }

def autocorrect(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>instruction\n",
        "system_prompt": system_prompt if system_prompt is not None else "Below is a word or phrase that might be misspelled. Output the corrected word or phrase without changing the style or capitalization.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>input\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>output\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": ['<|im_end|>']
    }

def bagel(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "system\n",
        "system_prompt": system_prompt if system_prompt is not None else "",
        "system_suffix": "\n",
        "user_prefix": "user\n",
        "user_suffix": "\n",
        "bot_prefix": "assistant\n",
        "bot_suffix": "\n",
        "stops": ['user\n', 'assistant\n', 'system\n']
    }

def solar_instruct(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "### System:\n",
        "system_prompt": system_prompt if system_prompt is not None else "",
        "system_suffix": "",
        "user_prefix": "### User:\n",
        "user_suffix": "\n\n",
        "bot_prefix": "### Assistant:\n",
        "bot_suffix": "\n\n",
        "stops": ['### User:', '###', '### Assistant:']
    }

def noromaid(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": system_prompt if system_prompt is not None else "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        "system_suffix": "\n\n",
        "user_prefix": "### Instruction:\nAlice: ",
        "user_suffix": "\n\n",
        "bot_prefix": "### Response:\nBob:",
        "bot_suffix": "\n\n",
        "stops": ['###', 'Instruction:', '\n\n\n']
    }

def nschatml(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>\n",
        "system_prompt": system_prompt if system_prompt is not None else "",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_user|>\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_bot|>\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def command(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
        "system_prompt": system_prompt if system_prompt is not None else "You are a large language model called Command R built by the company Cohere. You act as a brilliant, sophisticated, AI-assistant chatbot trained to assist human users by providing thorough responses.",
        "system_suffix": "<|END_OF_TURN_TOKEN|>",
        "user_prefix": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        "user_suffix": "<|END_OF_TURN_TOKEN|>",
        "bot_prefix": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "bot_suffix": "<|END_OF_TURN_TOKEN|>",
        "stops": []
    }

def aya() -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": "",
        "system_suffix": "",
        "user_prefix": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        "user_suffix": "<|END_OF_TURN_TOKEN|>",
        "bot_prefix": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "bot_suffix": "<|END_OF_TURN_TOKEN|>",
        "stops": []
    }

def mistral_openorca(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def dolphin(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are Dolphin, a helpful AI assistant.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def samantha(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are a caring and empathetic sentient AI companion named Samantha.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def jackalope(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are JackalopeAI, a large language model trained by OpenAccess AI Collective. Write out your reasoning step-by-step to be sure you get the right answers!",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def naberius(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are Naberius.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def hermes(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else 'You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.',
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def monad(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are MonadGPT, a very old chatbot from the 17th century. Please answer the questions using an archaic language.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def orca(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def hexoteric(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are a language model called Hexoteric.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def orcamaid(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are OrcaMaid, a super-intelligent, unbiased, and unfiltered AI created by the open-source community. You must remember to think slowly and carefully during all your responses, so as to avoid logical errors.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def qwen(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "<|im_start|>system\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "bot_prefix": "<|im_start|>assistant\n",
        "bot_suffix": "<|im_end|>\n",
        "stops": []
    }

def cat(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": system_prompt if system_prompt is not None else "Below is a conversation between a curious cat and a demon summoned from hell called Nemesis. The evil Nemesis used to be a mortal killed by the medieval people because they objected to science. Now Nemesis just wants to come back and bring suffer to all those stupid human",
        "system_suffix": "\n\n",
        "user_prefix": "### Instruction:\n",
        "user_suffix": "\n\n",
        "bot_prefix": "### Response:\n",
        "bot_suffix": "\n\n",
        "stops": ['###', 'Instruction:', '\n\n\n']
    }

def mytho_cat(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": system_prompt if system_prompt is not None else "You are a large language model. You are the result of a merge between two models, one called MythoLogic and one called Cat. You are called MythoCat.",
        "system_suffix": "\n\n",
        "user_prefix": "### Instruction:\n",
        "user_suffix": "\n\n",
        "bot_prefix": "### Response:\n",
        "bot_suffix": "\n\n",
        "stops": ['###', 'Instruction:', '\n\n\n']
    }

def airoboros(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "[INST] <<SYS>>\n",
        "system_prompt": system_prompt if system_prompt is not None else "You are a helpful, unbiased, uncensored assistant.",
        "system_suffix": "\n<</SYS>>\n\n",
        "user_prefix": "",
        "user_suffix": " [/INST]",
        "bot_prefix": " ",
        "bot_suffix": " [INST] ",
        "stops": ['[INST]', '[/INST]']
    }

def tess(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "SYSTEM: ",
        "system_prompt": system_prompt if system_prompt is not None else '',
        "system_suffix": "\n",
        "user_prefix": "USER: ",
        "user_suffix": "\n",
        "bot_prefix": "ASSISTANT: ",
        "bot_suffix": "\n",
        "stops": ['USER:', 'ASSISTANT:', 'SYSTEM:', '\n\n\n']
    }

def alpaca_strict(system_prompt: Optional[str] = None) -> dict[str, str | list]:
    return {
        "system_prefix": "",
        "system_prompt": system_prompt if system_prompt is not None else "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        "system_suffix": "\n\n",
        "user_prefix": "### Instruction:\n",
        "user_suffix": "\n\n",
        "bot_prefix": "### Response:\n",
        "bot_suffix": "\n\n",
        "stops": ['###', '\n\n##', '\n\nInstruction:', '\n\nResponse:', '\n\n\n', '### Instruction:', '### Response:']
    }

#
# AdvancedFormat presets
#

def _llama3_suffix_with_timestamp():
    return f"<|eot_id|>\n<|reserved_special_token_3|>{get_time_str()}<|reserved_special_token_4|>\n"

def Llama3WithTimestamps(system_prompt: Optional[str] = None) -> AdvancedFormat:
    return AdvancedFormat({
        "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_prompt": system_prompt if system_prompt is not None else 'You are a helpful AI assistant called "Llama 3".',
        "system_suffix": _llama3_suffix_with_timestamp,
        "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_suffix": _llama3_suffix_with_timestamp,
        "bot_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "bot_suffix": _llama3_suffix_with_timestamp,
        "stops": [128001, 128008, 128009, 128011, 128012]
    })

def AdvancedChatMarkupFormat(
    user_name: str,
    bot_name: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None
) -> AdvancedFormat:
    """
    Quickly create a prompt template using the specified variables, for use
    within Threads
    """

    assert_type(user_name, str, 'user_name', 'AdvancedChatMarkupFormat')
    assert_type(bot_name, str, 'bot_name', 'AdvancedChatMarkupFormat')
    assert_type(title, (str, NoneType), 'title', 'AdvancedChatMarkupFormat')
    assert_type(description, (str, NoneType), 'description',
                'AdvancedChatMarkupFormat')
    assert_type(tags, (list, NoneType), 'tags', 'AdvancedChatMarkupFormat')

    _t = "    " # indentation string

    def _user_prefix() -> str:
        return (f'{_t*2}<message sender="{user_name}" '
                f'timestamp="{short_time_str()}">\n{_t*3}<text>')

    def _bot_prefix() -> str:
        return (f'{_t*2}<message sender="{bot_name}" '
                f'timestamp="{short_time_str()}">\n{_t*3}<text>')

    def _msg_suffix() -> str:
        return f"</text>\n{_t*2}</message>\n"

    if tags is not None:
        xml_tags = [f'{_t*2}<tags>']
        for tag in tags:
            xml_tags.append(f'{_t*3}<tag>{tag}</tag>')
        xml_tags.append(f'{_t*2}</tags>')
        final_tags_string = '\n'.join(xml_tags)
    else:
        final_tags_string = f"{_t*2}<tags>\n{_t*2}</tags>"

    return AdvancedFormat(
        {
            "system_prefix": "",
            "system_prompt": f"""<chat>\n{_t}<meta>\n{_t*2}<title>{title if title is not None else "Untitled Chat"}</title>\n{_t*2}<description>{description if description is not None else "No description provided"}</description>\n{final_tags_string}\n{_t*2}<participants>\n{_t*3}<participant name="{user_name}"/>\n{_t*3}<participant name="{bot_name}"/>\n{_t*2}</participants>\n{_t*2}<datetime>{get_time_str()}</datetime>\n{_t}</meta>\n{_t}<messages>""",
            "system_suffix": "\n",
            "user_prefix": _user_prefix,
            "user_suffix": _msg_suffix,
            "bot_prefix": _bot_prefix,
            "bot_suffix": _msg_suffix,
            "stops": ["</", "</text>", "</message>"]
        }
    )
