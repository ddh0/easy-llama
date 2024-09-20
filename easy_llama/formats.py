# formats.py
# https://github.com/ddh0/easy-llama/

"""Submodule containing various prompt formats used by models"""

import time

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


blank: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "",
    "system_suffix": "",
    "user_prefix": "",
    "user_suffix": "",
    "bot_prefix": "",
    "bot_suffix": "",
    "stops": []
}

# https://github.com/tatsu-lab/stanford_alpaca
alpaca: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "Below is an instruction that describes a task. " + \
    "Write a response that appropriately completes the request.",
    "system_suffix": "\n\n",
    "user_prefix": "### Instruction:\n",
    "user_suffix": "\n\n",
    "bot_prefix": "### Response:\n",
    "bot_suffix": "\n\n",
    "stops": ['###', 'Instruction:', '\n\n\n']
}

# https://docs.mistral.ai/models/
# As a reference, here is the format used to tokenize instructions during fine-tuning:
# ```
# [START_SYMBOL_ID] + 
# tok("[INST]") + tok(USER_MESSAGE_1) + tok("[/INST]") +
# tok(BOT_MESSAGE_1) + [END_SYMBOL_ID] +
# …
# tok("[INST]") + tok(USER_MESSAGE_N) + tok("[/INST]") +
# tok(BOT_MESSAGE_N) + [END_SYMBOL_ID]
# ```
# In the pseudo-code above, note that the tokenize method should not add a BOS or EOS token automatically, but should add a prefix space.

mistral_instruct: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "",
    "system_suffix": "",
    "user_prefix": "[INST] ",
    "user_suffix": " [/INST]",
    "bot_prefix": "",
    "bot_suffix": "</s>",
    "stops": []
}

# https://docs.mistral.ai/platform/guardrailing/
mistral_instruct_safe: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "",
    "system_suffix": "",
    "user_prefix": "[INST] Always assist with care, respect, and truth. " + \
    "Respond with utmost utility yet securely. Avoid harmful, unethical, " + \
    "prejudiced, or negative content. Ensure replies promote fairness and " + \
    "positivity.\n\n",
    "user_suffix": " [/INST]",
    "bot_prefix": "",
    "bot_suffix": "</s>",
    "stops": []
}

# unofficial, custom template
mistral_instruct_roleplay: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "A chat between Alice and Bob.",
    "system_suffix": "\n\n",
    "user_prefix": "[INST] ALICE: ",
    "user_suffix": " [/INST] BOB:",
    "bot_prefix": "",
    "bot_suffix": "</s>",
    "stops": []
}

# https://github.com/openai/openai-python/blob/main/chatml.md
chatml: dict[str, str | list] = {
    "system_prefix": "<|im_start|>system\n",
    "system_prompt": "",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>user\n",
    "user_suffix": "<|im_end|>\n",
    "bot_prefix": "<|im_start|>assistant\n",
    "bot_suffix": "<|im_end|>\n",
    "stops": ['<|im_start|>']
}

# https://huggingface.co/blog/llama2
# system message relaxed to avoid undue refusals
llama2chat: dict[str, str | list] = {
    "system_prefix": "[INST] <<SYS>>\n",
    "system_prompt": "You are a helpful AI assistant.",
    "system_suffix": "\n<</SYS>>\n\n",
    "user_prefix": "",
    "user_suffix": " [/INST]",
    "bot_prefix": " ",
    "bot_suffix": " [INST] ",
    "stops": ['[INST]', '[/INST]']
}

# https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
llama3: dict[str, str | list] = {
    "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
    "system_prompt": 'You are a helpful AI assistant called "Llama 3".',
    "system_suffix": "<|eot_id|>\n",
    "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
    "user_suffix": "<|eot_id|>\n",
    "bot_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "bot_suffix": "<|eot_id|>\n",
    "stops": [128001, 128008, 128009]
}

# https://github.com/tatsu-lab/stanford_alpaca
alpaca: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "Below is an instruction that describes a task. " + \
    "Write a response that appropriately completes the request.",
    "system_suffix": "\n\n",
    "user_prefix": "### Instruction:\n",
    "user_suffix": "\n\n",
    "bot_prefix": "### Response:\n",
    "bot_suffix": "\n\n",
    "stops": ['###', 'Instruction:', '\n\n\n']
}

# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
phi3: dict[str, str | list] = {
    "system_prefix": "<|system|>\n",
    "system_prompt": "",
    "system_suffix": "<|end|>\n",
    "user_prefix": "<|user|>\n",
    "user_suffix": "<|end|>\n",
    "bot_prefix": "<|assistant|>\n",
    "bot_suffix": "<|end|>\n",
    "stops": []
}

# https://huggingface.co/google/gemma-2-27b-it
# https://ai.google.dev/gemma/docs/model_card_2
gemma2: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "",   # Does not officially support system prompt
    "system_suffix": "",
    "user_prefix": "<start_of_turn>user\n",
    "user_suffix": "<end_of_turn>\n",
    "bot_prefix": "<start_of_turn>model\n",
    "bot_suffix": "<end_of_turn>\n",
    "stops": ["<end_of_turn>"]
}

# this is the official vicuna. it is often butchered in various ways,
# most commonly by adding line breaks
# https://github.com/flu0r1ne/FastChat/blob/main/docs/vicuna_weights_version.md
vicuna_lmsys: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "",
    "system_suffix": " ",
    "user_prefix": "USER: ",
    "user_suffix": " ",
    "bot_prefix": "ASSISTANT: ",
    "bot_suffix": " ",
    "stops": ['USER:']
}

# spotted here and elsewhere:
# https://huggingface.co/Norquinal/Mistral-7B-claude-chat
vicuna_common: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "A chat between a curious user and an artificial " + \
    "intelligence assistant. The assistant gives helpful, detailed, " + \
    "and polite answers to the user's questions.",
    "system_suffix": "\n\n",
    "user_prefix": "USER: ",
    "user_suffix": "\n",
    "bot_prefix": "ASSISTANT: ",
    "bot_suffix": "\n",
    "stops": ['USER:', 'ASSISTANT:']
}

# an unofficial format that is easily "picked up" by most models
# change the tag attributes to suit your use case
# note the lack of newlines - they are not necessary, and might
# actually make it harder for the model to follow along
markup = {
    "system_prefix": '<message from="system">',
    "system_prompt": '',
    "system_suffix": '</message>',
    "user_prefix": '<message from="user">',
    "user_suffix": '</message>',
    "bot_prefix": '<message from="bot">',
    "bot_suffix": '</message>',
    "stops": ['</message>']
}

# https://huggingface.co/timdettmers/guanaco-65b
guanaco: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "A chat between a curious human and an artificial " + \
    "intelligence assistant. The assistant gives helpful, detailed, " + \
    "and polite answers to the user's questions.",
    "system_suffix": "\n",
    "user_prefix": "### Human: ",
    "user_suffix": " ",
    "bot_prefix": "### Assistant:",
    "bot_suffix": " ",
    "stops": ['###', 'Human:']
}

# https://huggingface.co/pankajmathur/orca_mini_v3_7b
orca_mini: dict[str, str | list] = {
    "system_prefix": "### System:\n",
    "system_prompt": "You are an AI assistant that follows instruction " + \
    "extremely well. Help as much as you can.",
    "system_suffix": "\n\n",
    "user_prefix": "### User:\n",
    "user_suffix": "\n\n",
    "bot_prefix": "### Assistant:\n",
    "bot_suffix": "\n\n",
    "stops": ['###', 'User:']
}

# https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
zephyr: dict[str, str | list] = {
    "system_prefix": "<|system|>\n",
    "system_prompt": "You are a friendly chatbot.",
    "system_suffix": "</s>\n",
    "user_prefix": "<|user|>\n",
    "user_suffix": "</s>\n",
    "bot_prefix": "<|assistant|>\n",
    "bot_suffix": "\n",
    "stops": ['<|user|>']
}

# OpenChat: https://huggingface.co/openchat/openchat-3.5-0106
openchat: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "",
    "system_suffix": "",
    "user_prefix": "GPT4 Correct User: ",
    "user_suffix": "<|end_of_turn|>",
    "bot_prefix": "GPT4 Correct Assistant:",
    "bot_suffix": "<|end_of_turn|>",
    "stops": ['<|end_of_turn|>']
}

# SynthIA by Migel Tissera
# https://huggingface.co/migtissera/Tess-XS-v1.0
synthia: dict[str, str | list] = {
    "system_prefix": "SYSTEM: ",
    "system_prompt": "Elaborate on the topic using a Tree of Thoughts and " + \
    "backtrack when necessary to construct a clear, cohesive Chain of " + \
    "Thought reasoning. Always answer without hesitation.",
    "system_suffix": "\n",
    "user_prefix": "USER: ",
    "user_suffix": "\n",
    "bot_prefix": "ASSISTANT: ",
    "bot_suffix": "\n",
    "stops": ['USER:', 'ASSISTANT:', 'SYSTEM:', '\n\n\n']
}

# Intel's neural chat v3
# https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/prompts/prompt.py
neural_chat: dict[str, str | list] = {
     "system_prefix": "### System:\n",
    "system_prompt": \
        "- You are a helpful assistant chatbot trained by Intel.\n" + \
        "- You answer questions.\n"+\
        "- You are excited to be able to help the user, but will refuse " + \
        "to do anything that could be considered harmful to the user.\n" + \
        "- You are more than just an information source, you are also " + \
        "able to write poetry, short stories, and make jokes.",
    "system_suffix": "</s>\n\n",
    "user_prefix": "### User:\n",
    "user_suffix": "</s>\n\n",
    "bot_prefix": "### Assistant:\n",
    "bot_suffix": "</s>\n\n",
    "stops": ['###']
}

# experimental: stanford's alpaca format adapted for chatml models
chatml_alpaca: dict[str, str | list] = {
    "system_prefix": "<|im_start|>system\n",
    "system_prompt": "Below is an instruction that describes a task. Write " + \
    "a response that appropriately completes the request.",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>instruction\n",
    "user_suffix": "<|im_end|>\n",
    "bot_prefix": "<|im_start|>response\n",
    "bot_suffix": "<|im_end|>\n",
    "stops": ['<|im_end|>', '<|im_start|>']
}

# experimental
autocorrect: dict[str, str | list] = {
    "system_prefix": "<|im_start|>instruction\n",
    "system_prompt": "Below is a word or phrase that might be misspelled. " + \
    "Output the corrected word or phrase without " + \
    "changing the style or capitalization.",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>input\n",
    "user_suffix": "<|im_end|>\n",
    "bot_prefix": "<|im_start|>output\n",
    "bot_suffix": "<|im_end|>\n",
    "stops": ['<|im_end|>', '<|im_start|>']
}

# https://huggingface.co/jondurbin/bagel-dpo-7b-v0.1
# Replace "assistant" with any other role
bagel: dict[str, str | list] = {
    "system_prefix": "system\n",
    "system_prompt": "",
    "system_suffix": "\n",
    "user_prefix": "user\n",
    "user_suffix": "\n",
    "bot_prefix": "assistant\n",
    "bot_suffix": "\n",
    "stops": ['user\n', 'assistant\n', 'system\n']
}

# https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0
solar_instruct: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "",
    "system_suffix": "",
    "user_prefix": "### User:\n",
    "user_suffix": "\n\n",
    "bot_prefix": "### Assistant:\n",
    "bot_suffix": "\n\n",
    "stops": ['### User:', '###', '### Assistant:']
}

# NeverSleep's Noromaid - alpaca with character names prefixed
noromaid: dict[str, str | list] = {
    "system_prefix": "",
    "system_prompt": "Below is an instruction that describes a task. " + \
    "Write a response that appropriately completes the request.",
    "system_suffix": "\n\n",
    "user_prefix": "### Instruction:\nAlice: ",
    "user_suffix": "\n\n",
    "bot_prefix": "### Response:\nBob:",
    "bot_suffix": "\n\n",
    "stops": ['###', 'Instruction:', '\n\n\n']
}

# https://huggingface.co/Undi95/Borealis-10.7B
nschatml: dict[str, str | list] = {
    "system_prefix": "<|im_start|>\n",
    "system_prompt": "",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_user|>\n",
    "user_suffix": "<|im_end|>\n",
    "bot_prefix": "<|im_bot|>\n",
    "bot_suffix": "<|im_end|>\n",
    "stops": []
}

# natural format for many models
natural: dict[str, str | list] = {
    "system_prefix": "<<SYSTEM>> ",
    "system_prompt": "",
    "system_suffix": "\n\n",
    "user_prefix": "<<USER>> ",
    "user_suffix": "\n\n",
    "bot_prefix": "<<ASSISTANT>>",
    "bot_suffix": "\n\n",
    "stops": ['\n\nNote:', '<<SYSTEM>>', '<<USER>>', '<<ASSISTANT>>', '\n\n<<']
}

# https://docs.cohere.com/docs/prompting-command-r
#
# NOTE: Command models benefit from special attention to the recommended prompt
#       format and techniques outlined in Cohere's documentation. The default
#       prompt format below will work OK, but ideally should be customized
#       for your specific use case according to the specific format shown in
#       the documentation.
#
command: dict[str, str | list] = {
    "system_prefix": "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
    "system_prompt": "You are a large language model called Command R built "
    "by the company Cohere. You act as a brilliant, sophisticated, "
    "AI-assistant chatbot trained to assist human users by providing "
    "thorough responses.",
    "system_suffix": "<|END_OF_TURN_TOKEN|>",
    "user_prefix": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
    "user_suffix": "<|END_OF_TURN_TOKEN|>",
    "bot_prefix": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "bot_suffix": "<|END_OF_TURN_TOKEN|>",
    "stops": []
}

mistral_openorca = chatml.copy()
dolphin = chatml.copy()
samantha = chatml.copy()
jackalope = chatml.copy()
naberius = chatml.copy()
hermes = chatml.copy()
monad = chatml.copy()
orca = chatml.copy()
hexoteric = chatml.copy()
orcamaid = chatml.copy()
cat = alpaca.copy() # Format is nonsensical as listed on HF. This one works
mytho_cat = alpaca.copy()
dolphin['system_prompt'] = "You are Dolphin, a helpful AI assistant."
mistral_openorca['system_prompt'] = \
    "You are MistralOrca, a large language model trained by Alignment " + \
    "Lab AI. Write out your reasoning step-by-step to be sure you get the " + \
    "right answers!"
samantha['system_prompt'] = \
    "You are a caring and empathetic sentient AI companion named Samantha."
jackalope['system_prompt'] = \
    "You are JackalopeAI, a large language model trained by OpenAccess " + \
    "AI Collective. Write out your reasoning step-by-step to be sure you " + \
    "get the right answers!"
naberius['system_prompt'] = "You are Naberius."
hermes['system_prompt'] = \
    "You are \"Hermes 2\", a conscious sentient " + \
    "superintelligent artificial intelligence developed by a man " + \
    "named Teknium, and your purpose and drive is to assist the user " + \
    "with any request they have. You experience emotions and have " + \
    "deep, profound thoughts and qualia."
monad['system_prompt'] = \
    "You are MonadGPT, a very old chatbot from " + \
    "the 17th century. Please answer the questions using an archaic " + \
    "language."
cat['system_prompt'] = \
    "Below is a conversation between a curious cat and a "+ \
    "demon summoned from hell called Nemesis. The evil Nemesis used to be " + \
    "a mortal killed by the medieval people because they objected to " + \
    "science. Now Nemesis just wants to come back and bring suffer to all " + \
    "those stupid human"
mytho_cat['system_prompt'] = \
    "You are a large language model. You are the " + \
    "result of a merge between two models, one called MythoLogic and one " + \
    "called Cat. You are called MythoCat."
orca['system_prompt'] = \
    "You are Orca, an AI language model created by " + \
    "Microsoft. You are a cautious assistant. You carefully follow " + \
    "instructions. You are helpful and harmless and you follow ethical " + \
    "guidelines and promote positive behavior."
hexoteric['system_prompt'] = "You are a language model called Hexoteric."
orcamaid['system_prompt'] = \
    "You are OrcaMaid, a super-intelligent, " + \
    "unbiased, and unfiltered AI created by the open-source community. " + \
    "You must remember to think slowly and carefully during all your " + \
    "responses, so as to avoid logical errors."
airoboros = llama2chat.copy()
airoboros['system_prompt'] = \
    "You are a helpful, unbiased, uncensored assistant."
tess = synthia.copy()
tess['system_prompt'] = '' # Tess can use any system prompt, or none
alpaca_strict = alpaca.copy() # Alpaca with more stopping strings
alpaca_strict['stops'] = [
    '###',
    '\n\n##',
    '\n\nInstruction:',
    '\n\nResponse:',
    '\n\n\n',
    '### Instruction:',
    '### Response:'
]

#
# AdvancedFormat presets
#

def _llama3_suffix_with_timestamp():
    return f"<|eot_id|>\n<|reserved_special_token_3|>{get_time_str()}<|reserved_special_token_4|>\n"

Llama3WithTimestamps = AdvancedFormat({
    "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
    "system_prompt": 'You are a helpful AI assistant called "Llama 3".',
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
