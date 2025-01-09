from .libllama import GGMLType
from .llama import Llama
from .thread import Thread
from .formats import PromptFormat
from .sampling import SamplerParams
from .webui import WebUI
MyLlama = Llama(
    path_model='/Users/dylan/Documents/AI/models/Meta-Llama-3.1-8B-Instruct-q8_0-q6_K.gguf',
    n_gpu_layers=-1,
    n_ctx=8192,
    type_k=GGMLType.GGML_TYPE_Q8_0,
    type_v=GGMLType.GGML_TYPE_Q8_0,
    offload_kqv=True,
    flash_attn=True,
    verbose=False
)
MySamplerParmas = SamplerParams(
    llama=MyLlama,
    temp=0.3,
    dry_multiplier=0.8
)
MyPromptFormat = PromptFormat(
    system_prefix='<|start_header_id|>system<|end_header_id|>\n\n',
    system_prompt='You are a helpful AI assistant called "Llama 3".',
    system_suffix='<|eot_id|>',
    user_prefix='<|start_header_id|>user<|end_header_id|>\n\n',
    user_suffix='<|eot_id|>',
    bot_prefix='<|start_header_id|>assistant<|end_header_id|>\n\n',
    bot_suffix='<|eot_id|>'
)
MyThread = Thread(
    llama=MyLlama,
    prompt_format=MyPromptFormat,
    sampler_params=MySamplerParmas
)
MyWebUI = WebUI(MyThread)
MyWebUI.start('127.0.0.1', 8080, ssl=True)
