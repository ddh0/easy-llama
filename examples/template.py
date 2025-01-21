# template.py
# Python 3.X.X

import easy_llama as ez

system_prompt = "You are an intelligent and helpful AI assistant."

prompt_format = ez.formats.PromptFormat(
    system_prefix='',
    system_prompt=system_prompt,
    system_suffix='',
    user_prefix='',
    user_suffix='',
    bot_prefix='',
    bot_suffix=''
)

Llama = ez.Llama(
    path_model='/path/to/your/model.gguf',
    n_gpu_layers=0,
    n_ctx=512,
    offload_kqv=False,
    flash_attn=False
)

sampler_preset = ez.sampling.SamplerPreset(
    top_k=40,
    top_p=0.95,
    min_p=0.05,
    temp=0.8,
    dry_multiplier=0.0,
    dry_base=1.75,
    dry_allowed_length=2,
    dry_penalty_last_n=-1
)

Thread = ez.Thread(Llama, prompt_format, sampler_preset)

WebUI = ez.WebUI(Thread)

WebUI.start(host='X.X.X.X', port=8080, ssl=True)

Thread.print_stats()
