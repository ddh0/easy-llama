# template.py
# Python 3.X.X

import easy_llama as ez

Llama = ez.Llama(
    path_model='/path/to/your/model.gguf',
    n_gpu_layers=0,
    n_ctx=512,
    offload_kqv=False,
    flash_attn=False
)

system_prompt = "You are an intelligent and helpful AI assistant."

prompt_format = ez.PromptFormats.ChatML(system_prompt)

sampler_preset = ez.SamplerPreset(
    top_k=-1,
    top_p=0.9,
    min_p=0.01,
    temp=0.8,
    dry_multiplier=0.0,
    dry_base=1.25,
    dry_allowed_length=4,
    dry_penalty_last_n=-1
)

Thread = ez.Thread(Llama, prompt_format, sampler_preset)

#Thread.interact(stream=True)

ez.WebUI(Thread).start(host='X.X.X.X', port=8080, ssl=True)

ids = Thread.get_input_ids()
ez.utils.cls()
print('-' * 80)
print(ids)
print('-' * 80)
print(repr(Thread))
print('-' * 80)
Thread.print_stats()
