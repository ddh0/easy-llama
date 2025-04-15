# instruct_demo.py
# Python 3.12.9

import easy_llama as ez

Llama = ez.Llama(
    path_model='/path/to/your-llama3-model.gguf',
    n_gpu_layers=-1,
    n_ctx=1024,
    offload_kqv=True,
    warmup=False,
    verbose=True
)

Thread = ez.Thread(
    llama=Llama,
    prompt_format=ez.PromptFormats.Llama3("You are a helpful assistant.")
)

ez.set_verbose(False)
ez.utils.cls()
Thread.interact(stream=True)
print('-' * 80)
Thread.print_stats()
