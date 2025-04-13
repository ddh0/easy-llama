# my_personal_chatbot_server.py

import easy_llama as ez

# specify a GGUF model file and optional parameters
Llama = ez.Llama(
    path_model='/path/to/your/model.gguf',
    n_gpu_layers=-1,
    n_ctx=8192,
    offload_kqv=True,
    flash_attn=True
)

# choose the right prompt format for your model
prompt_format = ez.PromptFormats.ChatML("You are an intelligent and helpful AI assistant.")

# customize sampling
sampler_preset = ez.SamplerPreset(
    top_k=-1,
    top_p=0.9,
    min_p=0.0,
    temp=0.6
)

# create the actual Thread and Server instances
Thread = ez.Thread(Llama, prompt_format, sampler_preset)
Server = ez.Server(Thread, host='127.0.0.1', port=8080)

# start the server
Server.run()

# once the server is stopped manually, print the thread context usage stats
Thread.print_stats()
