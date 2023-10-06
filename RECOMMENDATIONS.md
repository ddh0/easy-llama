# Recommended models
The following are my subjective, personal choices for each model size.

In general, the more parameters a model has - i.e. as you get further down this list - the smarter it will be, the longer it will take to generate responses, and the more RAM you'll need.

If you're just starting out I'd recommend trying a 7B model first to get a feel for things.

These models are quantized and uploaded to HuggingFace courtesy of [Tom Jobbins](https://www.patreon.com/TheBlokeAI). Thank you!

> NOTES: These models are finetunes, not foundation models. This means they are good at things like assisting, instuct-response, and chat, but they are not pure text inference models. This list only includes Llama-2 models with the default 4096 context length for the sake of simplicity.

### 7B
[Airoboros L2 7B v3.0 Q6_K](https://huggingface.co/TheBloke/airoboros-l2-7B-3.0-GGUF/resolve/main/airoboros-l2-7b-3.0.Q6_K.gguf)

### 13B
[Airoboros L2 13B v3.0 Q6_K](https://huggingface.co/TheBloke/airoboros-l2-13B-3.0-GGUF/resolve/main/airoboros-l2-13b-3.0.Q6_K.gguf)

### 22B
[MythoMax L2 22B Q3_K_M](https://huggingface.co/TheBloke/L2-MythoMax22b-Instruct-Falseblock-GGUF/resolve/main/l2-mythomax22b-instruct-Falseblock.Q3_K_M.gguf) (faster, smaller, slightly dumber)

[MythoMax L2 22B Q6_K](https://huggingface.co/TheBloke/L2-MythoMax22b-Instruct-Falseblock-GGUF/resolve/main/l2-mythomax22b-instruct-Falseblock.Q6_K.gguf) (slower, larger, slightly smarter)
