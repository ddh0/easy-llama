# Recommended models for CPU-based text inference
The following are my subjective, personal choices for each model size.

In general, the more parameters a model has - i.e. as you get further down this list - the smarter it will be, the longer it will take to generate responses, and the more RAM you'll need.

If you're just starting out I'd recommend trying a 7B model first to get a feel for things.

These models are quantized and uploaded to HuggingFace courtesy of [Tom Jobbins](https://www.patreon.com/TheBlokeAI). Thank you!

> NOTES: These models are finetunes, not foundation models. This means they are good at things like assisting, instuct-response, and chat, but they are not pure text inference models. Also, this list only includes LLaMA-1 models with the default 2048 context length for the sake of simplicity.

### 3B
[Orca Mini 3B q5_1](https://huggingface.co/TheBloke/orca_mini_3B-GGML/blob/main/orca-mini-3b.ggmlv3.q5_1.bin)

### 7B
[Orca Mini 7B v2 q5_K_M](https://huggingface.co/TheBloke/orca_mini_v2_7B-GGML/blob/main/orca-mini-v2_7b.ggmlv3.q5_K_M.bin)

### 13B
[Guanaco 13B q5_K_M](https://huggingface.co/TheBloke/guanaco-13B-GGML/blob/main/guanaco-13B.ggmlv3.q5_K_M.bin)

### 33B
[Guanaco 33B q4_K_S](https://huggingface.co/TheBloke/guanaco-33B-GGML/blob/main/guanaco-33B.ggmlv3.q4_K_S.bin)

---
> Models above this size are unlikely to run at usable speeds on CPU, and so are not listed.
