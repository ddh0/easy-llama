import sys
import tqdm
import numpy as np
from llama import Llama

Llama = Llama(
    path_model="/Users/dylan/Documents/AI/models/Meta-Llama-3.1-8B-Instruct-q8_0-q6_K.gguf",
    n_gpu_layers=-1,
    use_mmap=True,
    use_mlock=False,
    n_ctx=8192,
    offload_kqv=True,
    flash_attn=True
)

chktxt_a = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is Einstein famous for?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
chktxt_b = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is Einstein's full name?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

tokens_a = Llama.tokenize(chktxt_a.encode(), add_special=True, parse_special=True)
tokens_b = Llama.tokenize(chktxt_b.encode(), add_special=True, parse_special=True)

stream = Llama.stream(tokens_a, n_predict=32, stop_tokens=Llama.eog_tokens, sampler_params=None)

all_txt = ''

for tok in stream:
    txt = Llama.token_to_piece(tok, True).decode()
    all_txt += txt

logits = Llama.get_logits()
scores = Llama.get_scores()

# ---

Llama = Llama("/Users/dylan/Documents/AI/models/Meta-Llama-3.1-8B-Instruct-q8_0-q6_K.gguf", n_ctx=8192)
in_txt = "The apple doesn't fall far from"
in_toks = Llama.tokenize(in_txt.encode(), add_special=True, parse_special=False)
out_toks = Llama.generate(in_toks, n_predict=16)
out_txt = Llama.detokenize(out_toks, special=True)
print(out_txt)