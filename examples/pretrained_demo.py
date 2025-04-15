# pretrained_demo.py
# Python 3.12.9

import easy_llama as ez

Llama = ez.Llama(
    path_model='/path/to/your-pretrained-model.gguf',
    n_gpu_layers=-1,
    n_ctx=1024,
    offload_kqv=True,
    warmup=False,
    verbose=True
)

ez.set_verbose(False)
ez.utils.cls()

prompt = input(f'Enter your prompt:\n > ')

print(
    f'\n\n{ez.utils.ANSI.FG_BRIGHT_GREEN}{prompt}{ez.utils.ANSI.MODE_RESET_ALL}',
    end='',
    flush=True
)

prompt_bytes = prompt.encode('utf-8', errors='replace')

prompt_tokens = Llama.tokenize(prompt_bytes, add_special=True, parse_special=False)

token_generator = Llama.stream(
    input_tokens=prompt_tokens,
    n_predict=-1,
    stop_tokens=[]
)

for token in token_generator:
    tok_str = Llama.token_to_piece(token, special=True).decode('utf-8', errors='replace')
    print(
        f"{ez.utils.ANSI.FG_BRIGHT_CYAN}{tok_str}{ez.utils.ANSI.MODE_RESET_ALL}",
        sep='',
        end='',
        flush=True
    )

print(f'\n{"-" * 80}')
