# _internals.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import ctypes

from typing import Iterable
from libllama import (
    llama_sampler_init_greedy,
    llama_kv_cache_seq_rm,
    llama_sampler_sample,
    llama_token_to_piece,
    llama_batch_init,
    llama_batch_free,
    llama_tokenize,
    llama_context,
    llama_sampler,
    llama_decode,
    llama_model,
    NULL,
    ptr
)

MAX_TOKEN_LENGTH = 256
"""The maximum supported length of a single token's text, in bytes"""

def decode_pp(
    ctx: ptr[llama_context],
    pos: int,
    tokens: list[int],
    n_tokens: int
) -> None:
    """
    ### INTERNAL

    Decode with batch size > 1 (prompt processing)
    """
    batch = llama_batch_init(n_tokens=n_tokens, embd=0, n_seq_max=1)
    batch.n_tokens = n_tokens
    for i in range(n_tokens):
        batch.token[i] = tokens[i]
        batch.pos[i] = pos + i
        batch.seq_id[i][0] = 0
        batch.n_seq_id[i] = 1
        batch.logits[i] = False
    batch.logits[n_tokens - 1] = True
    ret = llama_decode(ctx, batch)
    llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(
            f'decode_pp: llama_decode failed with status code {ret}'
        )

def decode_tg(
    ctx: ptr[llama_context],
    pos: int,
    token: int
) -> None:
    """
    ### INTERNAL

    Decode with batch size == 1 (text generation)
    """
    batch = llama_batch_init(n_tokens=1, embd=0, n_seq_max=1)
    batch.n_tokens = 1
    batch.token[0] = token
    batch.pos[0] = pos
    batch.seq_id[0][0] = 0
    batch.n_seq_id[0] = 1
    batch.logits[0] = True
    ret = llama_decode(ctx, batch)
    llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(
            f'decode_tg: llama_decode failed with status code {ret}'
        )

greedy_sampler = llama_sampler_init_greedy()

def sample_greedy(ctx: ptr[llama_context]) -> int:
    """
    ### INTERNAL

    Sample the most likely token
    """
    return llama_sampler_sample(greedy_sampler, ctx, -1)

def tokenize(
    model: ptr[llama_model],
    text_bytes: bytes,
    n_tokens_max: int,
    add_special: bool,
    parse_special: bool,
) -> list[int]:
    """
    ### INTERNAL

    Convert the provided UTF-8 encoded text into tokens

    - text_bytes:
        The text to be tokenized
    - n_tokens_max:
        Tokenization will fail if the text is more than this many tokens.
        Larger numbers allow more text to be tokenized but will allocate
        more memory (4 bytes per token).
    - add_special:
        Allow to add BOS and EOS tokens if model is configured to do so.
    - parse_special:
        Allow tokenizing special and/or control tokens which otherwise are
        not exposed and treated as plaintext. Does not insert a leading
        space.
    """
    # unlike detokenization, this buffer is created and destroyed as needed
    # because it could potentially be quite large - each token takes 4 bytes
    tokens_buf = (ctypes.c_int32 * n_tokens_max)()
    n_tokens = llama_tokenize(
        model=model,
        text=text_bytes,
        text_len=len(text_bytes),
        tokens=tokens_buf,
        n_tokens_max=n_tokens_max,
        add_special=add_special,
        parse_special=parse_special
    )
    if n_tokens < 0:
        raise ValueError(
            f'tokenize: n_tokens value {-n_tokens} exceeds '
            f'n_tokens_max value {n_tokens_max}'
        )
    ret = list(tokens_buf[:n_tokens])
    del tokens_buf
    return ret

# this buffer is re-used every time llama_token_to_piece() is called
# it is only 256 bytes, so OK to keep in memory
detok_buffer = ctypes.create_string_buffer(MAX_TOKEN_LENGTH)

def token_to_piece(
    model: ptr[llama_model], token: int, special: bool
) -> bytes:
    """
    ### INTERNAL

    Convert token ID to text bytes
    """
    n_bytes = llama_token_to_piece(
        model=model,
        token=token,
        buf=detok_buffer,
        length=MAX_TOKEN_LENGTH,
        lstrip=0, # skip up to 'lstrip' leading spaces
        special=special
    )
    if n_bytes > MAX_TOKEN_LENGTH:
        raise ValueError(
            f"token_to_piece: the token with ID {token} requires a "
            f"buffer of size {n_bytes}, but the maximum buffer size is "
            f"{MAX_TOKEN_LENGTH}"
        )
    # NOTE: do not just do buf.value.decode() because the token could
    #       possibly be a part of a utf-8 bytestring, but not a valid utf-8
    #       string itself. let the caller handle this
    return detok_buffer.raw[:n_bytes]

def detokenize(
    model: ptr[llama_model],
    tokens: Iterable[int],
    special: bool
) -> bytes:
    """
    ### INTERNAL

    Convert the provided tokens into UTF-8 encoded text

    - special:
        If True, special tokens are rendered in the output
    """
    # this function is just like token_to_piece but in a loop
    detok_bytes = b""
    for token in tokens:
        n_bytes = llama_token_to_piece(
            model=model,
            token=token,
            buf=detok_buffer,
            length=MAX_TOKEN_LENGTH,
            lstrip=0, # skip up to 'lstrip' leading spaces
            special=special
        )
        if n_bytes > MAX_TOKEN_LENGTH:
            raise ValueError(
                f"detokenize: the token with ID {token} "
                f"requires a buffer of size {n_bytes}, but the maximum "
                f"buffer size is {MAX_TOKEN_LENGTH}"
            )
        detok_bytes += detok_buffer.raw[:n_bytes]
    return detok_bytes

def get_length(
    model: ptr[llama_model],
    text_bytes: bytes,
    add_special: bool,
    parse_special: bool,
) -> int:
    """
    ### INTERNAL

    Return the length of a given text, as measured in tokens
    """
    return -llama_tokenize(
        model=model,
        text=text_bytes,
        text_len=len(text_bytes),
        tokens=NULL,
        n_tokens_max=0,
        add_special=add_special,
        parse_special=parse_special
    )

def eval_single(
    ctx: ptr[llama_context],
    input_tokens: Iterable[int],
    n_batch: int,
    sampler: llama_sampler
) -> int:
    """
    ### INTERNAL

    Predict a single token
    """

    pos = 0

    llama_kv_cache_seq_rm(ctx, 0, pos, -1)

    while True:

        batch_tokens = input_tokens[pos:pos + n_batch]
        n_batch_tokens = len(batch_tokens)

        if n_batch_tokens == 0:
            raise RuntimeError(
                f'eval_single: n_batch_tokens == 0; this should not happen'
            )
        if n_batch_tokens == 1:
            decode_tg(ctx, pos, batch_tokens[0])
            return llama_sampler_sample(sampler, ctx, -1)
        if n_batch_tokens > 1:
            decode_pp(ctx, pos, batch_tokens, n_batch_tokens)
            pos += n_batch_tokens

def eval_loop(
    ctx: ptr[llama_context],
    input_tokens: Iterable[int],
    n_predict: int, # if >= 0, unlimited
    n_batch: int,
    stop_tokens: Iterable[int],
    sampler: llama_sampler
) -> list[int]:
    """
    ### INTERNAL

    Predict multiple tokens
    """
    
    output_tokens = []
    all_tokens = input_tokens

    n_predicted = 0
    pos = 0

    llama_kv_cache_seq_rm(ctx, 0, pos, -1)

    while True:

        if n_predict >= 0 and n_predicted >= n_predict:
            return output_tokens

        batch_tokens = all_tokens[pos:pos + n_batch]
        n_batch_tokens = len(batch_tokens)

        if n_batch_tokens == 0:
            raise RuntimeError(
                f'eval_loop: n_batch_tokens == 0; this should not happen'
            )
        if n_batch_tokens == 1:
            decode_tg(ctx, pos, batch_tokens[0])
            pos += 1
        if n_batch_tokens > 1:
            decode_pp(ctx, pos, batch_tokens, n_batch_tokens)
            pos += n_batch_tokens
        
        id = llama_sampler_sample(sampler, ctx, -1)
        output_tokens.append(id)
        all_tokens.append(id)
        n_predicted += 1

        if id in stop_tokens:
            return output_tokens