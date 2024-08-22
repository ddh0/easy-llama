# Documentation for easy-llama

> [!TIP]
>
> Click the icon in the top right of this box to expand the outline, which lets you jump to a particular item. 

> [!NOTE]
>
> The documentation assumes you have imported easy-llama as `ez`:
> 
> ```python
import easy_llama as ez
> ```

# `class ez.Model`

A high-level abstraction of a Llama model

The following attributes are available:
- verbose: Whether the model was loaded with `verbose=True`
- metadata: A dictionary containing the GGUF metadata of the model
- context_length: The currently loaded context length of the model, in tokens
- n_ctx: Alias to context_length
- llama: The underlying `llama_cpp.Llama` instance
- vocab: A list of all tokens in the model's vocabulary
- bos_token: The beginning-of-sequence token ID
- eos_token: The end-of-sequence token ID
- eot_token: The end-of-turn token ID (or `None` if not found)
- nl_token: The newline token ID (or `None` if not found)
- prefix_token: The infill prefix token ID (or `None` if not found)
- middle_token: The infill middle token ID (or `None` if not found)
- suffix_token: The infill suffix token ID (or `None` if not found)
- cls_token: The classifier token ID (or `None` if not found)
- sep_token: The separator token ID (or `None` if not found)
- filename: The name of the file the model was loaded from
- n_ctx_train: The native context length of the model
- rope_freq_base_train: The native RoPE frequency base (theta) value
- rope_freq_base: The currently loaded RoPE frequency base (theta) value
- flash_attn: Whether the model was loaded with Flash Attention enabled
- n_vocab: The number of tokens in the model's vocabulary
- n_layer: The number of layers in the model
- n_gpu_layers: The number of layers offloaded to the GPU (-1 for all layers)
- ctx_scale: The ratio of `context_length`/`n_ctx_train`
- type_k: The GGML data type used for the `K` cache. 1 == f16, q8_0 otherwise
- type_v: The GGML data type used for the `V` cache. 1 == f16, q8_0 otherwise
- n_gqa: The GQA (Grouped-Query Attention) factor of the model

## `Model.__init__()`

Given the path to a GGUF file, construct a Model instance.

The following parameter is required:
- `model_path: str` - The path to the GGUF model you wish to load

The following parameters are optional:
- `context_length: int` - The context length at which to load the model, in tokens. May be less than or greater than the native context length of the model. A warning will be displayed if the chosen context length is large enough to cause a loss of quality. Modifies `rope_freq_base` for context scaling, which does not degrade quality as much as linear RoPE scaling. Defaults to `None`, which will use the native context length of the model.
- `n_gpu_layers: int` - The number of layers to offload to the GPU. Defaults to `0`.
- `offload_kqv: bool` - Whether to offload the K, Q, and V caches (i.e. context) to the GPU. Defaults to `True`.
- `flash_attn: bool` - Whether to use Flash Attention ([ref](https://github.com/ggerganov/llama.cpp/pull/5021)). Defaults to `False`.
- `verbose: bool` - Whether to show output from `llama-cpp-python`/`llama.cpp` or not. This is lots of very detailed output.

### Example usage
```python
# load a model from a GGUF file
Mistral = ez.Model(
	'mistral-7b-instruct-v0.1.Q4_K_S.gguf',
	n_gpu_layers=-1,
	flash_attn=True
)
```


##  `Model.unload() -> None`

Unload the Model from memory. If the Model is already unloaded, do nothing.

If you attempt to use a Model after it has been unloaded, `easy_llama.model.ModelUnloadedException` will be raised.

### Example usage
```python
# load a model (allocates memory)
Mistral = ez.Model('mistral-7b-instruct-v0.1.Q4_K_S.gguf')

# returns ' the sun is shining, and blah blah blah...'
Mistral.generate('The sky is blue, and')

# unload model (frees memory)
Mistral.unload()

# raises ez.model.ModelUnloadedException
Mistral.generate('The girl walked down')
```

## `Model.get_length() -> int`

Get the length of a given text in tokens according to this Model, including the appended BOS token.

The following parameter is required:
- `text: str` - The text to read

### Example usage
```python
>>> Mistral.get_length('Gentlemen, owing to lack of time and adverse circumstances, most people leave this world without thinking too much about it. Those who try get a headache and move on to something else. I belong to the second group. As my career progressed, the amount of space dedicated to me in Who’s Who grew and grew, but neither the last issue nor any future ones will explain why I abandoned journalism. This will be the subject of my story, which I wouldn’t tell you under other circumstances anyway.')
109
```

## `Model.generate() -> str`

Given a prompt, return a generated string.

The following parameter is required:
- `prompt: str` - The text from which to generate

The following parameters are optional:
- `stops: list[Union[str, int]]` - A list of strings and/or token IDs at which to end the generation early. Defaults to `[]`.
- `sampler: SamplerSettings` - The `ez.samplers.SamplerSettings` object used to control text generation. Defaults to `ez.samplers.DefaultSampling`.

## `Model.stream() -> Generator`

Given a prompt, return a Generator that yields dicts containing tokens. To get the token string itself, subscript the dict with: `['choices'][0]['text']`.

The following parameter is required:
- `prompt: str` - The text from which to generate

The following parameters are optional:
- `stops: list[Union[str, int]]` - A list of strings and/or token IDs at which to end the generation early. Defaults to `[]`.
- `sampler: SamplerSettings` - The `ez.samplers.SamplerSettings` object used to control text generation. Defaults to `ez.samplers.DefaultSampling`.

## `Model.stream_print() -> str`

Given a prompt, stream text to a file as it is generated, and return the generated string. The returned string does not include the `end` parameter.

The following parameter is required:
- `prompt: str` - The text from which to generate

The following parameters are optional:
- `stops: list[Union[str, int]]` - A list of strings and/or token IDs at which to end the generation early. Defaults to `[]`.
- `sampler: SamplerSettings` - The `ez.samplers.SamplerSettings` object used to control text generation. Defaults to `ez.samplers.DefaultSampling`.
- `end: str` - A string to print after the generated text. Defaults to `\n`.
- `file: _SupportsWriteAndFlush` - The file where text should be printed. Defaults to `sys.stdout`.
- `flush: bool` - Whether to flush the stream after each token. The stream is always flushed at the end of generation.

## `Model.ingest() -> None`

Ingest the given text into the model's cache, to reduce latency of future generations that start with the same text.

The following parameter is required:
- `text: str` - The text to ingest

## `Model.candidates() -> list[tuple[str, np.floating]]`

Given prompt `str` and k `int`, return a sorted list of the top k candidates for most likely next token, along with their normalized probabilities.

The following parameters are required:
- `prompt: str` - The text to evaluate
- `k: int` - The number of candidate tokens to return

### Example usage
```python
>>> Mixtral.candidates('The dog says woof, but the cat says', 5)
[('▁me', 0.512151), ('▁“', 0.048059467), (',', 0.029822024), ('▁wo', 0.023914132), ('…', 0.023838354)]
```

## `Model.print_candidates() -> None`

Like `Model.candidates()`, but print the values instead of returning them.

The following parameters are required:
- `prompt: str` - The text to evaluate
- `k: int` - The number of candidate tokens to return

The following parameter is optional:
- `file: _SupportsWriteAndFlush` - The file where text should be printed. Defaults to `sys.stdout`.

### Example usage
```python
>>> Mixtral.print_candidates('The dog says woof, but the cat says', 5)
token '▁me' has probability 0.5121510028839111
token '▁“' has probability 0.04805946722626686
token ',' has probability 0.02982202358543873
token '▁wo' has probability 0.02391413226723671
token '…' has probability 0.023838354274630547
```

# `class ez.Thread`

Provide functionality to facilitate easy interactions with a Model

The following attributes are available:
- `.format` - The format being used for messages in this thread
- `.messages` - The list of messages in this thread
- `.model` - The `ez.Model` instance used by this thread
- `.sampler` - The `ez.SamplerSettings` object used in this thread

## `Thread.create_message() -> Message

Construct a message using the format of this thread. If you are looking to create a message and also add it to the Thread's message history, see `Thread.add_message()`.

The following parameters are required:
- `role: str` - The role of the message. Must be one of `'system'`, `'user'`, or `'bot'`. Case-insensitive.
- `content: str` - The content of the message.

## `Thread.len_messages() -> int`

Return the total length of all messages in this thread, in tokens. Equivalent to `len(Thread)`.

## `Thread.add_message() -> None`

Create a message and append it to `Thread.messages`.

`Thread.add_message(...)` is a shorthand for `Thread.messages.append(Thread.create_message(...))`

The following parameters are required:
- `role: str` - The role of the message. Must be one of `'system'`, `'user'`, or `'bot'`. Case-insensitive.
- `content: str` - The content of the message.

## `Thread.inference_str_from_messages() -> str`

Using the list of messages, construct a string suitable for inference, respecting the format and context length of this thread.

## `Thread.send() -> str`

Send a message in this thread. This adds your message and the bot's response to the list of messages. Returns a string containing the response to your message.

The following parameter is required:
- `prompt: str` - The content of the message to send

## `Thread.interact() -> None`

Start an interactive chat session using this Thread.

While text is being generated, press `^C` to interrupt the bot. Then you have the option to press `ENTER` to re-roll, or to simply type another message.

At the prompt, press `^C` to end the chat session.

End your input with a backslash `\` for multi-line input.

Type `!` and press `ENTER` to enter a basic command prompt. For a list of commands, type `help` at this prompt.

Type `<` and press `ENTER` to prefix the bot's next message, for example with `Sure!`.

Type `!!` at the prompt and press `ENTER` to insert a system message.

The following parameters are optional:
- `color: bool` - Whether to use colored text to differentiate user / bot. Defaults to `True`.
- `header: str` - Header text to print at the start of the interaction. Defaults to `None`.
- `stream: bool` - Whether to stream text as it is generated. If `False`, then print generated messages all at once. Defaults to `True`.

## `Thread.reset() -> None`

Clear the list of messages, which resets the thread to its original state.

## `Thread.as_string() -> str`

Return the thread's message history as a string

## `Thread.print_stats() -> None`

Print stats about the context usage in this thread. For example:

```
443 / 8192 tokens
5% of context used
7 messages
```

The following parameters are optional:
- `file: _SupportsWriteAndFlush` - The file where text should be printed. Defaults to `sys.stdout`.

# `class ez.thread.Message`

A dictionary representing a single message within a Thread

Normally, there is no need to interface with this class directly. Just use the methods of `Thread` to manage messages.

Works just like a normal `dict`, but adds a new method:

- `.as_string` - Return the full message string

Generally, messages have these keys:
- `role` - The role of the speaker: 'system', 'user', or 'bot'
- `prefix` - The text that prefixes the message content
- `content` - The actual content of the message
- `postfix` - The text that postfixes the message content

## `Message.as_string() -> str`

Return the full text of a message, including the prefix, content, and postfix.

# `class ez.samplers.SamplerSettings`

A SamplerSettings object specifies the sampling parameters that will be used to control text generation. It is passed as an optional parameter to `Thread()`, `Model.generate()`, `Model.stream()`, and `Model.stream_print()`.

## `ez.samplers.SamplerSettings.__init__()`

Construct a SamplerSettings object.

The following parameters are optional. If not specified, values will default to the current `llama.cpp` defaults.
- `max_len_tokens: int` - The maximum length of generations, in tokens. Set to less than 1 for unlimited.
- `temp: float` - The temperature value to use, which control randomness
- `top_p: float` - Nucleus sampling
- `min_p: float` - Min-P sampling
- `frequency_penalty: float` - Penalty applied to tokens based on the frequency with which they appear in the input
- `presence_penalty: float` - Flat penalty applied to tokens if they appear in the input
- `repeat_penalty: float` - Penalty applied to repetitive tokens
- `top_k: int` - The number of most likely tokens to consider when sampling

## Preset samplers

easy-llama comes with several built-in SamplerSettings objects that can be used out of the box for different purposes:
- `ez.samplers.GreedyDecoding` - Most likely next token is always chosen (temperature = 0.0)
- `ez.samplers.DefaultSampling` - Use `llama.cpp` default values for sampling (recommended for most cases)
- `ez.samplers.ClassicSampling` - Reflects old `llama.cpp` defaults
- `ez.samplers.SimpleSampling` - Original probability distribution
- `ez.samplers.SemiSampling` - Halfway between DefaultSampling and SimpleSampling
- `ez.samplers.TikTokenSampling` - Recommended for models with a large vocabulary, such as Llama 3 or Yi, which tend to run hot
- `ez.samplers.LowMinPSampling` - Use Min-P as the only active sampler (weak)
- `ez.samplers.MinPSampling` - Use Min-P as the only active sampler (moderate)
- `ez.samplers.StrictMinPSampling` - Use Min-P as the only active sampler (strict)
- `ez.samplers.ContrastiveSearch` - Use contrastive search with a moderate alpha value ([arXiv](https://arxiv.org/abs/2210.14140))
- `ez.samplers.WarmContrastiveSearch` - Use contrastive search with a high alpha value ([arXiv](https://arxiv.org/abs/2210.14140))
- `ez.samplers.RandomSampling` - Output completely random tokens from vocab (useless)
- `ez.samplers.LowTempSampling` - Default sampling with reduced temperature
- `ez.samplers.HighTempSampling` - Default sampling with increased temperature

# `ez.formats`

easy-llama comes with several built-in prompt formats that correspond to well-known language models or families of language models, such as Llama 3, Mistral Instruct, Vicuna, Guanaco, and many more. For a complete list of available formats, see [formats.py](easy_llama/formats.py).

Formats are instances of `dict`, and they look like this:

```python
# https://github.com/tatsu-lab/stanford_alpaca
alpaca: dict[str, Union[str, list]] = {
	"system_prefix": "",
	"system_prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
	"system_suffix": "\n\n",
	"user_prefix": "### Instruction:\n",
	"user_suffix": "\n\n",
	"bot_prefix": "### Response:\n",
	"bot_suffix": "\n\n",
	"stops": ['###', 'Instruction:', '\n\n\n']
}
```
