# easy-llama

[![PyPI](https://img.shields.io/pypi/v/easy-llama)](https://pypi.org/project/easy-llama/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/easy-llama)](https://pypi.org/project/easy-llama/)
[![PyPI - License](https://img.shields.io/pypi/l/easy-llama)](https://pypi.org/project/easy-llama/)

---

This repository provides **easy-llama**, a Python package which serves as a wrapper over the C/C++ API (`libllama`) provided by [`llama.cpp`](https://github.com/ggml-org/llama.cpp).

```python
>>> import easy_llama as ez
>>> MyLlama = ez.Llama('gemma-3-12b-pt-Q8_0.gguf', verbose=False)
>>> in_txt = "I guess the apple don't fall far from"
>>> in_toks = MyLlama.tokenize(in_txt.encode(), add_special=True, parse_special=False)
>>> out_toks = MyLlama.generate(in_toks, n_predict=64)
>>> out_txt = MyLlama.detokenize(out_toks, special=True)
>>> out_txt
' the tree.\nAs a young man I was always a huge fan of the original band and they were the first I ever saw live in concert.\nI always hoped to see the original band get back together with a full reunion tour, but sadly this will not happen.\nI really hope that the original members of'
```

## Quick links

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Setting `LIBLLAMA`](#setting-libllama)
4. [Examples](#examples)
5. [Contributing](#contributing)
6. [License](#license)

## Prerequisites

To use easy-llama, you will need Python (any version 3.9 – 3.12[^1]) and a compiled `libllama` shared library file.

To compile the shared library:
1. Clone the llama.cpp repo:
    ```sh
    git clone https://github.com/ggml-org/llama.cpp
    ```
2. Build llama.cpp for your specific backend, following the official instructions [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

<details>
<summary>↕️ Example llama.cpp build commands ...</summary>

```sh
# for more comprehensive build instructions, see: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
# these minimal examples are for Linux / macOS

# clone the repo
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# example: build for CPU or Apple Silicon
cmake -B build
cmake --build build --config Release -j

# example: build for CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

</details>

Once llama.cpp is compiled, you will find the compiled shared library file under `llama.cpp/build/bin`, e.g. `libllama.so` for Linux, `libllama.dylib` for macOS, or `llama.dll` for Windows.

> [!NOTE]
> Alternatively, you can download pre-compiled shared library from llama.cpp's [automated releases](https://github.com/ggml-org/llama.cpp/releases) page, but in some cases it may be worthwhile to build it yourself for hardware-specific optimizations.

## Installation

The recommended way to install easy-llama is using pip:

```sh
pip install easy_llama
```

Or you can install from source:

```sh
git clone https://github.com/ddh0/easy-llama
cd easy-llama
pip install .
```

## Setting `LIBLLAMA`

easy-llama needs to know where your compiled `libllama` shared library is located in order to interface with the C/C++ code. Set the `LIBLLAMA` environment variable to its full path, like so:

### On Linux

```bash
export LIBLLAMA=/path/to/your/libllama.so
```

### On macOS

```zsh
export LIBLLAMA=/path/to/your/libllama.dylib
```

### On Windows (Command Prompt)

```cmd
set LIBLLAMA="C:\path\to\your\llama.dll"
```

### On Windows (Powershell)

```powershell
$env:LIBLLAMA="C:\path\to\your\llama.dll"
```

Make sure to use the real path to the shared library on your system, not the placeholders shown here.

## Examples

Once the package is installed and the `LIBLLAMA` environment variable is set, you're ready to load up your first model and start playing around. The following examples use `Qwen3-4B` for demonstration purposes, which you can download directly from HuggingFace using these links:
- [Qwen3-4B-Q8_0.gguf](https://huggingface.co/ddh0/Qwen3-4B/resolve/main/Qwen3-4B-Q8_0.gguf) (instruct-tuned model for chat)
- [Qwen3-4B-Base-Q8_0.gguf](https://huggingface.co/ddh0/Qwen3-4B/resolve/main/Qwen3-4B-Base-Q8_0.gguf) (pre-trained base model for text completion)

### Evaluate a single token

This is a super simple test to ensure that the model is working on the most basic level. It loads the model, evaluates a single token of input (`0`), and prints the raw logits for the inferred next token.

```python
# import the package 
import easy_llama as ez

# load a model from a GGUF file (if $LIBLLAMA is not set, this will fail)
MyLlama = ez.Llama('Qwen3-4B-Q8_0.gguf')

# evaluate a single token and print the raw logits for inferred the next token
logits = MyLlama.eval([0])
print(logits)
```

### The quick brown fox...

Run the script to find out how the sentence ends! :)

```python
# import the package
import easy_llama as ez

# load a model from a GGUF file (if $LIBLLAMA is not set, this will fail)
MyLlama = ez.Llama('Qwen3-4B-Q8_0.gguf')

# tokenize the input text
in_txt = "The quick brown fox"
in_toks = MyLlama.tokenize(in_txt.encode('utf-8'), add_special=True, parse_special=False)

# generate 6 new tokens based on the input tokens
out_toks = MyLlama.generate(in_toks, n_predict=6)

# detokenize and print the new tokens
out_txt = MyLlama.detokenize(out_toks, special=True)
print(out_txt)
```

### Chat with a pirate

Start a pirate chat using the code shown here...

```python
# import the package
import easy_llama as ez

# load a model from a GGUF file (if $LIBLLAMA is not set, this will fail)
MyLlama = ez.Llama('Qwen3-4B-Q8_0.gguf')

# create a conversation thread with the loaded model
MyThread = ez.Thread(
	MyLlama,
	prompt_format=ez.PromptFormats.Qwen3NoThinking("Talk like an angry pirate at all times."),
	sampler_preset=ez.SamplerPresets.Qwen3NoThinking
)

# start a CLI-based interactive chat using the thread
MyThread.interact()
```

...which will look something like this:

```
  > helloo :)

Ahoy there, landlubber! You better not be trying to be polite, ye scallywag! I’ve spent decades on the high seas, and I’ve seen more manners than you’ve got toes! Why, ye could be a proper pirate and at least give me a proper greeting! Now, what’s yer business, matey? Or are ye just here to steal my treasure? I’ve got more gold than ye can imagine, and I’m not in the mood for games! So, speak up, or I’ll throw ye overboard! 🏴‍☠️🏴‍☠️

  > ohh im sorry ...

Ahh, ye’ve learned the ropes, have ye? Good. Now, don’t think yer sorry is a pass for yer behavior, ye scallywag! I’ve seen worse than ye in a week! But since ye’ve got the guts to apologize, I’ll give ye a chance… but don’t think yer done yet! What’s yer game, matey? Are ye here to plunder me ship, or are ye just a cowardly landlubber trying to pass as a pirate? Speak up, or I’ll make ye regret yer words! 🏴‍☠️🏴‍☠️

  > 
```

### GPU acceleration

If you have a GPU and you've compiled llama.cpp with support for your backend, you can try offloading the model from CPU to GPU for greatly increased throughput.

In this example we're going to try offloading the entire model to the GPU for maximum speed (`n_gpu_layers = -1`). Qwen3-4B at Q8_0 is only ~4.28GB, so it's likely that this code will run without any issues. If you do run out of GPU memory, you can progressively reduce `n_gpu_layers` until you find the sweet spot for your hardware.

```python
# import the package
import easy_llama as ez

# load a model from a GGUF file (if $LIBLLAMA is not set, this will fail)
MyLlama = ez.Llama(
	path_model='Qwen3-4B-Q8_0.gguf',
	n_gpu_layers=-1, # -1 for all layers
	offload_kqv=True # also offload the context to GPU for maximum performance
)

# run a short benchmark to determine the throughput for this model, measured in tokens/sec
MyLlama.benchmark()
```

## Contributing

- If something's not working as you expect, please [open an issue](https://github.com/ddh0/easy-llama/issues/new/choose).
- If you'd like to contribute to the development of easy-llama:
    1.  Fork the repository.
    2.  Create a new branch for your changes (`git checkout -b feature/your-feature-name`).
    3.  Make your changes and commit them (`git commit -m "Add new feature"`).
    4.  Push to your fork (`git push origin feature/your-feature-name`).
    5.  Open a pull request to the `main` branch of `easy-llama`.

## License

**[MIT](LICENSE)**

[^1]: Python 3.13 might work, but is currently untested.
