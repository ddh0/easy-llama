# easy-llama

*This project is still under development, and some functionality may be missing, incomplete, or broken. The documentation and examples on this page may be out of date.*

## Text generation in Python, made easy

```python
>>> import easy_llama as ez
>>> Mistral = ez.Model('Mistral-7B-v0.1-q8_0.gguf')
>>> Mistral.generate('The sky is')
' aflame with colour and the earth has been shaken, as the Lich King’s icy grip on Deathwing weakens. With his reign of terror nearing its end, it falls upon your shoulders to see that justice is served.\n\nFrom now until June 19th, Deathwing will appear in the skies above Azeroth and the Isles of Conquest will be transformed into a battlefield for the mightiest heroes of both factions.\n\nThe Burning Legion’s champion, Illidan Stormrage, will also join the battle against the Lich King in the Isles of Conquest as part of this event, bringing with him his powerful spell, Metamorphosis!\n\nFor more information about the upcoming Deathwing-themed content and other World of Warcraft events, please visit the official website.'
>>> 
```

easy-llama is designed to be as simple as possible to use, at the expense of some functionality. It is a layer of abstraction over [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), which itself provides the Python bindings for the underlying [llama.cpp](https://github.com/ggerganov/llama.cpp) library.

The following design choices are made:
- `Model.generate()` and `Model.stream()` require only one parameter, `prompt`
  - Other parameters `stops` and `sampler` are optional
- All model metadata is read from the GGUF file and exposed as a dictionary at `Model.metadata`
- Automatic context length extension using RoPE frequency scaling
- The following parameters are determined automatically:
  - `context_length`: read from GGUF metadata, may be set lower or higher
  - `num_gpu_layers`: set automatically where possible (CPU and Apple Silicon), otherwise can be set by user
  - `mmap`: sensible value based on BLAS backend
  - `mlock`: sensible value based on BLAS backend
  - `n_batch`: sensible value based on `os.cpu_count()`
  - `n_threads`: sensible value based on `os.cpu_count()`
  - `n_threads_batch`: `os.cpu_count()`, which is known to be optimal
  - `rope_freq_base`: native value from GGUF metadata, or multiplied according to requested context length
  - `mul_mat_q`: sensible value based on BLAS backend
- Extensive type hinting and helpful, informative error messages


## Other features
- ✅ Hardware acceleration on Apple Silicon (Metal), NVIDIA (CUDA), AMD (ROCm), and OpenBLAS
- ✅ Terminal-based interactive chat with text streaming
- ✅ Programmatic multi-turn interaction
- ✅ Optional time-aware interactions via timestamps in Threads
- ✅ Several common prompt formats built-in
  - `blank`, `chatml`, `llama2chat`, `alpaca`, `vicuna_lmsys`, `vicuna_common`, `mistral_openorca`, `dolphin`, `samantha`, `guanaco`, `orca_mini`, `airoboros`, `jackalope`, `mistral_instruct`, `zephyr`, `naberius`, `autocorrect`, `hermes`, `chatml_alpaca`, `monad`,  `orca`, `hexoteric`, `orcamaid`, `cat`, `mytho_cat`, `alpaca_strict`, `noromaid`, `solar_instruct`, `bagel`, `nschatml`
  - Easily extend, duplicate and modify built-in formats
  - `easy_llama.formats.wrap()` - Wrap a given string in any prompt format for single-turn completion
- ✅ Message-based context length handling
- ☑️ Retrieve sorted list of candidates for the most likely next token (i.e. logits)
- ✅ Support all models supported by llama.cpp. [See here](https://github.com/ggerganov/llama.cpp#description)

> ✅ implemented ⚠️ needs work ☑️ todo


## Examples

#### Programmatic chat
```python
>>> import easy_llama as ez
>>> Mistral = ez.Model('Mistral-7B-Instruct-v0.2-q8_0.gguf')
>>> Thread = ez.Thread(Mistral, ez.formats.mistral_instruct)
>>> Thread.send('Tell me a fun fact about Lions.')
'1. Male lions are called "mantles" because their manes resemble a cloak or mantle worn around their necks.\n2. Lions roar at an average volume of 50 decibels, which is about as loud as a car alarm.\n3. A group of lions is known as a "pride," and they often hunt together in coordinated teams.\n4. Lions are the only cats that live in groups with their cubs and non-related adults.\n5. Lions are excellent swimmers and can reach speeds of up to 36 miles per hour in water.'
>>> Thread.send('Now tell me a joke about them.')
"1. Why don't lions play cards in the wild? Too many cheetahs!\n2. What do you call a lion with no teeth? A gummy bear!\n3. How do lions like their jokes? With a lot of roar-ing laughter!\n4. Why did the lion join a band? He wanted to play the drums and be the king of rock!\n5. What do you call a lion that's bad at making decisions? A indecisive roar!"
>>> 
```

#### Interactive chat
```python
>>> import easy_llama as ez
>>> Dolphin = ez.Model('dolphin-2.1-mistral-7b-f16.gguf')
>>> Thread = ez.Thread(Dolphin, ez.formats.dolphin)
>>> Thread.interact()
```
```
  > What is HTML?

HTML (HyperText Markup Language) is the standard markup language for creating web pages and online content. It provides a structure for the content of a website, defining how text, images, videos, and other elements are displayed on a screen. By using various tags, developers can create a visually appealing and user-friendly experience for visitors to their websites. While HTML is primarily focused on layout and structure, it works closely with CSS (Cascading Style Sheets) for styling and JavaScript for adding interactive functionality.

  > How is that different from Python?

HTML is a markup language used for creating the structure and layout of web pages, while Python is a high-level programming language designed for general purpose tasks. They serve different purposes:

1. HTML is focused on presenting information in a structured format, allowing developers to create websites with text, images, videos, and interactive elements. It uses tags to define the content's structure and presentation.
2. Python, on the other hand, is a programming language that can be used for various tasks such as web development, data analysis, artificial intelligence, and more. It allows developers to create scripts or full-fledged applications with its easy-to-read syntax and extensive library support.

In summary, HTML is mainly about designing the appearance of a website, while Python enables you to write code to perform various tasks, including working with HTML to build web pages dynamically.

  > Thanks!

You're welcome! If you have any other questions or need help, feel free to ask. I'm here to assist you.

  > 
```

### Acknowledgments
Thank you to [Andrei Belten](https://github.com/abetlen) for [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), and to [Georgi Gerganov](https://github.com/ggerganov) for [llama.cpp](https://github.com/ggerganov/llama.cpp) and [GGML](https://github.com/ggerganov/ggml).

###### DISCLAIMER
All language models tend to produce writing that is factually inaccurate, stereotypically biased, and fundamentally disconnected from reality.

###### LICENSE
> [The Unlicense](LICENSE)
