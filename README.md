# easy-llama

*This project is still under heavy development, and some functionality may be missing, incomplete, or broken. The documentation and examples on this page may be out of date.*

## Natural text generation in Python, made easy
easy-llama is designed to be as simple as possible to use, at the expense of some functionality. It is a layer of abstraction over [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python), which itself provides the Python bindings for the underlying [`llama.cpp`](https://github.com/ggerganov/llama.cpp) library.

All generations utilize **contrastive search**, which [has been shown](#references) to produce more human-like text. The following hyperparameters are chosen:
```math
a=0.55, k=4
```
where $a$ is the degeneration penalty—which limits the similarity of new tokens to the tokens in the context, leading to more varied and less repetitive outputs—and $k$ is the number of candidate tokens that are considered from the language model's probability distribution.

The following design choices are made:
- `Model.generate()` takes only two parameters
  - `prompt` is the text to be evaluated by the model
  - `stops` is list of strings at which to end the generation early. defaults to `None`
- Context length is set automatically thanks to GGUF
- `n_batch`, `n_threads`, `n_threads_batch`, and `MUL_MAT_Q` are determined automatically
- On Apple Silicon and CPU, `NUM_GPU_LAYERS` is set automatically
- Extensive type hinting and helpful, informative error messages


## Other features
- ✅ Hardware acceleration on Apple Silicon (Metal), NVIDIA (CUDA), AMD (ROCm), and OpenBLAS
- ✅ Terminal-based interactive chat with text streaming
- ✅ Programmatic multi-turn interaction
- ✅ Several common prompt formats built-in
  - `blank`, `chatml`, `llama2chat`, `alpaca`, `vicuna`, `mistral_instruct`, `mistral_openorca`, `dolphin`, `samantha`, `guanaco`, `orca_mini`, `airoboros`
- ✅ Message-based context-length handling
- ☑️ Retrieve sorted list of candidates for the most likely next token (i.e. logits)
- ✅ Support all models supported by llama.cpp. [See here](https://github.com/ggerganov/llama.cpp#description)

> ✅ implemented ⚠️ needs work ☑️ todo


## Examples
#### Basic example
```python
>>> import easy_llama as ez
>>> Carl = ez.Model('llama-2-13b.Q6_K.gguf')
>>> Carl.generate('The sky is')
' blue and the sun is shining.\nI have no idea why I feel so happy today, but I do. It\'s a good feeling.\nThe weather has been beautiful for weeks now. I love springtime in Florida, it\'s just perfect.\nMy husband and I went to see "R'
>>> 
```

#### Programmatic chat example
```python
>>> import easy_llama as ez
>>> Mistral = ez.Model('mistral-7B-instruct-v0.1-f16.gguf')
>>> Thread = ez.Thread(Mistral, format=ez.mistral_instruct)
>>> Thread.send('Tell me a fun fact about Lions.')
'1. Male lions are called "mantles" because their manes resemble a cloak or mantle worn around their necks.\n2. Lions roar at an average volume of 50 decibels, which is about as loud as a car alarm.\n3. A group of lions is known as a "pride," and they often hunt together in coordinated teams.\n4. Lions are the only cats that live in groups with their cubs and non-related adults.\n5. Lions are excellent swimmers and can reach speeds of up to 36 miles per hour in water.'
>>> Thread.send('Now tell me a joke about them.')
"1. Why don't lions play cards in the wild? Too many cheetahs!\n2. What do you call a lion with no teeth? A gummy bear!\n3. How do lions like their jokes? With a lot of roar-ing laughter!\n4. Why did the lion join a band? He wanted to play the drums and be the king of rock!\n5. What do you call a lion that's bad at making decisions? A indecisive roar!"
>>> 
```

#### Interactive chat example
```python
>>> import easy_llama as ez
>>> Dolphin = ez.Model('dolphin-2.1-mistral-7b-f16.gguf')
>>> Thread = ez.Thread(Dolphin, format=ez.Formats.Dolphin)
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

### References
- [arXiv:2210.14140](https://arxiv.org/abs/2210.14140) - Contrastive Search Is What You Need For Neural Text Generation (25 Oct 2022, Yixuan Su, Nigel Collier)
- [HuggingFace](https://huggingface.co/blog/introducing-csearch) - Generating Human-level Text with Contrastive Search in Transformers 🤗 (Nov 8 2022)

###### DISCLAIMER
All language models tend to produce writing that is factually inaccurate, stereotypically biased, and fundamentally disconnected from reality.

###### LICENSE
This project is licensed under the terms of the [MIT license](LICENSE).
