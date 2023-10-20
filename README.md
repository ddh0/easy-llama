# easy-llama
Natural text generation in Python, made easy

## Overview
TODO


## Features
- ✅ Human-level text generation using [Contrastive Search](https://huggingface.co/blog/introducing-csearch)
- ✅ Automatic detection of model's native context length via GGUF
- ✅ Hardware acceleration on Apple Silicon (Metal), NVIDIA (CUDA), AMD (ROCm), OpenBLAS
- ✅ Programmatic multi-turn interaction
- ✅ Terminal-based interactive chat
- ⚠️ Several common prompt formats built-in, e.g Alpaca, Vicuna, Llama2, ChatML, etc
- ⚠️ Message-based context-length handling
- ☑️ Retrieve sorted list of candidates for the most likely next token (i.e. logits)
- ✅ Support all [models](https://github.com/ggerganov/llama.cpp#description) supported by llama.cpp

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

TODO
