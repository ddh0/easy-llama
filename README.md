# easy-llama - Simple, on-device text inference in Python

This project is currently unfinished and under heavy development.

COMING SOON: You can install easy-llama with pip:
```
pip install easy-llama
```

# Showcase
### Basic usage
```python
>>> import easy_llama as ez
>>> Orca = ez.Model('./orca-mini-7B.bin')
>>> Orca.generate('The sky is ')
'360 degrees of blue.\nThe ocean is a vast expanse of deep, dark blue.\nI am in awe of the beauty around me.'
>>> 
```

### Creative writing
```python
>>> import easy_llama as ez
>>> Guanaco = ez.Model('./guanaco-33B.bin')
>>> Guanaco.generate('The breeze was crisp against her skin, and ', preset=ez.Presets.Creative, max_length=128, stop_sequences=['\n\n'])
'icy rain splattered on the sidewalk. It had been a long day at work, but she didn’t mind. She loved being out in the city this time of year. There were so many people rushing around with their umbrellas and holiday shopping bags that it made her feel alive.\nShe stopped to look through the window of an antique shop on the corner. The store was filled with old books, vintage jewelry, and delicate glass figurines. She could have spent hours there if she had more time. As she turned away from the window, a man'
>>> 
```

# Support
### Platforms
I aim to make easy-llama completely portable between Linux, macOS and Windows, without you having to change any code. If this isn't your experience, please [open an issue](https://github.com/ddh0/easy-llama/issues).

### Documentation
Official documentation and FAQ can be found [here](https://github.com/ddh0/easy-llama/blob/main/docs.md).

# Thank you!
easy-llama stands on the shoulders of giants. Thank you to the following:
- [Andrei Belten](https://github.com/abetlen) for [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), which this project relies on heavily
- [Georgi Gerganov](https://github.com/ggerganov) for the underlying implementation of [llama.cpp](https://github.com/ggerganov/llama.cpp) and [GGML](https://github.com/ggerganov/ggml)
- [Tom Jobbins](https://huggingface.co/TheBloke) for providing the community with an enormous selection of model quantizations
- [Meta AI](https://ai.meta.com/) for training and releasing LLaMA, Llama 2, and Llama-2-Chat

###### DISCLAIMER
Language models tend to produce writing that is factually inaccurate, stereotypically biased, and completely disconnected from reality. You should never rely on a language model for anything important.

###### LICENSE
This project is licensed under the terms of the [MIT license](https://github.com/ddh0/easy-llama/blob/main/LICENSE).
