# easy-llama - Simplified deployment of Llama models in Python

easy-llama allows developers that are unfamiliar with large language models to deploy them in an simplified manner, without sacrificing core functionality. It supports LLaMA and Llama 2 models released by Meta AI, and any model that derives from them.

> *this project is unfinished and under active development*

You can install easy-llama with pip:
```
pip install easy-llama
```

# easy-llama is...
### ...easy!
```python
>>> import easy_llama
>>> Orca = easy_llama.Model('./orca-mini-7B.bin')
>>> Orca.generate('The sky is ')
'360 degrees of blue.\nThe ocean is a vast expanse of deep, dark blue.\nI am in awe of the beauty around me.'
>>> 
```

### ...powerful!
```python
>>> import easy_llama
>>> Guanaco = easy_llama.Model('./guanaco-33B.bin', context_length=8192)
>>> Guanaco.generate('The breeze was crisp against her skin, and ', preset=easy_llama.Presets.Creative, max_length=128, stop_sequences=['\n\n'])
'icy rain splattered on the sidewalk. It had been a long day at work, but she didn’t mind. She loved being out in the city this time of year. There were so many people rushing around with their umbrellas and holiday shopping bags that it made her feel alive.\nShe stopped to look through the window of an antique shop on the corner. The store was filled with old books, vintage jewelry, and delicate glass figurines. She could have spent hours there if she had more time. As she turned away from the window, a man'
>>> 
```

### ...portable!
I aim to make easy-llama completely portable between Linux, macOS and Windows, without you having to change any code. If this isn't your experience, please [open an issue](https://github.com/ddh0/easy-llama/issues).

### ...documented!
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
> easy-llama is licensed under the terms of the MIT license.
