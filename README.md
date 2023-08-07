# easy-llama - Simplified deployment of llama models in Python

**easy-llama** allows developers that are unfamiliar with language models to
deploy them in an simplified manner, without sacrificing core
functionality. We support the LLaMA and Llama 2 models released by Meta AI,
and any model that derives from them.

# easy-llama is...
...easy!
```python
>>> import easy_llama
>>> Orca = easy_llama.Model('./orca-mini-7B.bin')
>>> Orca.generate('The sky is ')
'360 degrees of blue.\nThe ocean is a vast expanse of deep, dark blue.\nI am in awe of the beauty around me.'
>>> 
```

...powerful!
```python
>>> import easy_llama
>>> Guanaco = easy_llama.Model('./guanaco-33B.bin')
>>> Guanaco.generate('The breeze was crisp against her skin, and ', preset=easy_llama.Presets.Creative, max_length=128, stop_sequences=['\n\n'])
'icy rain splattered on the sidewalk. It had been a long day at work, but she didn’t mind. She loved being out in the city this time of year. There were so many people rushing around with their umbrellas and holiday shopping bags that it made her feel alive.\nShe stopped to look through the window of an antique shop on the corner. The store was filled with old books, vintage jewelry, and delicate glass figurines. She could have spent hours there if she had more time. As she turned away from the window, a man'
>>> 
```
