# easy-llama

## Text generation in Python, made easy

```python
>>> import easy_llama as ez
>>> Mixtral = ez.Model('Mixtral-8x7B-v0.1-q8_0.gguf')
>>> Mixtral.generate('The sky is')
' an inverted bowl that covers the earth with an infinite dome, and the sun, moon and stars are attached to its inside surface. The sun revolves around the earth once a day and the stars revolve around the earth at night.\n\nThis is the basic theory of the geocentric model of the universe. This model was the accepted cosmological view of the world in Ancient Greece, and was adopted by the Roman Catholic Church and was the official view of the Church until 1600.\n\nThe geocentric model was based on the writings of Aristotle and Ptolemy, and the Catholic Church accepted it as an article of faith. The Catholic Church was opposed to any views that contradicted the geocentric model, and threatened with excommunication anyone who held to the heliocentric view.\n\nThe heliocentric model of the universe was first proposed by Aristarchus of Samos in the 3rd Century BC, and was based on the observation that the planets and the moon revolve around the sun. The sun is in the center of the universe and the earth and other planets revolve around it.\n\nThe heliocentric model was not accepted by the Catholic Church, and was not accepted as a scientific theory until the 1600s.'
>>> 
```

easy-llama's purpose is to make use of **on-device large language models (LLMs) as easily as possible**. It is a layer of abstraction over [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), which itself provides the Python bindings for the underlying [llama.cpp](https://github.com/ggerganov/llama.cpp) project.

## Documentation
The latest documentation is available [here](DOCS.md).

## Features
- [x] Automatic arbitrary context length extension
	- Just specify your desired context length, and easy-llama will adjust the necessary parameters accordingly
	- A warning will be displayed if the chosen context length is likely to cause a loss of quality
- [x] Terminal-based interactive chat
    - `Thread.interact()`  
	- Text streaming
	- Different colored text to differentiate user / bot
	- Some basic commands accessible by typing `!` and pressing `ENTER`:
		- `reroll` - Re-roll/swipe last response
		- `remove` - Remove last message
		- `reset` - Reset the Thread to its original state without re-loading model
		- `clear` - Clear the screen
		- `sampler` - View and modify sampler settings on-the-fly
- [x] Programmatic multi-turn interaction
	- `Thread.send(prompt)` -> `response`
	- Both your message and bot's message are added to Thread
- [x] Several common prompt formats built-in
  - accessible under `easy_llama.formats`
  - Stanford Alpaca, Mistral Instruct, Mistral Instruct Safe, ChatML, Llama2Chat, Llama3, Command-R, Vicuna LMSYS, Vicuna Common, Dolphin, Guanaco, & more
  - Easily extend, duplicate and modify built-in formats
  - `easy_llama.formats.wrap(prompt)` - Wrap a given prompt in any prompt format for single-turn completion
- [X] Message-based context length handling
- [ ] Retrieve sorted list of candidates for the most likely next token (i.e. logits)
- [X] Support all models supported by llama-cpp-python

## Examples

#### Programmatic chat
```python
>>> import easy_llama as ez
>>> Llama3 = ez.Model('Meta-Llama-3-8B-Instruct-q8_0.gguf')
>>> Thread = ez.Thread(Llama3, ez.formats.llama3, ez.samplers.TikTokenSampling)
>>> Thread.send('tell me a fun fact about lions')
'Roar! Hi there!\n\nHere\'s a fun fact about lions: Did you know that lions are social butterflies... er, I mean, social big cats? They live in prides, which are typically made up of several females, their cubs, and one or more males. But here\'s the cool part: lionesses often give each other "grooming massages" by licking and nuzzling each other to strengthen their bonds and show affection! Who knew lions could be so... pampered?\n\nWant to learn more fun facts about lions?'
>>> Thread.send('now tell me a joke about them')
'Here\'s one that\'s the "mane" event:\n\nWhy did the lion go to the dentist?\n\nBecause it had a ROAR-ing toothache! (get it?)\n\nHope that made you roar with laughter! Do you want another one?'
>>> 
```

#### Interactive chat
```python
>>> import easy_llama as ez
>>> Llama3 = ez.Model('Meta-Llama-3-8B-Instruct-q8_0.gguf')
>>> Thread = ez.Thread(Llama3, ez.formats.llama3, ez.samplers.TikTokenSampling)
>>> Thread.interact()
```

<details>
<summary>Click to expand interactive chat example</summary>

In practice, the prompt and the response are differentiated by color for improved readability.

```
  > What is HTML?

Hello there! I'm Llama 3, here to help you with any questions you may have.

HTML stands for HyperText Markup Language. It's the standard markup language used to create web pages. In simple terms, HTML is a way to add structure and content to a webpage, making it look and behave as intended.

HTML consists of a series of elements, represented by tags (<>). These tags are used to define different parts of a webpage, such as headings, paragraphs, images, links, forms, tables, and more. For example, the `<p>` tag is used to define a paragraph of text, while the `<img>` tag is used to add an image to a webpage.

HTML documents typically consist of:
  
1. Headings: <h1>, <h2>, <h3>, etc.
2. Paragraphs: <p>
3. Links: <a>
4. Images: <img>
5. Lists: <ul>, <ol>, <li>
6. Tables: <table>, <tr>, <td>
7. Forms: <form>, <input>, <textarea>, <select>

HTML is used in conjunction with other technologies, such as CSS (Cascading Style Sheets) for styling and JavaScript for adding interactivity to a webpage.

HTML5 is the latest version of HTML, which introduced new features like semantic elements, improved forms, and multimedia support. It's widely supported by modern web browsers and is used to build most websites on the internet today.

I hope that helps you understand what HTML is! Do you have any other questions about it?

  > How is that different from python?

Python and HTML are two completely different programming languages, with different purposes and uses. Here's a brief comparison:

**Purpose:**

* HTML (HyperText Markup Language) is used for creating the structure and content of web pages. Its primary purpose is to define the layout, format, and organization of information on the web.
* Python is a high-level, interpreted programming language that can be used for a wide range of purposes, including:
    + Web development (e.g., building web applications, scraping websites)
    + Data analysis and science
    + Machine learning and artificial intelligence
    + Automation and scripting
    + Game development

**Syntax:**

* HTML uses markup tags to define the structure and content of a webpage. It's primarily used for defining the layout and organization of information.
* Python uses indentation (spaces or tabs) to denote block-level syntax, and is typically written in a text editor or IDE.

**Programming paradigm:**

* HTML is a declarative language, meaning you describe what you want to see on the page, rather than how to achieve it. It's used for defining the structure of a webpage.
* Python is an imperative language, meaning you provide step-by-step instructions on how to solve a problem or perform a task.

**Output:**

* HTML produces an output that is rendered by web browsers, resulting in a visual representation of the content.
* Python can produce various outputs, such as:
    + Text files
    + Images
    + Audio files
    + Web pages (using frameworks like Flask or Django)
    + Data analysis results

In summary, HTML is used for creating the structure and content of web pages, while Python is a programming language that can be used for a wide range of purposes, including web development, data analysis, machine learning, and more. While they're both important tools in web development, they serve different purposes and are used in different ways.

Do you have any other questions about HTML or Python?

  > no that's it thanks

You're welcome! It was a pleasure chatting with you and helping you understand the basics of HTML. If you have any more questions or need further clarification, feel free to reach out anytime.  

Have a great day and happy coding (or web-browsing)!

  > 
```

</details>

## Installing with pip

In most cases, the best way to install easy-llama is through pip.

Select your backend from the list below to see your installation instructions. If you run into issues with the installation, please see the [llama-cpp-python installation instructions](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation) for a more detailed guide. If you're still having trouble, feel free to [open an issue](https://github.com/ddh0/easy-llama/issues/new/).

<details>
<summary>CPU only</summary>

```
pip uninstall llama-cpp-python -y
pip install --upgrade easy-llama
```

</details>
<details>
<summary>CUDA (for NVIDIA)</summary>

You will need `cmake` to install easy-llama. It is probably available in your preferred package manager, such as `apt`, `brew`, `yum`, etc. Or you can install it [from source](https://cmake.org/download/).

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --no-cache-dir llama-cpp-python 
pip install --upgrade easy-llama
```
</details>
<details>
<summary>Metal (for Apple Silicon)</summary>

You will need `cmake` to install easy-llama. It is probably available in your preferred package manager, such as `brew`. Or you can install it [from source](https://cmake.org/download/).

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --no-cache-dir llama-cpp-python 
pip install --upgrade easy-llama
```
</details>
<details>
<summary>ROCm (for AMD)</summary>

You will need `cmake` to install easy-llama. It is probably available in your preferred package manager, such as `apt`, `brew`, `yum`, etc. Or you can install it [from source](https://cmake.org/download/).

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install --no-cache-dir llama-cpp-python 
pip install --upgrade easy-llama
```
</details>
<details>
<summary>Vulkan</summary>

You will need `cmake` to install easy-llama. It is probably available in your preferred package manager, such as `apt`, `brew`, `yum`, etc. Or you can install it [from source](https://cmake.org/download/).

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install --no-cache-dir llama-cpp-python 
pip install --upgrade easy-llama
```
</details>
<details>
<summary>CLBlast</summary>

You will need `cmake` to install easy-llama. It is probably available in your preferred package manager, such as `apt`, `brew`, `yum`, etc. Or you can install it [from source](https://cmake.org/download/).

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install --no-cache-dir llama-cpp-python 
pip install --upgrade easy-llama
```
</details>
<details>
<summary>OpenBLAS</summary>

You will need `cmake` to install easy-llama. It is probably available in your preferred package manager, such as `apt`, `brew`, `yum`, etc. Or you can install it [from source](https://cmake.org/download/).

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install --no-cache-dir llama-cpp-python 
pip install --upgrade easy-llama
```
</details>
<details>
<summary>Kompute</summary>

You will need `cmake` to install easy-llama. It is probably available in your preferred package manager, such as `apt`, `brew`, `yum`, etc. Or you can install it [from source](https://cmake.org/download/).

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_KOMPUTE=on" pip install --no-cache-dir llama-cpp-python 
pip install --upgrade easy-llama
```
</details>
<details>
<summary>SYCL</summary>

You will need `cmake` to install easy-llama. It is probably available in your preferred package manager, such as `apt`, `brew`, `yum`, etc. Or you can install it [from source](https://cmake.org/download/).

```bash
pip uninstall llama-cpp-python -y
source /opt/intel/oneapi/setvars.sh
CMAKE_ARGS="-DLLAMA_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" pip install --no-cache-dir llama-cpp-python 
pip install --upgrade easy-llama
```
</details>

## Installing from source

Installation from source is only necessary if you rely on some functionality from llama.cpp that is not yet supported by llama-cpp-python. In most cases, you should prefer installing with pip.

> [!NOTE]
>
> You will need to modify the `CMAKE_ARGS` variable according to your backend. The arguments shown below are for CUDA support. If you're not using CUDA, select your backend above to see the correct `CMAKE_ARGS`.

To install easy-llama from source, copy and paste the following commands into your terminal:

```bash
pip uninstall easy-llama llama-cpp-python -y
rm -rf ./easy-llama
rm -rf ./llama-cpp-python
git clone https://github.com/abetlen/llama-cpp-python
cd ./llama-cpp-python/vendor/
rm -rf ./llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd -
git clone https://github.com/ddh0/easy-llama
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install -e ./llama-cpp-python
pip install -e ./easy-llama
```

## Acknowledgments
easy-llama stands on the shoulders of giants. Thank you to [Andrei Betlen](https://github.com/abetlen) for [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), and to [Georgi Gerganov](https://github.com/ggerganov) for [llama.cpp](https://github.com/ggerganov/llama.cpp) and [GGML](https://github.com/ggerganov/ggml). Thank you to all who have made contributions to these projects.

###### DISCLAIMER
All language models tend to produce writing that is factually inaccurate, stereotypically biased, and fundamentally disconnected from reality.

###### LICENSE
> [The Unlicense](LICENSE)
> 
> *easy-llama is free and unencumbered software released into the public domain*
