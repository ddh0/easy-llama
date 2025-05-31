# import the package 
import easy_llama as ez

# load a model from a GGUF file (if $LIBLLAMA is not set, this will fail)
MyLlama = ez.Llama('Qwen3-4B-Q8_0.gguf')

# evaluate a single token and print the raw logits for inferred the next token
logits = MyLlama.eval([0])
print(logits)