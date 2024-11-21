import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Input text for inference
input_text = "What is the meaning of life?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
output = model.generate(**inputs, max_new_tokens=50)

# Decode and print the result
result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
