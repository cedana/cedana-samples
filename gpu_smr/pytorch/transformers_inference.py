import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Initialize the input text
input_text = "What is the meaning of life?"

while True:
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate output
    output = model.generate(**inputs, max_new_tokens=50)

    # Decode and print the result
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(result)

    # Update the input text with the generated output
    input_text = result

    # Optional: Add a small delay to make the loop more readable
    import time
    time.sleep(1)
