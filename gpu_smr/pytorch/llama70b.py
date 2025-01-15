#!/usr/bin/env python3

import transformers
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"}
]

# Infinite loop to simulate continuous conversation
while True:
    # Get user input
    user_input = input("User: ")

    # Append the user message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Generate the model's response
    prompt = "\n".join([message["content"] for message in messages])

    # Generate a new output
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
    )

    # Extract the generated text
    model_response = outputs[0]["generated_text"]

    # Print the model's response
    print(f"Pirate Chatbot: {model_response.split('\n')[-1]}")

    # Append the model's response to the conversation history for context
    messages.append({"role": "assistant", "content": model_response.split('\n')[-1]})
