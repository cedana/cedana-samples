from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-2-1_6b")
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablelm-2-1_6b",
    torch_dtype="auto",
)
model.cuda()

while True:
    user_input = "some prompt" 
    
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    
    # Generate tokens
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.70,
        top_p=0.95,
        do_sample=True,
    )
    
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f"Generated Output:\n{output}")
