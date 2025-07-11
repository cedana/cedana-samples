# Use for generic transformers inference

import argparse
import time

from transformers import AutoModelForCausalLM, AutoTokenizer


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Configure model and sleep duration for the script.'
)
parser.add_argument(
    '--model',
    type=str,
    default='stabilityai/stablelm-2-1_6b',
    help="Model name or path to use. Default is 'stabilityai/stablelm-2-1_6b'.",
)
parser.add_argument(
    '--sleep',
    type=int,
    default=0,
    help='Duration (in seconds) to sleep before starting the loop. Default is 0 seconds.',
)
args = parser.parse_args()

# Load the tokenizer and model
print(f'Loading model: {args.model}')
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype='auto',
)
model.cuda()

# Sleep for the specified duration
time.sleep(args.sleep)

while True:
    user_input = 'some prompt'
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors='pt').to(model.device)
    # Generate tokens
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.70,
        top_p=0.95,
        do_sample=True,
    )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f'Generated Output:\n{output}')
