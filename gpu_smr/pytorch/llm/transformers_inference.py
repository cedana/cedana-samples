# Use for generic transformers inference

import argparse
import signal
import socket
import sys
import threading
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
parser.add_argument(
    '--readiness-port',
    type=int,
    default=8888,
    help='Port to listen on for readiness TCP probe. Default is 8888.',
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


def readiness_accept_loop(sock):
    while True:
        conn, addr = sock.accept()
        conn.close()


readiness_socket = None
first_pass = True

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

    if first_pass:
        # Open TCP socket for k8s readiness (only opens once)
        try:
            readiness_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            readiness_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            readiness_socket.bind(('0.0.0.0', args.readiness_port))
            readiness_socket.listen(1)
            threading.Thread(
                target=readiness_accept_loop,
                args=(readiness_socket,),
                daemon=True,
            ).start()
            print(f'Readiness TCP socket open on port {args.readiness_port}')
        except Exception as e:
            print(f'Failed to open readiness socket: {e}')
            sys.exit(2)
        first_pass = False
