#!/usr/bin/env python3
"""
Generic VLLM inference runner with optional tensor parallelism.

Supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
"""

import argparse
import signal
import socket
import sys
import threading

from vllm import LLM, SamplingParams


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run VLLM inference on a specified model.'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name or path to use for inference.',
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Enable lazy graph execution (compilation). Default is eager mode.',
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism (default: 1).',
    )
    parser.add_argument(
        '--prompt-file',
        type=str,
        help='Path to file with prompts (one per line). If not given, read from stdin or use a default prompt.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8).',
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p sampling value (default: 0.95).',
    )
    parser.add_argument(
        '--readiness-port',
        type=int,
        default=8888,
        help='Port to listen on for readiness TCP probe. Default is 8888.',
    )
    return parser.parse_args()


def readiness_accept_loop(sock):
    while True:
        conn, addr = sock.accept()
        conn.close()


readiness_socket = None
first_pass = True


def load_prompts(args):
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        if not sys.stdin.isatty():
            return [line.strip() for line in sys.stdin if line.strip()]
        return ['What is the answer to life, the universe, and everything?']


def inference(
    model,
    compile=False,
    tensor_parallel_size=1,
    temperature=0.8,
    top_p=0.95,
    prompts=None,
    readiness_port=8888,
):
    global readiness_socket, first_pass
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    try:
        llm = LLM(
            model=model,
            swap_space=0,
            enforce_eager=not compile,  # compile=True → lazy execution
            tensor_parallel_size=tensor_parallel_size,
        )
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n')

        if first_pass:
            # Open TCP socket for k8s readiness (only opens once)
            try:
                readiness_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM
                )
                readiness_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
                )
                readiness_socket.bind(('0.0.0.0', readiness_port))
                readiness_socket.listen(1)
                threading.Thread(
                    target=readiness_accept_loop,
                    args=(readiness_socket,),
                    daemon=True,
                ).start()
                print(f'Readiness TCP socket open on port {readiness_port}.')
            except Exception as e:
                print(f'Failed to open readiness socket: {e}')
                sys.exit(2)
            first_pass = False
    except Exception as e:
        raise e


def main():
    args = parse_args()
    prompts = load_prompts(args)
    while True:
        inference(
            args.model,
            compile=args.compile,
            tensor_parallel_size=args.tensor_parallel_size,
            temperature=args.temperature,
            top_p=args.top_p,
            prompts=prompts,
            readiness_port=args.readiness_port,
        )


def handle_exit(signum, frame):
    sys.exit(1)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_exit)
    main()
