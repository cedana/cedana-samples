#!/usr/bin/env python3
"""
VLLM readiness probe runner with continuous inference loop.

Mimics the transformers-based readiness loop pattern.
"""

import argparse
import signal
import socket
import sys
import threading
import time

from vllm import LLM, SamplingParams


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def readiness_accept_loop(sock):
    while True:
        conn, _ = sock.accept()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="VLLM readiness probe runner.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path to load.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism degree. Default: 1",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="Seconds to sleep before initialization. Default: 0",
    )
    parser.add_argument(
        "--readiness-port",
        type=int,
        default=8888,
        help="TCP port for readiness probe. Default: 8888",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable lazy graph execution (compilation) instead of eager mode.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between inference iterations. Default: 30.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    time.sleep(args.sleep)

    # Initialize vLLM engine
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=not args.compile,
    )

    sampling_params = SamplingParams(max_tokens=32, temperature=0.7, top_p=0.95)
    readiness_socket = None
    first_pass = True

    while True:
        try:
            prompt = "Hello from vLLM readiness probe!"
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            print(f"Prompt: {prompt}\nGenerated Output:\n{generated_text}\n")

            if first_pass:
                # Mark readiness only after successful inference
                try:
                    readiness_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    readiness_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    readiness_socket.bind(("0.0.0.0", args.readiness_port))
                    readiness_socket.listen(1)
                    threading.Thread(
                        target=readiness_accept_loop,
                        args=(readiness_socket,),
                        daemon=True,
                    ).start()
                    print(f"Readiness TCP socket open on port {args.readiness_port}")
                except Exception as e:
                    print(f"Failed to open readiness socket: {e}")
                    sys.exit(2)
                first_pass = False

        except Exception as e:
            print(f"Inference error: {e}")
            time.sleep(5)
            continue

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
