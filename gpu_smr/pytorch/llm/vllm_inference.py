#!/usr/bin/env python3
"""
Generic VLLM inference runner.

Supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
"""

import argparse
import signal
import sys

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VLLM inference on a specified model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path to use for inference.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable lazy graph execution (compilation). Default is eager mode.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to file with prompts (one per line). If not given, read from stdin or use a default prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value (default: 0.95).",
    )
    return parser.parse_args()


def load_prompts(args):
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # If stdin is piped in, read prompts from it
        if not sys.stdin.isatty():
            return [line.strip() for line in sys.stdin if line.strip()]
        # Fallback default
        return ["What is the answer to life, the universe, and everything?"]


def inference(model, compile=False, temperature=0.8, top_p=0.95, prompts=None):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    try:
        llm = LLM(
            model=model,
            swap_space=0,
            enforce_eager=not compile,  # compile=True -> lazy execution
        )
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n")
    except Exception as e:
        raise e


def main():
    args = parse_args()
    prompts = load_prompts(args)
    inference(
        args.model,
        compile=args.compile,
        temperature=args.temperature,
        top_p=args.top_p,
        prompts=prompts,
    )


def handle_exit(signum, frame):
    sys.exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    main()

