#!/usr/bin/env python3
"""
VLLM inference
Supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
"""
import sys
import argparse
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLLM inference on a specified model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path to use for inference."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable model compilation. Default is False."
    )
    return parser.parse_args()

def inference(model, compile=False):
    prompts = ['What is the answer to life, the universe, and everything?']
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    try:
        # Load the model into GPU memory
        llm = LLM(model=model, swap_space=0, enforce_eager=False, compile=compile)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
    except Exception as e:
        raise e

def main():
    args = parse_args()
    inference(args.model, args.compile)

if __name__ == '__main__':
    main()
