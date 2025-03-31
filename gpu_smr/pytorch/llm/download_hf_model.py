#!/usr/bin/env python

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download model weights from Hugging Face.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path to download (e.g., 'stabilityai/stablelm-2-1_6b')."
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Download only the tokenizer, not the model weights."
    )
    args = parser.parse_args()

    print(f"Downloading from {args.model}...")

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Tokenizer downloaded successfully to the default HuggingFace cache")

    if not args.tokenizer_only:
        print("Downloading model weights (this may take some time)...")
        model = AutoModelForCausalLM.from_pretrained(args.model)
        print("Model weights downloaded successfully to the default HuggingFace cache")

    print("Download complete!")
    print(f"The model has been cached in the default HuggingFace cache directory.")

if __name__ == "__main__":
    main()
