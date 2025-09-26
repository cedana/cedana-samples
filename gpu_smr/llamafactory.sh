#!/bin/bash

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
