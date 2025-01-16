#!/usr/bin/env python3

import torch
import time
from diffusers import FluxPipeline

pipe_cpu = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe = pipe_cpu.to("cuda:0")
print("sleeping for 20s after loaded into gpu memory")
time.sleep(20)
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=25,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
