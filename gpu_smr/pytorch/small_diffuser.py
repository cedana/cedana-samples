#!/usr/bin/env python3

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# load pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

print("sleeping for 10s at the point to checkpoint")

# generate
prompt = "Spiderman is surfing. Darth Vader is also surfing and following Spiderman"
video_frames = pipe(prompt, num_inference_steps=25, num_frames=200).frames

# convent to video
video_path = export_to_video(video_frames)
