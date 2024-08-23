import json
import logging
import torch
from PIL import Image, PngImagePlugin
from diffusers import DiffusionPipeline
import copy
import random
import time
import csv



prompts = []
with open('./prompt.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        prompts.append(row[0])


lora_path = '../output/Yoneyama_Mai/Yoneyama_Mai_000009000.safetensors'
base_model = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(lora_path)
pipe.to("cuda")



trigger_word='anime style'
for prompt in prompts:
    generator = torch.Generator(device="cuda").manual_seed(random.randint(0, 1000000))
    trigger_word = 'anime style'
    image = pipe(
        prompt=f"{trigger_word}, {prompt}",
        num_inference_steps=28,
        guidance_scale=4,
        width=768,
        height=1280,
        generator=generator,
        joint_attention_kwargs={"scale": 0.9},
    ).images[0]
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters", f"{prompt}\n\n\n")
    image.save(f"generated_image/{prompts.index(prompt)}.png", pnginfo=metadata)

