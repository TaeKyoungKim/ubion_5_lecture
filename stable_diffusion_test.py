# 토큰 없이 바로 실행 가능
from diffusers import StableDiffusionXLPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # 공개 레포
    torch_dtype=dtype,
    use_safetensors=True
).to(device)

img = pipe("photorealistic portrait, 8k, studio lighting").images[0]
img.save("output.png")