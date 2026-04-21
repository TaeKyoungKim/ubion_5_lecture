# realistic_image_gen.py
# Stable Diffusion XL - 실사 이미지 생성 (완전 무료, 공개)

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

# ──────────────────────────────────────────
# 모델 로드 (자동 다운로드, 약 6GB)
# ──────────────────────────────────────────
def load_pipeline(device="cuda"):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    # 빠른 샘플러로 교체
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()   # VRAM 절약
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

# ──────────────────────────────────────────
# 실사 인물 사진 생성
# ──────────────────────────────────────────
def generate_realistic_portrait(pipe, seed=42, save_dir="./outputs"):
    os.makedirs(save_dir, exist_ok=True)
    generator = torch.Generator("cuda").manual_seed(seed)

    # 실사 품질 극대화 프롬프트
    positive_prompt = """
        portrait photo of a 30 year old woman,
        photorealistic, ultra detailed, 8k uhd,
        professional studio lighting, shallow depth of field,
        shot on Canon EOS R5, 85mm lens, f/1.8,
        natural skin texture, subsurface scattering,
        cinematic color grading
    """
    negative_prompt = """
        cartoon, anime, illustration, painting,
        blurry, low quality, deformed, ugly,
        extra fingers, bad anatomy, watermark
    """

    result = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=30,   # 품질-속도 트레이드오프
        guidance_scale=7.5,       # 프롬프트 준수도 (7~9 권장)
        generator=generator,
        num_images_per_prompt=4
    )

    for i, img in enumerate(result.images):
        path = f"{save_dir}/portrait_seed{seed}_{i}.png"
        img.save(path)
        print(f"✓ 저장: {path}")

    return result.images

# ──────────────────────────────────────────
# 실사 풍경/배경 생성
# ──────────────────────────────────────────
def generate_realistic_scene(pipe, scene_type="city", seed=100):
    prompts = {
        "city": "photorealistic Seoul city street at golden hour, "
                "wet pavement reflections, bokeh lights, "
                "shot on Sony A7R IV, cinematic",
        
        "nature": "hyperrealistic dense forest in morning mist, "
                  "god rays through trees, macro detail, "
                  "National Geographic style photography",
        
        "interior": "luxury modern apartment interior, "
                    "architectural photography, soft natural light, "
                    "ultra realistic, 4K render"
    }
    
    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt=prompts[scene_type],
        negative_prompt="cartoon, CGI obvious, low quality, artifacts",
        width=1024, height=768,
        num_inference_steps=30,
        guidance_scale=8.0,
        generator=generator
    )
    result.images[0].save(f"./outputs/{scene_type}_{seed}.png")
    return result.images[0]

# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {DEVICE}")
    
    pipe = load_pipeline(DEVICE)
    
    print("\n=== 실사 인물 생성 ===")
    imgs = generate_realistic_portrait(pipe, seed=42)
    
    print("\n=== 실사 도시 풍경 생성 ===")
    scene = generate_realistic_scene(pipe, "city", seed=777)
    
    print("\n✅ 전체 완료!")