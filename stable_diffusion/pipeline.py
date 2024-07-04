import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from os.path import join, dirname, abspath

#full face
from ip_adapter import IPAdapterFull

def sd_process(file):
    try:    
        #folder for results
        OUTPUT_DIR = join(dirname(abspath(__file__)), "generations")

        if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
        
        #file.show()
        print(f"Image '{file}' loaded successfully.")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        #set models
        base_model_path =  os.path.join(base_dir, "models", "models--SG161222--Realistic_Vision_V4.0_noVAE")
        vae_model_path = os.path.join(base_dir, "models", "models--stabilityai--sd-vae-ft-mse")
        ip_ckpt = os.path.join(base_dir, "models", "ip-adapter-full-face_sd15.bin")
        device = "cuda"
        image_encoder_path = os.path.join(base_dir, "models", "image_encoder")

        #set scheduler
        noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        )

        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        # load SD pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
        )
        print("Trying to load model...")
        #set IP adapter
        ip_model = IPAdapterFull(pipe, image_encoder_path, ip_ckpt, device, num_tokens=257)

        print("Generating images...")

        #generation 
        images = ip_model.generate(
                prompt="A sepia-toned vintage photograph of a young student sitting casually on an ornate chair. He is wearing a formal suit with a graduation cap and has a relaxed, confident posture. The man is exuding a scholarly and nonchalant vibe. The setting is a simple studio with a plain backdrop and a curtain on one side, adding to the old-fashioned, antique aesthetic.",
                negative_prompt="deformed iris, deformed pupils, semi-realistic, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                pil_image=file,
                num_samples=4,
                guidance_scale=12,
                num_inference_steps=50,
                scale=0.75
        )

       
        return images
    except Exception as e:
          print(f"Error: {e}")