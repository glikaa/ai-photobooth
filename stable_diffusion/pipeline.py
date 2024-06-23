import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from os.path import join, dirname, abspath

#full face
#from ip_adapter import IPAdapterFull

#folder for results
OUTPUT_DIR = join(dirname(abspath(__file__)), "generations")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#set image from crop-folder to use as input 
project_folder = os.path.dirname(os.path.dirname(__file__))
folder_path = os.path.join(project_folder, "camera", "crop")
input_image = "captured_photo.jpg"
file_path = os.path.join(folder_path, input_image)

# Check if the file exists
if os.path.exists(file_path):
    # Open the image using PIL.Image
    img = Image.open(file_path)
    print(f"Image '{input_image}' loaded successfully.")

    # Convert the image to RGB
    img_rgb = img.convert('RGB')
    print(f"Image '{input_image}' converted to RGB.")

    img_rgb.show()  # This will open the image using the default image viewer

else:
    print(f"File '{input_image}' does not exist in folder '{folder_path}'.")

"""
#set models
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = "models/ip-adapter-plus-face_sd15.bin"
device = "cuda"
image_encoder_path = "models/image_encoder"

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

#set IP adapter
ip_model = IPAdapterFull(pipe, image_encoder_path, ip_ckpt, device, num_tokens=257)

#generation 
images = ip_model.generate(
        prompt="astronaut, portrait, high quality",
        negative_prompt="deformed iris, deformed pupils, semi-realistic, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        pil_image=img_rgb,
        num_samples=4,
        guidance_scale=12,
        num_inference_steps=50,
        scale=0.75
)

#save image to folder 
"""