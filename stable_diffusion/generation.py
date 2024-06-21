from pipe_config import ip_model, image_grid

#get image from take_photo() to use as input for IP model

images = ip_model.generate(
        prompt="student at graduation, portrait, high quality",
        negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        pil_image=input_image,
        num_samples=4,
        guidance_scale=7.5,
        num_inference_steps=50,
        seed=-1,
        scale=0.75
)
grid = image_grid(images, 1, 4)

#export generated image and display 
grid