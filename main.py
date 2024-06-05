import torch 
import tkinter as tk
from transformers import CLIPProcessor
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk
from authtoken import auth_token

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 

prompt = tk.Entry(master=app, width=64) 
prompt.place(x=10, y=10)

lmain = tk.Label(master=app, height=32, width=64)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant="fp16", torch_dtype=torch.float32, use_auth_token=auth_token) 
pipe.to(device) 

def generate(): 
    image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0].cpu().numpy() # Convert to numpy array
    img = Image.fromarray(image)
    img = ImageTk.PhotoImage(img)
    lmain.configure(image=img) 
    lmain.image = img

trigger = tk.Button(master=app, height=2, width=10, text="Generate", command=generate) 
trigger.place(x=206, y=60) 

app.mainloop()
