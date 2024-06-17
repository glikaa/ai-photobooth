import torch 
import tkinter as tk
from transformers import CLIPProcessor
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk

# create python_app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 

app.mainloop()
