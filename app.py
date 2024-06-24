import torch 
import tkinter as tk
from transformers import CLIPProcessor
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk
import cv2

from camera.cameraApp import CameraApp

# create python_app
app = tk.Tk()
app.geometry("670x650")
app.title("AI-Photoboot Dev") 

camera_app = CameraApp(app)

app.mainloop()
