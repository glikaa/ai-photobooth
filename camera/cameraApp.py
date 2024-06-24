import cv2
import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import numpy as np
from os.path import join, dirname, abspath
from PIL import Image, ImageTk

from face_crop_plus import Cropper
from stable_diffusion.pipeline import sd_process

INPUT_DIR = join(dirname(abspath(__file__)), "images")
OUTPUT_DIR = join(dirname(abspath(__file__)), "crop")

if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Photobooth")

        self.video_stream = cv2.VideoCapture(0)
        self.is_camera_opened = self.video_stream.isOpened()

        if not self.is_camera_opened:
            print("Error: Could not open camera.")
            exit()

        self.frame_label = ttk.Label(self.root)
        self.frame_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.capture_button = ttk.Button(self.root, text="take picture", command=self.start_countdown)
        self.capture_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.process_button = ttk.Button(self.root, text="start generation", command=self.process_photo)
        self.process_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.countdown_label = ttk.Label(self.root, text="", font=("Helvetica", 20))
        self.countdown_label.grid(row=2, column=0, columnspan=2, pady=10)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.video_stream.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.frame_label.imgtk = imgtk
            self.frame_label.configure(image=imgtk)
        else:
            print("Failed to capture frame.")
        self.root.after(10, self.update_frame)
    
    def start_countdown(self):
        self.capture_button.config(state=tk.DISABLED)
        threading.Thread(target=self.countdown).start()

    def countdown(self):
        for i in range(3, 0, -1):
            self.countdown_label.config(text=str(i))
            time.sleep(1)
        self.countdown_label.config(text="")
        self.capture_photo()
        self.capture_button.config(state=tk.NORMAL)

    def capture_photo(self):
            ret, frame = self.video_stream.read()
            if ret:
                #save photo
                original_filename = "captured_photo.jpg"
                cv2.imwrite(join(INPUT_DIR, original_filename), frame)
                print(f"Photo saved as {original_filename}")
                            
                # face_crop_plus to get image from folder and save to different folder
                crop_img = Cropper(face_factor=0.8, strategy="largest", output_size=(512, 512))
                crop_img.process_dir(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

                self.display_cropped_photo()

            else:
                print("Failed to capture photo.")
    
    def display_cropped_photo(self):
        # Display the cropped photo from the OUTPUT_DIR
        cropped_photo_path = join(OUTPUT_DIR, "captured_photo.jpg")
        if os.path.exists(cropped_photo_path):
            img = Image.open(cropped_photo_path)
            img = img.resize((640, 480), Image.ANTIALIAS)  # Resize image to fit the window
            img_tk = ImageTk.PhotoImage(img)
            self.frame_label.imgtk = img_tk
            self.frame_label.configure(image=img_tk)
            print(f"Displayed {cropped_photo_path}")
        else:
            print(f"Cropped photo {cropped_photo_path} not found.")
    
    def process_photo(self):
        cropped_photo_path = join(OUTPUT_DIR, "captured_photo.jpg")
        if os.path.exists(cropped_photo_path):
            img = Image.open(cropped_photo_path)
            img_rgb = img.convert("RGB")

            # Process the image using the Stable Diffusion pipeline
            sd_process(img_rgb)
            print(f"Processing {cropped_photo_path} with Stable Diffusion.")
        else:
            print(f"Cropped photo {cropped_photo_path} not found.")
    
    def on_closing(self):
        self.video_stream.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

#crop photo
#cropper = Cropper(face_factor=0.7, strategy="largest")
#cropper.process_dir(input_dir="path/to/images")

#save photo to path/to/images
