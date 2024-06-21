import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import threading
import time
import numpy as np
from face_crop_plus import Cropper

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
        self.frame_label.grid(row=0, column=0, columnspan=2)

        self.capture_button = ttk.Button(self.root, text="Capture Photo", command=self.start_countdown)
        self.capture_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.countdown_label = ttk.Label(self.root, text="", font=("Helvetica", 20))
        self.countdown_label.grid(row=2, column=0, columnspan=2, pady=10)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.video_stream.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.add_overlay(frame)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.frame_label.imgtk = imgtk
            self.frame_label.configure(image=imgtk)
        else:
            print("Failed to capture frame.")
        self.root.after(10, self.update_frame)

    def add_overlay(self, frame):
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Dimensions of the image
        width, height = pil_img.size

        # Coordinates for the oval (circle)
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 4  # Radius of the circle

        # Bounding box for the oval
        left = center_x - radius
        top = center_y - radius
        right = center_x + radius
        bottom = center_y + radius

        # Draw oval
        draw.ellipse([left, top, right, bottom], outline="red", width=5)

        return np.array(pil_img)

    
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
            output_filename = "captured_photo.jpg"
            cv2.imwrite(output_filename, frame)
            print(f"Photo saved as {output_filename}")

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
