from face_crop_plus import Cropper

cropper = Cropper(face_factor=0.7, strategy="largest")
cropper.process_dir(input_dir="path/to/images")