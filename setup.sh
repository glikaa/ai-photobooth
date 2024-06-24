#!/bin/bash

# Install Git Large File Storage (LFS)
git lfs install

# Clone the IP-Adapter repository from Hugging Face
cd IP-Adapter
git clone https://huggingface.co/h94/IP-Adapter

# Move the models to the correct directory
mv IP-Adapter/models models
