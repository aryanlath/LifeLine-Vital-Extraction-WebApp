import streamlit as st
import cv2
from PIL import Image
from Screen_extraction import screen_extraction
from Number_detection import number_detection
from Number_classification import final_inference
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn, optim
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import timm
import segmentation_models_pytorch as smp
import imutils
from skimage.transform import ProjectiveTransform

import os
from tqdm import tqdm
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
import gc
import glob

import random
import yolov5
import paddleocr
from paddleocr import PaddleOCR,draw_ocr
from ensemble_boxes import *
import re
import torch.nn.functional as F
import copy
import tempfile
import time

root = '/Users/aryanlath/Downloads/VitalView App/Saved Models/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Corner Regression Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet34', pretrained=True, num_classes=768)
        self.ll = nn.Linear(768,8)
    
    def forward(self, img):
        x = self.model(img)
        x = self.ll(x)

        return x

model_reg = CNN()

## UNET++ Model
ENCODER = 'resnext101_32x8d'
ENCODER_WEIGHTS = 'imagenet'

model_unet = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=1,
    in_channels=3,
    activation=None,
)

model_reg = model_reg.to(device)
model_unet = model_unet.to(device)

model_reg.load_state_dict(torch.load(root + 'corner_reg.pt', map_location=device))
model_unet.load_state_dict(torch.load(root + 'unet++_weights.pt', map_location=device))

preprocessing_unet = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def vital_extraction(img_path, mode = 'accurate'):

  if mode == 'accurate':

    img = cv2.imread(img_path)
    transformed_image = screen_extraction(img_path, model_unet, model_reg, preprocessing_unet, mode)

    detection_dict = number_detection(transformed_image, mode)

    output_dict = final_inference(detection_dict)

    return output_dict

  else:
    
    img = cv2.imread(img_path)

    transformed_image = screen_extraction(img_path, model_unet, model_reg, preprocessing_unet, mode)

    detection_dict = number_detection(transformed_image, mode)

    return detection_dict

# Mapping of short forms to full forms
vital_full_forms = {
    'HR': 'Heart rate(HR)',
    'RR': 'Respiratory rate(RR)',
    'SPO2': 'Oxygen saturation(SPO2)',
    'SBP': 'Systolic blood pressure(SBP)',
    'DBP': 'Diastolic blood pressure(DBP)',
    'MAP': 'Mean arterial pressure(MAP)'
}

# Format the result dictionary into a readable paragraph
def format_vitals(result_dict):
    output = ""
    for key, value in result_dict.items():
        full_form = vital_full_forms.get(key, key)
        if key == 'SPO2':
            output += f"{full_form}: {value} percent<br>"
        elif key in ['SBP', 'DBP', 'MAP']:
            output += f"{full_form}: {value} millimeters of mercury<br>"
        elif key == 'HR':
            output += f"{full_form}: {value} beats per minute<br>"
        elif key == 'RR':
            output += f"{full_form}: {value} breaths per minute<br>"
        else:
            output += f"{full_form}: {value}<br>"
    return output

def main():
    st.set_page_config(page_title="Lifeline", page_icon="🩺", layout="centered")

    # --- Custom Styling ---
    st.markdown("""
        <style>
            body, .main, .block-container {
                background-color: #121212;
                color: #ffffff;
            }

            .title-container {
                background: linear-gradient(135deg, #00bcd4, #1de9b6);
                padding: 1rem 2rem;
                border-radius: 12px;
                margin-bottom: 30px;
                text-align: center;
            }

            .title-container h1 {
                color: #ffffff;
                font-size: 2.5rem;
                font-weight: bold;
                margin: 0;
            }

            .card {
                background-color: #1f1f1f;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            }

            .mode-card {
                background-color: #232931;
                padding: 1.2rem;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                margin-top: 1.5rem;
                margin-bottom: 1.5rem;
                border-left: 5px solid #00bfa5;
            }

            .result-box {
                margin-top: 2rem;
                padding: 1rem;
                background-color: #263238;
                border-left: 5px solid #03dac6;
                border-radius: 8px;
                font-size: 1.1rem;
                color: #ffffff;
            }

            .stButton>button {
                background-color: #00bfa5;
                color: white;
                font-weight: 600;
                border-radius: 5px;
                padding: 0.5rem 1.5rem;
                border: none;
                font-size: 1rem;
            }

            .stButton>button:hover {
                background-color: #1de9b6;
                color: black;
            }

            .upload-label {
                font-weight: 500;
                font-size: 1.1rem;
                margin-bottom: 0.5rem;
                color: #cfd8dc;
            }

            .stFileUploader {
                background-color: #292929;
                border: 1px solid #444;
                border-radius: 10px;
                padding: 1rem;
            }

            hr {
                border: 1px solid #333;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.markdown('<div class="title-container"><h1>🩺 Lifeline</h1></div>', unsafe_allow_html=True)

    # --- Upload Card Container ---
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="upload-label">📤 Upload a patient monitor image</div>', unsafe_allow_html=True)

        image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

        if image_file is not None:
            our_image = Image.open(image_file).convert("RGB")
            st.markdown("### 📷 Preview")
            st.image(our_image, use_column_width=True)

            # --- Mode Selection Card ---
            st.markdown("""
                <div class="mode-card">
                    <b>Processing Mode:</b>
            """, unsafe_allow_html=True)
            mode = st.selectbox(
                "",
                options=["accurate", "fast"],
                index=0,
                help="Choose 'accurate' for best results (slower), or 'fast' for quicker, less precise extraction."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_file.getbuffer())
                tmp_path = tmp_file.name

            if st.button("🧠 Extract Vitals"):
                start_time = time.time()
                result = vital_extraction(tmp_path, mode=mode)
                end_time = time.time()
                inference_time = round(end_time - start_time, 2)
                formatted_vitals = format_vitals(result)
                st.markdown(f"""
                    <div class="result-box">
                        <b>✅ Extracted Vitals:</b><br>
                        {formatted_vitals}<br><br>
                        <b>⏱️ Inference Time:</b> {inference_time} seconds
                    </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close card


if __name__ == '__main__':
    main()
