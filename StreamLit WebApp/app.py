import streamlit as st
import numpy as np
import tensorflow_hub as hub
import os, pathlib
import io
import warnings
import cv2
from six import BytesIO
from PIL import Image
import scipy.misc
import tensorflow as tf
from PIL import ImageDraw, ImageOps, Image, ImageFont
from six.moves.urllib.request import urlopen
from tensorflow.keras.preprocessing import image
import math

# Ignore all the warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

# Warnings ignore 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()



# Actual Interface 
st.markdown("<center><h1>🚦🚗Traffic Monitoring System🚗🚦</h1></center>", unsafe_allow_html=True)
st.markdown("<center><h3>Welcome to our traffic monitoring system</h3></center><br><br>", unsafe_allow_html=True)

left,right = st.columns(2)


# ------------------- Get & Save Image into desired folder -------------
img = right.file_uploader("Upload the image at a traffic signal", type=['png', 'jpg','jpeg'])
if img:
    right.image(img, caption="Traffic signal image")
else:
    right.success("Please Upload an Image")

if(img):
    data = img.read()
    img = Image.open(io.BytesIO(data))
    x = image.img_to_array(img)
    
    # Save image into folder
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    cv2.imwrite("D:\\Om\Fast Track Semester\\TARP\Project\\Images for testing\\User.jpg",x)

# -----------------------------------------------------------------------

# ----------------- Traffic threshold calculation module ----------------
left.markdown("<center><h3>🖩 Traffic Calculator 🖩: <h3></center>", unsafe_allow_html=True)
with left.expander("Information about lanes: "):
    st.markdown("<br>1. Standard Road Lane width - 2.7 to 4.6 m (9 to 15 feet). <br>2. India - 3.75m wide<br>3. India 2 lane road - 7m to 7.5m wide.<br>4. United States of America (USA) - 12 feet (3.7m) wide.<br>5. European countries - 2.5m to 3.65m wide.", unsafe_allow_html=True)

left.markdown("<br><br>", unsafe_allow_html=True)
lanes = left.number_input("Number of lanes")
width = left.number_input("Width of the lane")
length = left.number_input("Metres considered")
area = lanes*width*length
cararea = 1;
if(width!=0): cararea = 4.8768*width

maxcars = area/cararea

# left.markdown("<br><br>", unsafe_allow_html=True)
st.write("Total area considered ",length," :", lanes*width*length,"sq.m")
st.write("Maximum cars pre-traffic condition: ",math.floor(maxcars))
# -------------------------------------------------------------------------





