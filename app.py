import cv2
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers

# imports
from tensorflow import keras
from keras.models import load_model
from keras.optimizers import SGD

# constraining the image size 
img_width, img_height = 200, 200

#functions
# load and compile model previously made with the training.ipynb
model = load_model('chest_images_vgg19.h5')
model.compile(loss = "binary_crossentropy", 
              optimizer = SGD(lr=0.001, momentum=0.9), 
              metrics=["accuracy"])


def modelPredict(image):
    image = Image.open(image) 
    image_to_predict = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_to_predict = np.expand_dims(cv2.resize(image_to_predict,(200,200)), axis=0) 
    predict_x=model.predict(img_to_predict)
    classes_x=np.argmax(predict_x,axis=1)
    
    if classes_x[0] == 0:
        st.header("Prediction : Healthy")
    elif classes_x[0] == 1 :
        st.header("Prediction : Pneumonia")

    st.image(image, width=500)
    
#interface

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.write(' ')
with col3:
    st.image('LOGO_EFREI.png')

st.title('Pneumonnia Detection Application')

st.write('This tool will help you to discover if you have a pneumonia. Thanks to one of your chest X-ray, our system will make a prediction (92 percent of accuracy).')
st.write(' ')

with st.sidebar:
    st.write('Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia. Pneumonia can range in seriousness from mild to life-threatening. It is most serious for infants and young children, people older than age 65, and people with health problems or weakened immune systems.\n\n ____________________\n\nThe objective of this app is to detect if a patient as a pneumonia thanks to a picture of his lungs.')



#download file

uploaded_file = st.file_uploader("Select your Chest X-ray file you want to analyse.")

#treatment of the datas

if uploaded_file is not None:

    modelPredict(uploaded_file)

st.write(' ')
st.write(' _______________________________________________________________________________ ')
st.markdown('Web application developped by **HIET Samuel**, **POGGI Alexandre** & **BRUMELOT Noé**')
st.markdown('M1.BIOINF, 2022-2023')
