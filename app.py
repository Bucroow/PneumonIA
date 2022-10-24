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
num_classes=2
model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(128,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(num_classes, activation='softmax')
        ])

model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])



#interface

st.write('HIET Samuel, POGGI Alexandre, BRUMELOT Noé - BIOINF _ 2022/2023')
st.title('DataCamp')

with st.sidebar:
    st.write('Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia. Pneumonia can range in seriousness from mild to life-threatening. It is most serious for infants and young children, people older than age 65, and people with health problems or weakened immune systems.\n\n ____________________\n\nThe objective of this app is to detect if a patient as a pneumonia thanks to a picture of his lungs.')



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
    print(classes_x[0])
    st.image(image, width=500)
    if classes_x[0] == 0:
        st.balloons()
        st.write("Prediction : Healthy")
    elif classes_x[0] == 1 :
        st.snow()
        st.write("Prediction : Pneumonia")


#download file

uploaded_file = st.file_uploader("Select your Chest X-ray file you want to analyse.")

#treatment of the datas

if uploaded_file is not None:

    modelPredict(uploaded_file)

