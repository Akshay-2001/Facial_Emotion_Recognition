import cv2
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
    

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(r'D:\Final Year Project\FER\haarcascade_frontalface_default.xml')

# Load pre-trained emotion detection model
emotion_model = load_model(r'D:\Final Year Project\FER\Checkpoint\Model.h5', compile=False)
optims = [
    optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    optimizers.Adam(0.001),
]
emotion_model.compile(
        loss='categorical_crossentropy',
        optimizer=optims,
        metrics=['accuracy']
    )

# Define emotions labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def detect_emotion(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Process each detected face
    for (x, y, w, h) in faces:

        # Extract Region of Interest(ROI)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = emotion_model.predict(roi)[0]
            label=emotions[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        else:
            cv2.putText(image,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    
    # Display frame with emotions
    cv2.imshow('Emotion Detector',image)
    return image

# Streamlit app
st.title("Face Emotion Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Convert the image to numpy array
    img_array = np.array(image)
    
    # Detect emotions and display result
    result_img = detect_emotion(img_array)
    
    # Display the result image
    st.image(result_img, channels="BGR", caption="Emotion Detection Result")
