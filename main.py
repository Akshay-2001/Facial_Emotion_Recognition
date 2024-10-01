from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

# Load Harr cascade for face detection
face_classifier = cv2.CascadeClassifier(r'D:\Final Year Project\FER\haarcascade_frontalface_default.xml')

# Load pre-trained facial emotion recognition model
classifier = load_model(r'D:\Final Year Project\FER\Checkpoint\Model.h5', compile=False)
optims = [
    optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    optimizers.Adam(0.001),
]
classifier.compile(
        loss='categorical_crossentropy',
        optimizer=optims,
        metrics=['accuracy']
    )
# Define list of emotions
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad', 'Surprise', 'Neutral']

# Open Camera
cap = cv2.VideoCapture(0)



while True:
    
    # Capture frame from camera
    _, frame = cap.read()
    labels = [] # Defining labels
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detect faces in the frame
    faces = face_classifier.detectMultiScale(gray)

    # Iterate over detected faces
    for (x,y,w,h) in faces:
        
        # Extract Region of Interest(ROI)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    # Display frame with emotions
    cv2.imshow('Emotion Detector',frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()