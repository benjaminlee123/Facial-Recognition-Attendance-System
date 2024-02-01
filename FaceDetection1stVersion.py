import sys
import numpy as np
import cv2
import os
import sys
import cv2
from tensorflow import keras
from keras.preprocessing import image

sys.path.append('/usr/local/lib/python2.7/site-packages')

#load model
model = keras.models.model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        
        cv2.rectangle(img,(x, y),(x+w,y+h),(255,0,0), 2)
        cv2.putText(img,"Face Detected",(x,y+h+30),cv2.FONT_HERSHEY_SIMPLEX,1,255)
        
    # Display
    cv2.imshow('img',img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()