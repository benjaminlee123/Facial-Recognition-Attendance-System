import sys
import numpy as np
import cv2
import os
from keras.models import model_from_json
from keras.preprocessing import image
import dlib


sys.path.append('/usr/local/lib/python2.7/site-packages')


#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# To use a video file as input 

# cap = cv2.VideoCapture('filename.mp4')
detector = dlib.get_frontal_face_detector()
p = "shape_predictor_68_face_landmarks.dat"
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3



while True:
    # Read the frame
    ret_val, img = cam.read()
    # Convert to grayscale
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect the faces
    dets = detector(rgb_image)
    # Draw the rectangle around each face
    for det in dets:
        cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        cv2.putText(img,"FaceDetected",(det.left(),det.top()+det.bottom()-200),cv2.FONT_HERSHEY_SIMPLEX,1,100)
        
        
    # Display
    cv2.imshow('img',img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cam.release()