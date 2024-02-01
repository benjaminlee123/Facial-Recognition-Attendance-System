import sys
import cv2
import imutils
import time
import numpy as np
import sqlite3
import os
from keras.models import model_from_json
from keras.preprocessing import image
import argparse
import dlib


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainningData.yml")
#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


sys.path.append('/usr/local/lib/python2.7/site-packages')


id = 0




def getProfile(id):

    conn=sqlite3.connect("FaceData.db")
    cmd="SELECT Name FROM StudentsData WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


cap.set(3, 480) #set width
cap.set(4, 640) #set height

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def initialize_caffe_models():
	
	
	age_net = cv2.dnn.readNetFromCaffe(
		'deploy_age.prototxt', 
		'age_net.caffemodel')

	gender_net = cv2.dnn.readNetFromCaffe(
		'deploy_gender.prototxt', 
		'gender_net.caffemodel')

	return(age_net,gender_net)

def read_from_camera(age_net, gender_net):
	font = cv2.FONT_HERSHEY_SIMPLEX

	while True:
		
		ret, img= cap.read()
			
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(gray,1.3,5)

		if(len(faces)>0):
			print("Found {} faces".format(str(len(faces))))

		for (x, y, w, h )in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
			id,confidence=rec.predict(gray[y:y+h,x:x+w])

			# Get Face 
			face_img = img[y:y+h, h:h+w].copy()
			blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

			#Predict Gender
			gender_net.setInput(blob)
			gender_preds = gender_net.forward()
			gender = gender_list[gender_preds[0].argmax()]
			print("Gender : " + gender)

			#Predict Age
			age_net.setInput(blob)
			age_preds = age_net.forward()
			age = age_list[age_preds[0].argmax()]
			print("Age Range: " + age)

			overlay_text = "%s %s" % (gender, age)
			cv2.putText(img, overlay_text, (x,y+h+60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

			roi_gray=gray[y:y+w,x:x+h]
			roi_gray=cv2.resize(roi_gray,(48,48))
			img_pixels = image.img_to_array(roi_gray)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255

			predictions = model.predict(img_pixels)

			max_index = np.argmax(predictions[0])

			emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

			predicted_emotion = emotions[max_index]

			cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			
			profile=getProfile(id)

			if(profile !=None and confidence < 100):
				cv2.putText(img,str(id),(x,y+h+30),cv2.FONT_HERSHEY_SIMPLEX,1,255)
				cv2.putText(img,str(confidence),(x,y+h+55),cv2.FONT_HERSHEY_SIMPLEX,1,255)
				confidence = "  {0}%".format(round(100 - confidence))
			else:
				cv2.putText(img,"Unknown",(x,y+h+30),cv2.FONT_HERSHEY_SIMPLEX,1,255)
				
			

		cv2.imshow('frame',img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	age_net,gender_net = initialize_caffe_models()

	read_from_camera(age_net,gender_net)