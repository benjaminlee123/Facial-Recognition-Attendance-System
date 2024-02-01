import tkinter as tk 
from tkinter import Message ,Text
import cv2,os
import csv
import numpy as np
from PIL import Image, ImageTk
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import sys
import sqlite3
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import dlib
import imutils
import face_recognition



window = tk.Tk()
window.geometry('500x500')
window.title("Student Registration Form")



def TakeImages():
    sys.path.append('/usr/local/lib/python2.7/site-packages')

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    Id=(entry_2.get())
    Name=(entry_1.get())

    def insertOrUpdate(Id,Name):


        conn=sqlite3.connect("FaceData.db")

        cursor = conn.cursor()
        # cmd= INSERT INTO StudentsFaces(ID,Name)Values((Id),(Name))
        cmd = "SELECT * FROM StudentsData WHERE ID = " + Id
        cursor = conn.execute(cmd)
        isRecordExist = 0
        for row in cursor:       
            isRecordExist = 1
        if isRecordExist == 1:
            conn.execute("UPDATE StudentsData SET Name = ? WHERE ID = ?",(Name,Id))
        else:
            conn.execute("INSERT INTO StudentsData(ID,Name)Values(?,?)",(Id,Name))
        conn.commit()
        conn.close()

    sampleNum = 0

    insertOrUpdate(Id,Name)

    while True:
        # Read the frame
        ret, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            sampleNum=sampleNum+1

            cv2.imwrite("dataSet/Users."+str(Id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x,y), (x+w, y+h),(255,0,0),2)
            cv2.putText(img,"Face Detected",(x,y+h+30),cv2.FONT_HERSHEY_SIMPLEX,1,255)
            cv2.waitKey(300)
            
        # Display
        cv2.imshow("Face", img)
        # Stop if escape key is pressed
        cv2.waitKey(1)
    # Release the VideoCapture object
        if(sampleNum>20):
            cap.release()
            break   

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    path='dataSet'

    def getImagesWithID(path):

        imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
        faces = []
        IDs = []
        for imagePath in imagePaths:
            faceImg=Image.open(imagePath).convert('L');
            faceNp=np.array(faceImg,'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
        
            IDs.append(ID)
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
        return IDs, faces

    Ids,faces=getImagesWithID(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.save('recognizer/trainningData.yml')


def FacialRecognition():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rec=cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer\\trainningData.yml")
    #load model
    model = model_from_json(open("fer.json", "r").read())
    #load weights
    model.load_weights('fer.h5')

    detector = dlib.get_frontal_face_detector()
    cam = cv2.VideoCapture(0)
    color_green = (0,255,0)
    line_width = 2

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


    cam.set(3, 480) #set width
    cam.set(4, 640) #set height

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
            
            ret_val, img = cam.read()
                
            rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            dets = detector(rgb_image)


            for det in dets:
                cv2.rectangle(img,(det.left(),det.top()),(det.right(), det.bottom()),color_green,line_width)
                id,confidence=rec.predict(rgb_image[det.top():det.top()+det.bottom(),det.left():det.left()+det.right()])

                # Get Face 
                face_img = img[det.top():det.top()+det.bottom(),det.bottom():det.bottom()+det.right()].copy()
                blob = cv2.dnn.blobFromImage(face_img,1,(227, 227),MODEL_MEAN_VALUES,swapRB=False)

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
                cv2.putText(img,overlay_text, (det.left(),det.top()+det.bottom()-200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                roi_gray=rgb_image[det.top():det.top()+det.right(),det.left():det.left()+det.bottom()]
                roi_gray=cv2.resize(roi_gray,(48,48))
                img_pixels = img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

                predicted_emotion = emotions[max_index]

                cv2.putText(img, predicted_emotion,(int(det.left()), int(det.top())), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                profile=getProfile(id)

                if(profile !=None and confidence < 100):                    
                    cv2.putText(img,str(id),(det.left(),det.top()+det.bottom()-175),cv2.FONT_HERSHEY_SIMPLEX,1,255)
                    cv2.putText(img,str(confidence),(det.left(),det.top()+det.bottom()-225),cv2.FONT_HERSHEY_SIMPLEX,1,255)
                    confidence = "  {0}%".format(round(100 - confidence))                                
                else:                   
                    cv2.putText(img,"Unknown",(det.left(),det.top()+det.bottom()-175),cv2.FONT_HERSHEY_SIMPLEX,1,255)
                    confidence = "  {0}%".format(round(100 - confidence))
                 

                    

            cv2.imshow('frame',img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;

    if __name__ == "__main__":
        age_net,gender_net = initialize_caffe_models()

        read_from_camera(age_net,gender_net)








label1 = tk.Label(window,text="Student Registration Form",relief="solid",width=30,font=("arial",19,"bold"))
label1.place(x=80,y=150)


label2 = tk.Label(window,text="Name:",width=20,font=("arial",10,"bold"))
label2.place(x=80,y=250)

entry_1 = tk.Entry(window,width=20,font=('times', 15, ' bold ')  )
entry_1.place(x=300,y=250)

label3 = tk.Label(window,text="ID:",width=20,font=("arial",10,"bold"))
label3.place(x=68,y=300)

entry_2 = tk.Entry(window,width=20  ,font=('times', 15, ' bold ')  )
entry_2.place(x=300,y=300)




takeImg = tk.Button(window, text="Step 1. TakeImages", command=TakeImages ,fg="brown"  ,bg="white"  ,width=28  ,height=2,font=('times', 10, ' bold '))
takeImg.place(x=300, y=350)
trainImg = tk.Button(window, text="Step 2. TrainImages", command=TrainImages  ,fg="brown"  ,bg="white"  ,width=28  ,height=2 ,font=('times', 10, ' bold '))
trainImg.place(x=300, y=400)
trackImg = tk.Button(window, text="Step 3. FacialRecognition", command=FacialRecognition ,fg="brown"  ,bg="white"  ,width=28  ,height=2 ,font=('times', 10, ' bold '))
trackImg.place(x=300, y=450)

 
window.mainloop()