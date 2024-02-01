import sys
import numpy as np
import cv2
import sqlite3
import dlib



sys.path.append('/usr/local/lib/python2.7/site-packages')



# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3


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




Id = input("enter users id :")
Name = input("enter the name:")



sampleNum = 0

insertOrUpdate(Id,Name)

while True:
    # Read the frame
    ret_val, img = cam.read()
    # Convert to grayscale
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    dets = detector(rgb_image)
    # Draw the rectangle around each face
    for det in dets:
        sampleNum=sampleNum+1

        cv2.imwrite("dataSet/Users."+str(Id)+"."+str(sampleNum)+".jpg",rgb_image [det.top():det.top()+det.bottom(),det.left():det.left()+det.right()])
        cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()),(255,0,0),2)
        cv2.putText(img,"FaceDetected",(det.left(),det.top()+det.bottom()-200),cv2.FONT_HERSHEY_SIMPLEX,1,100)
        cv2.waitKey(300)
        
    # Display
    cv2.imshow("Face", img)
    # Stop if escape key is pressed
    cv2.waitKey(1)
# Release the VideoCapture object
    if(sampleNum>20):
        cam.release()
        break
cam.release()