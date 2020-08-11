# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:29:48 2020

@author: HP
"""

import cv2
import dlib  #dlib is used for facial landmarking

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(r"D:\Machine Learning Bootcamp\Projects\Face landmark\shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    #print(faces) #Provide dlib rectangle of faces and each rectangle will give value of face
    for face in faces:
        landmarks = predictor(gray, face)
        # print(landmarks.parts()) #values of points that we display on our face    
        nose = landmarks.parts()[27] #28 points to the value of nose
      #  print(nose.x, nose.y)
        for point in landmarks.parts():
          #  cv2.circle(frame, (nose.x, nose.y), 2, (255, 0, 0), 3) # it will point one circle point on nose
            cv2.circle(frame, (point.x, point.y), 2, (255, 0, 0), 3) # point all the points in faces


    if ret:
        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()