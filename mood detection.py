# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:08:59 2020

@author: HP
"""

import cv2
import dlib


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(r"D:\Machine Learning Bootcamp\Projects\Face landmark\shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        # print(landmarks.parts())
        lip_up = landmarks.parts()[62].y
        lip_down = landmarks.parts()[66].y

        if lip_down - lip_up > 5:
            
            print("open")
        else:
            print("close")


        # print(nose.x, nose.y)


    # print(faces)

    if ret:
        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
#You can also create games using these concept