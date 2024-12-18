import cv2
from random import randrange

#load a pre-trained data from (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Using pc webcacm
#webcam = cv2.VideoCapture(0)

#from video
webcam = cv2.VideoCapture('walking.mp4')



while True:

    suc_frame_read, frame = webcam.read()

    #Must convert to grayscale
    #Grayscale Images only have one channel, representing varying shades of gray. This reduces the computational complexity ... ask chatgpt fro more
    gray_scaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_image)
    
    #draw rectangles
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)) , 5) 
        
    #show the image
    cv2.imshow('From webcam !', frame)
    key = cv2.waitKey(1) #one for continues capture

    if key==81 or key==113:
        break

webcam.release()