import cv2
from random import randrange

#load a pre-trained data from (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#read an image with one face
#image = cv2.imread('leo.jpg')

#read an image with more than one face
image = cv2.imread('faces3.jpg')

#resize image
scale_factor = 0.1  # 50% of the original size
width = int(image.shape[1] * scale_factor)
height = int(image.shape[0] * scale_factor)
resized_image = cv2.resize(image, (width, height))

#Must convert to grayscale
#Grayscale Images only have one channel, representing varying shades of gray. This reduces the computational complexity ... ask chatgpt fro more
gray_scaled_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_scaled_image)

#draw rectangles
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)) , 5) 

#show the image
cv2.imshow('This is leonardo de caprio !', resized_image)
cv2.waitKey()