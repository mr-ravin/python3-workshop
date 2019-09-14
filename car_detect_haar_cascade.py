import cv2 as cv
import sys

filename=sys.argv[1] # we pass filename as command-line-argument here i.e. python3 car.py vehicle.jpg
car_cascade = cv.CascadeClassifier('cars.xml')
img=cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert image to gray scale
cars = car_cascade.detectMultiScale(gray, 1.1, 1) # apply haar-cascade
for (x,y,w,h) in cars:                            ## for each car found in the image
  cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)  ## draw a rectangle (in RED COLOR of WIDTH 2) around the car
cv.imwrite("detected_car.jpg",img)
