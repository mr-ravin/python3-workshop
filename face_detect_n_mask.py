import cv2 as cv
import sys

human_filename=sys.argv[1] ## we pass filename of image containing human face as command-line-argument 1 
mask_filename=sys.argv[2]  ## we pass filename of image having mask as command-line -argument 2
                           ## i.e. python3 face_detect_n_mask.py human_image.jpg mask.jpg
human_img=cv.imread(human_filename)
mask_img=cv.imread(mask_filename)

car_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv.cvtColor(human_img, cv.COLOR_BGR2GRAY) # convert image to gray scale
faces = car_cascade.detectMultiScale(gray, 1.1, 1) # apply haar-cascade
for (x,y,w,h) in faces:                         ## for each face found in the image
  replace_roi = cv.resize(mask_img,(w,h),interpolation = cv.INTER_CUBIC)
  mask_gray= cv.cvtColor(replace_roi, cv.COLOR_BGR2GRAY)
  for i in range(h):                            # loop to apply mask over the face
    for j in range(w):
      if mask_gray[i][j]>235:
         continue
      else:
         human_img[y+i][x+j]=replace_roi[i][j]    

cv.imwrite("masked_image.jpg",human_img)

  

