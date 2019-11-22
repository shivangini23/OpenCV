import numpy as numpy
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


img = cv2.imread('C:\\Users\\shivangini priya\\Desktop\\my_jupyter\\OpenCV\\image.jpg')
scale_factor = 1.3

#while True:
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,scale_factor,5)
    
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_color = img[y:y+h,x:x+w]
    roi_gray = gray[y:y+h,x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    smile = smile_cascade.detectMultiScale(roi_gray)
    for (sx,sy,sw,sh) in smile:
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
cv2.imshow("Face & Eyes",img)
#k = cv2.waitKey(1) & 0xff
#if k==ord('q'):
   # break

cv2.waitKey(0)
cv2.destroyAllWindows()