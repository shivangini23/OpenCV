import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread("C:\\Users\\shivangini priya\\Pictures\\New folder\\100D3200\\picture.jpg")
scale_factor=1.5
faces = face_cascade.detectMultiScale(img,scale_factor,5)
for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
resize = cv2.resize(img,(500,500))
cv2.imshow('PICTURE',resize)
cv2.waitKey(0)
cv2.destroyAllWindows()