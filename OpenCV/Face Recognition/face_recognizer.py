import cv2
import pickle
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

video = cv2.VideoCapture(0)
scale_factor=1.5
min_neighbors=5

labels = {}
with open("labels.pickle","rb") as f:
	lab = pickle.load(f)
	labels = {v:k for k,v in lab.items()}   #Reversing the dictionary

while True:
	ret,frame = video.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,scale_factor,min_neighbors)
	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h,x:x+w]
		roi_color = frame[y:y+h,x:x+w]
		id_,conf = recognizer.predict(roi_gray)
		if conf>=50:
			name = labels[id_]
			font = cv2.FONT_HERSHEY_SIMPLEX
			stroke = 2
			cv2.putText(frame,name,(x,y),font,1,(255,255,255),stroke,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

	cv2.imshow("faces",frame)
	k = cv2.waitKey(1) & 0xff
	if k==ord("q"):
		break

video.release()
cv2.destroyAllWindows()


