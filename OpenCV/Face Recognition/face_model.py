import os
from PIL import Image
import pickle
import cv2
import numpy as np

#Face Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Getting the path of the directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#Getting the image directory
image_dir = os.path.join(BASE_DIR,"images")

#Defining the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id=0
label_ids={} # For encoding the labels to numerical values
x_train = []
y_labels = []
scale_factor=1.5
min_neighbors=5
for root,dirs,files in os.walk(image_dir):
	for file in files:
		if file.endswith("jpg"):
			path = os.path.join(root,file)
			label = os.path.basename(root).replace(" ","-").lower() #get the labels from the directories
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_label = label_ids[label]
			pil_image = Image.open(path).convert("L") #convert image to grayscale
			pil_image = pil_image.resize((600,600),Image.ANTIALIAS) #Resize all images to the same size
			image_array = np.array(pil_image,"uint8") #Converting image to a numpy array
			faces = face_cascade.detectMultiScale(image_array,scale_factor,min_neighbors)
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_labels.append(id_label)

#Making the pickle file for the labels
with open("labels.pickle","wb") as f:
	pickle.dump(label_ids,f)

#Training the recognizer
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")

