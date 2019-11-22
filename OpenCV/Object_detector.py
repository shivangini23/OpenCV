import cv2
import numpy as np

#Loading the model
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
#Loading the class file
classes = []
with open("coco.names" ,"r") as f:
	classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layer = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Generating different colors for each class name
colors = np.random.uniform(0,255,size=(len(classes),3))

#Loading the image
img = cv2.imread("object.jpg")
img = cv2.resize(img,(600,600))
height,width,channels = img.shape
print(height,width,channels)

#Detecting objects

blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
#for b in blob:
#	for n,img_blob in enumerate(b):
#		cv2.imshow(str(n),img_blob)

#Pass the blob into the algorithm
net.setInput(blob)
outs = net.forward(output_layer)
boxes =[]
confidences = []
class_ids = []
for out in outs:
	for detection in out:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]
		if confidence>0.5:
			center_x = int(detection[0]*width)
			center_y = int(detection[1]*height)
			w = int(detection[2]*width)
			h = int(detection[3]*height)

			x = int(center_x-w/2)
			y = int(center_y-h/2)

			boxes.append([x,y,w,h])
			confidences.append(float(confidence))
			class_ids.append(class_id)

#Using non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
object_detected = len(boxes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(object_detected):
	if i in indexes:
		x,y,w,h = boxes[i]
		label = str(classes[class_ids[i]])
		color = colors[i]
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),color,2)
		cv2.putText(img,label,(x,y+30),font,1,3,color,3)


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

