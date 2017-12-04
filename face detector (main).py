import numpy as np

import cv2 as cv

import os

import emotion_recognizer as er


NNET_FOLDER = os.path.join("models", "face_detection")#place the adress to caffe file
CONFIDENCE_THRESHOLD = 0.2


net=cv.dnn.readNetFromCaffe(os.path.join(NNET_FOLDER, "face_detector_config.prototxt"), os.path.join(NNET_FOLDER, "face_detector_weights.caffemodel"))


def detectFaces(frame):
    
	blob=cv.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123),False, False)
    
        net.setInput(blob)
    
	out = net.forward()
    

        N = out.shape[2]
    
	bboxes = []
    
	for i in range (N):
        
		vec = out[0, 0 , i]
        
		#print vec.shape
        
		confidence = vec[2]
        
		left = vec[3]
        
		top = vec[4]
        
		right = vec [5]
        
		bottom = vec[6]
        
		#print confidence, left, right, bottom
        
		row = frame.shape[0]
        
		column = frame.shape[1]
        
		left = left*column
        
		left = int(left)
        
		right = right*column
        
		right = int(right)
       
		top = top*row
        
		top = int(top)
        
		bottom = bottom*row
        
		bottom = int(bottom)
        
		if confidence > CONFIDENCE_THRESHOLD:
            
			cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 255))
            
		bboxes.append([left, top, right, bottom])
    
	return bboxes

if
cap = cv.VideoCapture(0)  # Open camera stream.
    
while cv.waitKey(1) < 0:
        # Read frame from camera
        
has_frame, frame = cap.read()
        
if not has_frame:
            
	break
        
detectFaces(frame)
        
print(er.getEmotion(frame))
        
cv.imshow('Frame', frame)  # Display output
