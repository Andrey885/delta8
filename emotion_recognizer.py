import cv2

import os

import numpy as np


NNET_FOLDER = os.path.join("models", "emotion_recognition")
 #place the adress to file with caffe


nnet = cv2.dnn.readNetFromTensorflow(os.path.join(NNET_FOLDER, 'emotion_recognition.pb'))

emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]


def getEmotion(img):
    
	img = cv2.resize(img, (48, 48))
    
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
	img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
	blob = cv2.dnn.blobFromImage(img)
    
	nnet.setInput(blob)
    
	return emotions[ np.argmax(nnet.forward()[0]) ]
