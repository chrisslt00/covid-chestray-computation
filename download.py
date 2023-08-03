import requests
import os
import cv2

url = "https://www.kaggle.com/datasets/sid321axn/covid-cxr-image-dataset-research/download?datasetVersionNumber=1"
r = requests.get(url)

r.headers

print(r.headers)

data=[]
labels=[]
Uninfected=os.listdir("~/normal/")
for a in Uninfected:
	# extract the class label from the filename
	

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread("/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/normal/"+a)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(gray_image, (224, 224))

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(0)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Input
import matplotlib.pyplot as plt
import keras
from keras.utils.np_utils import to_categorical
import cv2
import os

data=[]
labels=[]
Uninfected=os.listdir("/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/normal/")
for a in Uninfected:
	# extract the class label from the filename
	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread("/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/normal/"+a)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(gray_image, (224, 224))
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(0)
Covid=os.listdir("/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/covid/")
for b in Covid:
	# extract the class label from the filename
	

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread("/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/covid/"+b)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(gray_image, (224, 224))
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(1)
Virus=os.listdir("/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/virus/")
for c in Virus:
	# extract the class label from the filename
	

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread("/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/virus/"+c)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(gray_image, (224, 224))
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(2)
data = np.array(data) / 255.0
labels = np.array(labels)
from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(data, labels,stratify=labels,
	test_size=0.20,  random_state=42)