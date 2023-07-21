import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 

directory = r"C:/Users/pedro/OneDrive/Escritorio/Proyecto_IA_SS/Cancer_de_Mama_IA_SS_2023/data/clahe_images/"


train_data = pd.read_csv("train_data.csv",
 names=["Filepaths", "Labels"],
 header=None)

test_data = pd.read_csv("test_data.csv",
 names=["Filepaths", "Labels"],
 header=None)

def clahe_filter(filename, clipLimit=2.0, tileGridSize=(8,8)):
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # or just put 0 to refer to grayscale
	clahe = cv2.createCLAHE(clipLimit=clipLimit,
                            tileGridSize=tileGridSize)
	img = clahe.apply(img)
	return img


train_files = []
train_labels = []
for i in range(len(train_data["Filepaths"])):
	file_name = str(directory) + train_data["Filepaths"][i].lstrip("C:/Users/pedro/OneDrive/Escritorio/Proyecto_IA_SS/Cancer_de_Mama_IA_SS_2023/data/raw_images/")
	train_files.append(file_name)
	train_labels.append(train_data["Labels"][i])


test_files = []
test_labels = []
for i in range(len(test_data["Filepaths"])):
	file_name = str(directory) + test_data["Filepaths"][i].lstrip("C:/Users/pedro/OneDrive/Escritorio/Proyecto_IA_SS/Cancer_de_Mama_IA_SS_2023/data/raw_images/")
	test_files.append(file_name)
	test_labels.append(test_data["Labels"][i])

train_data_preprocessed = pd.DataFrame(list(zip(train_files, train_labels)),
	columns=["Filepaths", "Labels"])
test_data_preprocessed = pd.DataFrame(list(zip(test_files, test_labels)),
	columns=["Filepaths", "Labels"])

train_data_preprocessed.to_csv('train_data_prec.csv', header=False, index=False)

test_data_preprocessed.to_csv('test_data_prec.csv', header=False, index=False)


"""
for i in range(len(data["Filepaths"])):
	test_img = clahe_filter(data["Filepaths"][i])
	cv2.imwrite(str(directory) + data["Filepaths"][i].lstrip("C:/Users/pedro/OneDrive/Escritorio/Proyecto_IA_SS/Cancer_de_Mama_IA_SS_2023/data/raw_images/"), test_img)
	print("saving", str(i), "image")
"""


