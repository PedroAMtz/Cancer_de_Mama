import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 

data = pd.read_csv("test_data.csv",
 names=["Filepaths", "Labels"],
 header=None)
print(data["Filepaths"][0].lstrip("C:/Users/pedro/OneDrive/Escritorio/Proyecto_IA_SS/Cancer_de_Mama_IA_SS_2023/data/raw_images/"))

def clahe_filter(filename, clipLimit=2.0, tileGridSize=(8,8)):
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # or just put 0 to refer to grayscale
	clahe = cv2.createCLAHE(clipLimit=clipLimit,
                            tileGridSize=tileGridSize)
	img = clahe.apply(img)
	return img

test_img = clahe_filter(data["Filepaths"][5])

plt.imshow(test_img, cmap='gray')
plt.title('CLAHE filter')
plt.axis('off')
plt.show()
