import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def apply_HE_filter(image, plot_filter=True):
    try:
        img = cv2.equalizeHist(image)
        print('Filter applied successfully!')
    except:
        print('Filter applied unsuccessfully')
    
    if plot_filter == True:
        plt.imshow(img, cmap='gray')
        plt.title('HE filter')
        plt.axis('off')
    
    return img

def apply_CLAHE_filter(image, clipLimit=2.0, tileGridSize=(8,8),
                        plot_filter=True):
    
    try: 
        clahe = cv2.createCLAHE(clipLimit=clipLimit,
                            tileGridSize=tileGridSize)
        img = clahe.apply(image)
        print('Filter applied successfully!')
    except:
        print('Filter applied unsuccessfully')

    if plot_filter == True:
        plt.imshow(img, cmap='gray')
        plt.title('CLAHE filter')
        plt.axis('off')

    return img

def apply_sigmoidal_filter(image, window_center=0.5, slope=5.0,
                            plot_filter=True):
    
    try: 
        height, width = image.shape[:2]
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        window = 1 / (1 + np.exp(-slope * (X - window_center)))
        windowed_image = image * window
        print('Filter applied successfully!')
    except:
        print('Filter applied unsuccessfully')

    if plot_filter == True:
        plt.imshow(windowed_image, cmap='gray')
        plt.title('Sigmoidal filter')
        plt.axis('off')

    return windowed_image


