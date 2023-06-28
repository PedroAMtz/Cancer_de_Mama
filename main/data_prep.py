import numpy as np
import pandas as pd
import tensorflow as tf
import os
from config import image_path, csv_data


data = pd.read_csv(csv_data)

# Auxiliary function for cnoncatenating the image path
def concat_img_path( patient_id, image_id, img_path = image_path):
    image_path = img_path + str(patient_id) + "_" + str(image_id) + ".png"
    return image_path

# This function will clean the data in order to contain only image path and label
def clean_data(data):
    new_data = data[['cancer']].copy()
    new_data.rename(columns={'cancer': "label"}, inplace=True)
    images_path = []
    for i in range(len(data)):
        image = concat_img_path(data["patient_id"][i], data["image_id"][i])
        images_path.append(image)
    
    new_data['images_path'] = images_path
    new_data = new_data[['images_path', 'label']]
    return new_data 

#Split data
# Process for splitting the data
def split_data(data):
    np.random.seed(10)
    rnd = np.random.rand(len(data))
    train = data[ rnd < 0.8  ]
    test = data[ (rnd >= 0.8)]


    #if os.path.exists('train_data.csv') & os.path.exists('test_data.csv') == False:
    train.to_csv('train_data.csv', header=False, index=False)
    test.to_csv('test_data.csv', header=False, index=False)
    print('Created data files')
    #print('Files already exist->Splitting...')
    print(len(data), len(train), len(test))    

# Use tensorflow to read and decode data

def read_and_decode(filename):
  img = tf.io.read_file(filename)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return img

def decode_csv(csv_row):
  record_defaults = ["filepaths", "labels"]
  filename, label_string = tf.io.decode_csv(csv_row, record_defaults)
  img = read_and_decode(filename)
  return img, label_string

def _decode_csv(csv_row):
    record_defaults = ["path", "class"]
    try:
        filename, label_string = tf.io.decode_csv(csv_row, record_defaults)
        img = read_and_decode(filename)
        label = tf.argmax(tf.math.equal(["0","1"], label_string))
    except:
        print('File corrupted')
    return img, label


def generate_data(train_filename, test_filename):
    train_dataset = (tf.data.TextLineDataset(
        train_filename).
        map(_decode_csv)).batch(32)
    test_dataset = (tf.data.TextLineDataset(
        test_filename).
        map(_decode_csv)).batch(32)
    
    return train_dataset, test_dataset

def sanity_check(dataset):
    for img, label in dataset.take(5):
        avg = tf.math.reduce_mean(img, axis=[0, 1]) # average pixel in the image
        print(label, avg)

train, test = generate_data('train_data.csv', 'test_data.csv')

if __name__ == "__main__":
    print('Starting data preparation...')
    data_to_split = clean_data(data)
    split_data(data_to_split)
    print('Now reading files on main...')
    train, test = generate_data('train_data.csv', 'test_data.csv')
    print(train, test)
    print('Data generated...')
    print('Data preparation ran succesfully!')



