import numpy as np 
import pandas as pd 
import os
import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

base_directory = '../dataset/Image Super Resolution - Unsplash'
hires_folder = os.path.join(base_directory, 'high res') # high resolution images를 저장한 폴더
lowres_folder = os.path.join(base_directory, 'low res') # low resolution images를 저장한 폴더

batch_size = 4

image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
mask_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


train_hiresimage_generator = image_datagen.flow_from_dataframe