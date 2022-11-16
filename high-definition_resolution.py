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

data = pd.read_csv("../input/image-super-resolution-from-unsplash/Image Super Resolution - Unsplash/image_data.csv") # image_data.csv 파일을 읽어옴
data['low_res'] = data['low_res'].apply(lambda x: os.path.join(lowres_folder,x)) # low resolution images의 경로를 저장 
data['high_res'] = data['high_res'].apply(lambda x: os.path.join(hires_folder,x)) # high resolution images의 경로를 저장

atch_size = 4