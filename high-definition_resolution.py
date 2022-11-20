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

batch_size = 4

image_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.15) # train_generator를 생성
mask_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.15) # train_generator를 생성

train_hiresimage_generator = image_datagen.flow_from_dataframe(
        data, 
        x_col='high_res', # high resolution images의 경로를 저장한 열
        target_size=(800, 1200), # high resolution images의 크기
        class_mode = None, # high resolution images의 label이 없음
        batch_size = batch_size,
        seed=42,
        subset='training')

train_lowresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='low_res', # low resolution images의 경로를 저장한 열
        target_size=(800, 1200), # low resolution images의 크기
        class_mode = None, # low resolution images의 label이 없음
        batch_size = batch_size,
        seed=42,
        subset='training')

val_hiresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='high_res', # high resolution images의 경로를 저장한 열
        target_size=(800, 1200), # high resolution images의 크기
        class_mode = None, # high resolution images의 label이 없음
        batch_size = batch_size,
        seed=42,
        subset='validation')

val_lowresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='low_res', # low resolution images의 경로를 저장한 열
        target_size=(800, 1200), # low resolution images의 크기
        class_mode = None, # low resolution images의 label이 없음
        batch_size = batch_size,
        seed=42,
        subset='validation')

train_generator = zip(train_lowresimage_generator, train_hiresimage_generator) # train_generator를 생성
val_generator = zip(val_lowresimage_generator, val_hiresimage_generator) # train_generator를 생성

def imageGenerator(train_generator):
    for (low_res, hi_res) in train_generator: # train_generator에서 low resolution images와 high resolution images를 가져옴
            yield (low_res, hi_res) # low resolution images와 high resolution images를 반환

n = 0
for i,m in train_generator:
    img,out = i,m # low resolution images와 high resolution images를 가져옴

    if n < 5:
        fig, axs = plt.subplots(1 , 2, figsize=(20,5)) # 1행 2열의 그래프를 생성
        axs[0].imshow(img[0]) # low resolution images를 그래프에 출력
        axs[0].set_title('Low Resolution Image') # 그래프의 제목을 설정
        axs[1].imshow(out[0]) # high resolution images를 그래프에 출력
        axs[1].set_title('High Resolution Image') # 그래프의 제목을 설정
        plt.show()
        n+=1
    else:
        break

input_img = Input(shape=(800, 1200, 3))

l1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l1)
l3 = MaxPooling2D(padding='same')(l2)
l3 = Dropout(0.3)(l3)
l4 = Conv2D(128, (3, 3),  padding='same', activation='relu')(l3)
l5 = Conv2D(128, (3, 3), padding='same', activation='relu')(l4)
l6 = MaxPooling2D(padding='same')(l5)
l7 = Conv2D(256, (3, 3), padding='same', activation='relu')(l6)

l8 = UpSampling2D()(l7)

l9 = Conv2D(128, (3, 3), padding='same', activation='relu')(l8)
l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)

l11 = add([l5, l10])
l12 = UpSampling2D()(l11)
l13 = Conv2D(64, (3, 3), padding='same', activation='relu')(l12)
l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)

l15 = add([l14, l2])

decoded = Conv2D(3, (3, 3), padding='same', activation='relu')(l15)

autoencoder = Model(input_img, decoded) # autoencoder 모델을 생성
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) # 모델을 컴파일

autoencoder.summary() # autoencoder 모델의 요약을 출력


train_samples = train_hiresimage_generator.samples # train 데이터의 개수