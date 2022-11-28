# Change_image_high-definition_resolution
Autoencoders를 활용한 Image Super Resolution 프로젝트

## 프로젝트 목적
화질이 낮은 사진을 높은화질로 변경해주기위한 프로젝트

## 프로젝트 배경
Resolution 기술에 대한 이해력 상승 및 모델 이해력 상승

## 연구 및 개발에 필요한 데이터 셋 소개
https://www.kaggle.com/datasets/quadeer15sh/image-super-resolution-from-unsplash

1.kaggle에서 이미지의 데이터셋을 다운로드 받아 모델을 구현하였습니다.


## 연구 및 개발에 필요한 기술 스택
### AutoEncoders
1. 오토인코더는 비지도unsupervised 방식으로 훈련된 인공 신경망으로, 먼저 데이터에 인코딩 된 표현을 학습한 다음, 학습 된 인코딩 표현에서 입력 데이터를 (가능한한 가깝게) 생성하는 것을 목표로 한다. 따라서, 오토인코더의 출력은 입력에 대한 예측이다.

![image](https://user-images.githubusercontent.com/97720878/204270831-89c21cf3-436d-4803-9912-729de383718f.png)
기본 오토인코더의 아키텍쳐

2. 인코더 (\boldsymbol{W_h}W h로 정의되는 아핀 변환affine transformation 후 스쿼싱squashing)를 거치는 입력 \boldsymbol{x}x으로 맨 아래서부터 시작한다. 이는 중간 은닉층hiddenn layer \boldsymbol{h}h을 형성한다. 이는 디코더 (또 다른 \boldsymbol{W_x}W 
x  로 정의되는 아핀 변환 다음에 또 다른 스쿼싱이 뒤따름)의 대상이 된다. 그러면 모델의 입력에 대한 예측/재건인 출력 \boldsymbol{\hat{x}} x이 생성된다. 우리는 이를 관습convention에 따라 3층 신경망으로 부른다.

  
U-Net은 적은 데이터로 충분한 학습을 하기 위해 Data Augmentation을 사용
Data Augmentation이란 원래의 데이터를 부풀려서 더 좋은 성능을 만든다는 뜻
Data Augmentation이 중요한 이유
1. Preprocessing & augmentation 진행 시 성능 상승
2. 원본에 추가되는 개념으로 성능이 떨어지지 않음
3. 쉽고 패턴이 정해져 있음

      
```Python3
def get_unet_model():
    
    inputs = tf.keras.layers.Input(shape = [128, 128, 3])
    
    #First Downsample
    f1 = tf.keras.layers.Conv2D(64, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(inputs)
    b1 = tf.keras.layers.BatchNormalization()(f1)
    f2 = tf.keras.layers.Conv2D(64, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b1)    # Used later for residual connection
    
    m3 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2)(f2)
    d4 = tf.keras.layers.Dropout(0.2)(m3)
    
    # Second Downsample
    f5 = tf.keras.layers.Conv2D(128, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(d4)
    b5 = tf.keras.layers.BatchNormalization()(f5)
    f6 = tf.keras.layers.Conv2D(128, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b5)    # Used later for residual connection
    
    m7 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2)(f6)
    d8 = tf.keras.layers.Dropout(0.2)(m7)
    
    # Third Downsample
    f9 = tf.keras.layers.Conv2D(256, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(d8)
    b9 = tf.keras.layers.BatchNormalization()(f9)
    f10 = tf.keras.layers.Conv2D(256, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b9)    # Used later for residual connection
    
    m11 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2)(f10)
    d12 = tf.keras.layers.Dropout(0.2)(m11)
    
    #Forth Downsample
    f13 = tf.keras.layers.Conv2D(512, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(d12)
    b13 = tf.keras.layers.BatchNormalization()(f13)
    f14 = tf.keras.layers.Conv2D(512, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b13)    # Used later for residual connection
    
    m15 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2)(f14)
    d16 = tf.keras.layers.Dropout(0.2)(m15)
    
    #Fifth Downsample
    f17 = tf.keras.layers.Conv2D(1024, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(d16)
    b17 = tf.keras.layers.BatchNormalization()(f17)
    f18 = tf.keras.layers.Conv2D(1024, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b17)

    
    # First Upsample
    m19 = tf.keras.layers.UpSampling2D(size = (2, 2))(f18)
    d19 = tf.keras.layers.Dropout(0.2)(m19)
    c20 = tf.keras.layers.Concatenate()([d19, f14])
    f21 = tf.keras.layers.Conv2D(512, kernel_size = (3, 3), padding = "same", strides = 1 ,activation = "relu")(c20)
    b21 = tf.keras.layers.BatchNormalization()(f21)
    f22 = tf.keras.layers.Conv2D(512, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b21)
    
    # Second Upsample
    m23 = tf.keras.layers.UpSampling2D(size = (2, 2))(f22)
    d23 = tf.keras.layers.Dropout(0.2)(m23)
    c24 = tf.keras.layers.Concatenate()([d23, f10])
    f25 = tf.keras.layers.Conv2D(256, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(c24)
    b25 = tf.keras.layers.BatchNormalization()(f25)
    f26 = tf.keras.layers.Conv2D(256, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b25)
    
    # Third Upsample
    m27 = tf.keras.layers.UpSampling2D(size = (2, 2))(f26)
    d27 = tf.keras.layers.Dropout(0.2)(m27)
    c28 = tf.keras.layers.Concatenate()([d27, f6])
    f29 = tf.keras.layers.Conv2D(128, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(c28)
    b29 = tf.keras.layers.BatchNormalization()(f29)
    f30 = tf.keras.layers.Conv2D(128, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b29)
    
    #Forth Upsample
    m31 = tf.keras.layers.UpSampling2D(size = (2, 2))(f30)
    d31 = tf.keras.layers.Dropout(0.2)(m31)
    c32 = tf.keras.layers.Concatenate()([d31, f2])
    f33 = tf.keras.layers.Conv2D(64, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(c32)
    b33 = tf.keras.layers.BatchNormalization()(f33)
    f34 = tf.keras.layers.Conv2D(64, kernel_size = (3, 3), padding = "same", strides = 1, activation = "relu")(b33)
    
    # Output Layer
    outputs = tf.keras.layers.Conv2D(num_classes, kernel_size = (3, 3), padding = "same", strides = 1, activation = "softmax")(f34)
    
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    return model

model = get_unet_model() # 모델 생성
tf.keras.utils.plot_model(model, show_shapes = True) # 모델 구조 확인
```


## 결과
![__results___23_1](https://user-images.githubusercontent.com/97720878/197787923-0ff968f8-4ca6-47e2-9581-4e31ecdb58b8.png)

## 한계점 및 해결 방안
국내 데이터를 활용하여 모델을 구축해볼 예정
다른 사람의 참고코드를 활용한것이 아닌 직접 모델을 만드는 프로젝트 진행 예정


참고 코드
https://www.kaggle.com/code/tr1gg3rtrash/car-driving-segmentation-unet-from-scratch
