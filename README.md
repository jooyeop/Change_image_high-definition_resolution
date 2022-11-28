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


2. 인코더 (W h로 정의되는 아핀 변환affine transformation 후 스쿼싱squashing)를 거치는 입력 x으로 맨 아래서부터 시작한다. 이는 중간 은닉층hiddenn layer h을 형성한다.
이는 디코더 (또 다른 W x  로 정의되는 아핀 변환 다음에 또 다른 스쿼싱이 뒤따름)의 대상이 된다. 그러면 모델의 입력에 대한 예측/재건인 출력 x이 생성된다.
우리는 이를 관습convention에 따라 3층 신경망으로 부른다.

  
Autoencoders를 사용하는 이유
오토인코더의 주요primary 응용 분야는 이상 감지anomaly detection 또는 이미지 노이즈 제거image denoising이다.
오토인코더의 작업이 매니폴드 즉 주어진 데이터 매니폴드에 있는 데이터를 재건하는 것임을 알고 있고,
오토인코더가 해당 매니폴드 안에 존재하는 입력만 재건할 수 있기를 원한다.
따라서 우리는 모델이 훈련 중에 관찰한 것들만을 재건할 수 있도록 제한하고,
따라서 새로운 입력에 존재하는 어떠한 변화variation도 제거되는데,
왜냐하면 이 모델은 이러한 미세한 변화perturbations에 영향을 받지 않을insensitive 것이기 때문이다.

오토인코더의 또 다른 응용은 이미지 압축기compressor이다.
만일 입력 차원 n 보다 낮은 중간 차원 d 를 갖고 있다면,
인코더는 압축기로 사용될 수 있고, 은닉 표현hidden representation (코딩 된 표현)은 특정 입력의 모든 (또는 대부분) 정보를 전달하지만 적은 공간을 차지한다.

      
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
