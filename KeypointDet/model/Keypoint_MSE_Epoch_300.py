# Facial Keypoint Detection
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
# 0. Random seed 세팅
def my_seed_everywhere(seed):
    random.seed(seed) # random
    np.random.seed(seed) # np
    tf.random.set_seed(seed) # tensorflow
    os.environ["PYTHONHASHSEED"] = '0' # os
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
my_seed = 42
my_seed_everywhere(my_seed)

# 1. Data 불러오기
Train_Dir = '../input/training.csv'
Test_Dir = '../input/test.csv'
lookid_dir = '../input/IdLookupTable.csv'
train_data = pd.read_csv(Train_Dir)  
test_data = pd.read_csv(Test_Dir)
lookid_data = pd.read_csv(lookid_dir)
os.listdir('../input')

# 2. 비어있는 데이터 채우기 
train_data.isnull().any().value_counts()
train_data.fillna(method = 'ffill',inplace = True)
train_data.isnull().any().value_counts()
len(train_data)

# 3. 학습 데이터셋에서 이미지 파싱
imag = []
for i in range(0,len(train_data)):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imag.append(img)
image_list = np.array(imag,dtype = 'float')
X_train = image_list.reshape(-1,96,96,1)

# 4. 학습 데이터셋에서 Keypoint 파싱
training = train_data.drop('Image',axis = 1)
training = training.drop('left_eye_inner_corner_x',axis = 1)
training = training.drop('left_eye_inner_corner_y',axis = 1)
training = training.drop('left_eye_outer_corner_x',axis = 1)
training = training.drop('left_eye_outer_corner_y',axis = 1)

training = training.drop('right_eye_inner_corner_x',axis = 1)
training = training.drop('right_eye_inner_corner_y',axis = 1)
training = training.drop('right_eye_outer_corner_x',axis = 1)
training = training.drop('right_eye_outer_corner_y',axis = 1)

y_train = []
for i in range(0,len(train_data)):
    y = training.iloc[i,:]

    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')

# 5. 모델 구축

from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D


model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1,seed=42))
model.add(Dense(22))
# model.summary()

# 6. 옵티마이저, Loss Function 설정

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])
# 7. 모델 학습
history = model.fit(X_train,y_train,epochs = 300,batch_size = 512, shuffle=False, validation_split=0.2)
min_loss = min(history.history['val_mae'])
min_epoch = history.history['val_mae'].index(min_loss)
print(f"min epoch is: {min_epoch}")

print(f'train mae is : ', history.history['mae'][min_epoch])
print(f'validation mae is : ', history.history['val_mae'][min_epoch])

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()