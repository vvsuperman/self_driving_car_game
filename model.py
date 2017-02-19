import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import random
from PIL import Image
import csv
import numpy as np
import cv2

x_train =[]
y_train =[]

def crop_img(image_path):
    im = Image.open(image_path.strip())
    im = im.crop((0,im.size[1]/3,im.size[0],im.size[1]-25)) #crop the top 1/5 and the bottom 25 pix down
    return im

#import image and crop and make some flip
i=0
files=['data/driving_log.csv','driving_log_20.csv']
for file in files:
    with open(file) as csvfile:
        spamreader = csv.reader(csvfile)
        prefix=file[0:file.find('/')+1]
        for row in spamreader:
            angle = float(row[3].strip())
            if angle==0:
                if random.random()<0.02:            
                    i+=1
                    im_center=crop_img(prefix+row[0].strip())
                    x_train.append(np.array(im_center))
                    y_train.append(angle)

            else:

                #center image
                im_center=crop_img(prefix+row[0].strip())
                x_train.append(np.array(im_center))
                y_train.append(angle)

                #flip image and add

                imageFlipped = cv2.flip(np.array(im_center), 1)
                x_train.append(imageFlipped)
                y_train.append(-angle)

                #left image
                left_angle = angle+0.25
                im_left=crop_img(prefix+row[1].strip())
                x_train.append(np.array(im_left))
                y_train.append(left_angle)

                #flip image and add
                imageFlipped = cv2.flip(np.array(im_left), 1)
                x_train.append(imageFlipped)
                y_train.append(-left_angle)

                #right image
                right_angle = angle-0.25
                im_right=crop_img(prefix+row[2].strip())
                x_train.append(np.array(im_right))
                y_train.append(right_angle)

                #flip image and add
                imageFlipped = cv2.flip(np.array(im_right), 1)
                x_train.append(imageFlipped)
                y_train.append(-right_angle)
                      

print('loading data finished')
print(i,len(y_train))

## 1.3 resize and hsv the image    
def hsv_image(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

for i in range(len(x_train)):
    img = hsv_image(x_train[i])
    x_train[i] = cv2.resize(img,(100,33),interpolation=cv2.INTER_AREA) 


# 2 split the valid data and the test data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

x_train, y_train = shuffle(x_train, y_train)
x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape)


#split the data to train and test
train_features, test_features, train_labels, test_labels = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    random_state=40)


#split the data to train and valid
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.2,
    random_state=11)

# 3 Architechture
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten,Dropout,Lambda
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x/255.-0.5, input_shape=(33,100,3),))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(1,name='output'))
model.summary()


from keras.optimizers import Adam
import json

model.compile(loss='mse',optimizer=Adam(lr=0.001),metrics=['accuracy'])


def generator(x_train,y_train,batch_size):
    len_train = len(train_features)
    while 1:        
        for offset in range(0, len_train, batch_size):
            yield x_train[offset:offset+batch_size],y_train[offset:offset+batch_size]

            


history = model.fit_generator(generator(train_features, train_labels, 256), len(train_features), 
                              nb_epoch=4,validation_data=generator(valid_features, valid_labels, 256),
                              nb_val_samples = len(valid_features))


score = model.evaluate(test_features, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

json_string = model.to_json()
model.save_weights('model.h5')
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

