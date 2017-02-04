import csv
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten,Dropout
from keras.layers.convolutional import Convolution2D
import cv2
import json



x_train=[]
y_train=[]

resize_parm=4

#read data from csv
with open('driving_log.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
    	im = Image.open(row[0])
    	im = im.crop((0,im.size[1]/3,im.size[0],im.size[1])) #crop the img to 2/3
    	im = im.resize((200,66))#resize the image to nvida size	
    	x_train.append(np.array(im))
    	y_train.append(row[3])

#preprocess the data
#shuffle the data
x_train, y_train = shuffle(x_train, y_train)

x_train = np.array(x_train)
y_train = np.array(y_train)



#grayscale the data
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

x_train = normalize_grayscale(x_train)

#split the data to train and test
train_features, test_features, train_labels, test_labels = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    random_state=40)


#use CLAHE, maybe later


#split the data to train and valid
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.2,
    random_state=11)

#set the network
#input_shape = x_train.shape[1:]
#input_shape = train_features.shape[1:]


model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(66,200,3),border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1,name='output'))
model.summary()



def generator_data(x_set,y_set,batch_size):
    i=0
    while 1:   	
    	#yield(x_set[i:(i+1)*batch_size],y_set[i:(i+1)*batch_size])
    	#i+=1
    	for (x_data,y_data) in zip(x_set,y_set):	    		          	
	    	x_data= np.expand_dims(x_data,axis=0)
	    	y_data= np.expand_dims(y_data,axis=0)
	    	yield(x_data,y_data)
data_generator = generator_data(train_features, train_labels,1)
valid_generator = generator_data(valid_features, valid_labels,1)

#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

##Apply fit generator##

model.fit_generator(data_generator, samples_per_epoch = 4000,
                   nb_epoch=4, validation_data = valid_generator,
                   nb_val_samples = 400)




score = model.evaluate(test_features, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

json_string = model.to_json()
model.save_weights('model.h5')
with open('model.json', 'w') as outfile:
	json.dump(model.to_json(), outfile)










