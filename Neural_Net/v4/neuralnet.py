"""
Main Neural Network Pipeline.
"""
#-------------------------- set gpu using tf ---------------------------#
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#-------------------  start importing keras module ---------------------#

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv1D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from datagenerator import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import itertools
import os
import re
import sys

# Parameters

elevations = [-45, -30, -15, 0, 15, 30, 45]
azimuths = [15*x for x in range(24)]
el_az = list(itertools.product(elevations, azimuths))
classes = [str(x) + '_' + str(y) for x, y in el_az]
encoder = LabelEncoder()
encoder.fit(classes)

params_train = {'batch_size': 32,
		        'Y_encoder': encoder,
                'shuffle': True, 
                'dataroot': '/scratch/ghunkins/STFT_TRAIN_NOISE/'}

params_test = {'batch_size': 32,
                'Y_encoder': encoder,
                'shuffle': True, 
                'dataroot': '/scratch/ghunkins/STFT_TEST_NOISE/'}

TRAIN_LIMIT = 8000000
TEST_LIMIT = 8000000
RANDOM_STATE = 3

# Datasets
#IDs = os.listdir(data_root)[:LIMIT]

#Train_IDs, Test_IDs, _, _, = train_test_split(IDs, np.arange(len(IDs)), test_size=0.2, random_state=RANDOM_STATE)
db = '10db'

Train_IDs = os.listdir(params_train['dataroot'])
Train_IDs = list(filter(lambda z: (db in z), Train_IDs))[:TRAIN_LIMIT]
Test_IDs = os.listdir(params_test['dataroot'])
Test_IDs = list(filter(lambda z: (db in z), Test_IDs))[:TEST_LIMIT]
#np.save('Test_IDs.npy', Test_IDs)
#np.save('Train_IDs.npy', Train_IDs)
#sys.exit()
#Train_IDs = np.load('./train_test/train_speakers_list.npy')
#Test_IDs = np.load('./train_test/test_speakers_list.npy')


# Generators
training_generator = DataGenerator(**params_train).generate(Train_IDs)
validation_generator = DataGenerator(**params_test).generate(Test_IDs)

# Design model
model = Sequential()
model.add(Conv2D(256, kernel_size=(804, 1), activation='relu', input_shape=(804, 47, 1)))
model.add(Conv2D(256, kernel_size=(1, 3), strides=(1, 2), activation='relu'))
model.add(Conv2D(256, kernel_size=(1, 3), strides=(1, 2), activation='relu'))
model.add(Conv2D(256, kernel_size=(1, 3), strides=(1, 2), activation='relu'))
model.add(GlobalAveragePooling2D(data_format='channels_last'))

model.add(Dense(168, activation='relu'))
model.add(Dense(168, activation='softmax'))

# compile the model for categorical cross entropy using 
# Adam optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# print out the model summary
model.summary()

# set callback: https://machinelearningmastery.com/check-point-deep-learning-models-keras/

filepath = db + "-log-model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

# Train model on dataset
model.fit_generator(generator = training_generator,
					steps_per_epoch = len(Train_IDs)//params_train['batch_size'],
                    nb_epoch = 20, 
                    validation_data = validation_generator,
                    validation_steps = len(Test_IDs)//params_train['batch_size'],
                    verbose=2,
                    callbacks=callbacks_list)
