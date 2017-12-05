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
from datagenerator import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import itertools
import os

# Parameters

if os.path.abspath('~') == '/Users/ghunk/~':
	data_root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/stft_binaural_0.5s/"
else:
	data_root = "/scratch/ghunkins/stft_binaural_0.5s/"

elevations = [-45, -30, -15, 0, 15, 30, 45]
azimuths = [15*x for x in range(24)]
el_az = list(itertools.product(elevations, azimuths))
classes = [str(x) + '_' + str(y) for x, y in el_az]
encoder = LabelEncoder()
encoder.fit(classes)

params = {'batch_size': 32,
		  'Y_encoder': encoder,
          'shuffle': True}

LIMIT = 1000000
RANDOM_STATE = 3

# Datasets
IDs = os.listdir(data_root)[:LIMIT]

Train_IDs, Test_IDs, _, _, = train_test_split(IDs, np.arange(len(IDs)), test_size=0.2, random_state=RANDOM_STATE)

# Generators
training_generator = DataGenerator(**params).generate(Train_IDs)
validation_generator = DataGenerator(**params).generate(Test_IDs)

# Design model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(804, 47, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(GlobalAveragePooling2D(data_format='channels_last'))

model.add(Dense(168, activation='relu'))
model.add(Dense(168, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model on dataset

model.fit_generator(generator = training_generator,
					steps_per_epoch = len(Train_IDs)//params['batch_size'],
                    nb_epoch = 10, 
                    validation_data = validation_generator,
                    validation_steps = len(Test_IDs)//params['batch_size'],
                    verbose=2)

model.save_weights("./weights.h5py")
