from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv1D, GlobalAveragePooling2D

# Design model
model = Sequential()
model.add(Conv2D(256, kernel_size=(804, 1), activation='relu', input_shape=(804, 47, 1)))
model.add(Conv2D(256, kernel_size=(1, 3), strides=(1, 2), activation='relu'))
model.add(Conv2D(256, kernel_size=(1, 3), strides=(1, 2), activation='relu'))
model.add(Conv2D(256, kernel_size=(1, 3), strides=(1, 2), activation='relu'))
model.add(GlobalAveragePooling2D(data_format='channels_last'))

model.add(Dense(168, activation='relu'))
model.add(Dense(168, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()