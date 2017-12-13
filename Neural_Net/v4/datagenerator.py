"""
Modified from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""
import numpy as np
import os

if os.path.abspath('~') == '/Users/ghunk/~':
  data_root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/stft_binaural_0.5s/"
else:
  data_root = "/scratch/ghunkins/stft_binaural_0.5s/"

class DataGenerator(object):
  '''Generates data for Keras'''
  def __init__(self, batch_size=32, Y_encoder=None, shuffle=True):
      'Initialization'
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.Y_encoder = Y_encoder

  def generate(self, list_IDs):
      '''Generates batches of samples'''
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              try:
              	X, Y = self.__data_generation(list_IDs_temp)
              except IOError:
              	continue

              yield X, Y

  def __get_exploration_order(self, list_IDs):
      '''Generates order of exploration'''
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, list_IDs_temp):
      '''Generates data of batch_size samples''' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization

      loaded_X = [np.load(data_root + ID) for ID in list_IDs_temp]
      x_dim, y_dim = min([x.shape for x in loaded_X]) 
      standardized_X = [x[...,:(y_dim-1)] for x in loaded_X]


      X = np.empty((self.batch_size, x_dim, y_dim-1, 1))
      Y = np.empty((self.batch_size), dtype = int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # transform to log
          ref_X = standardized_X[i]
          ref_X[ref_X == 0] = np.finfo(dtype='float32').tiny
          log_X = np.log10(standardized_X[i])
          # Store volume
          X[i, :, :, 0] = log_X
          #X[i, :, :, 0] = standardized_X[i]
          # Store class
          split = ID[:-4].split('_')
          Y[i] = self.Y_encoder.transform([split[2] + '_' + split[3]])[0]

      return X, sparsify(Y)

def sparsify(y):
  '''Returns labels in binary NumPy array'''
  n_classes = 168
  return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])

