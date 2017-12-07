"""
Confusion matrix script for 
binaural localization neural
net. 

Author: Greg Hunkins
Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
"""
from keras.models import load_model
from datagenerator import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

#if os.path.abspath('~') == '/Users/ghunk/~':
if True:
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
      'shuffle': False}


LIMIT = 200000
RANDOM_STATE = 3

# Datasets
IDs = os.listdir(data_root)[:LIMIT]

Train_IDs, Test_IDs, _, _, = train_test_split(IDs, np.arange(len(IDs)), test_size=0.2, random_state=RANDOM_STATE)
validation_generator = DataGenerator(**params).generate(Test_IDs)

y_test = []
for ID in Test_IDs:
	split = ID[:-4].split('_')
	y_test.append(encoder.transform([split[2] + '_' + split[3]])[0])


model = load_model('results/model_200000_job_1689840.h5py')

y_pred = model.predict_generator(generator=validation_generator, steps=len(Test_IDs)//32, verbose=1)
y_pred_transform = encoder.inverse_transform(y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap="gpuplot2"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred_transform)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig("confusion_matrix.png")
plt.show()