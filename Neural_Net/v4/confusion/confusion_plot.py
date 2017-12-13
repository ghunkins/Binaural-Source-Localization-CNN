"""
Confusion matrix script for 
binaural localization neural
net. 

Author: Greg Hunkins
Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

elevations = [-45, -30, -15, 0, 15, 30, 45]
azimuths = [15*x for x in range(24)]
el_az = list(itertools.product(elevations, azimuths))
classes = [str(x) + '_' + str(y) for x, y in el_az]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap="gnuplot2"):
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

np.set_printoptions(precision=2)

cnf_matrix = np.load('cnf_matrix.npy')
cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
ticks=np.linspace(0, 167,num=168)
plt.imshow(cnf_matrix, interpolation='none', cmap='plasma')
v = np.linspace(0., 1.0, 11, endpoint=True)
print v
plt.clim(0, 1.0)
plt.colorbar(ticks=v)
#plt.clim(0, 1.0)
plt.yticks(ticks, fontsize=4)
plt.xticks(ticks, fontsize=4)
#plt.set_xlabels([])
#plt.set_ylabels([])
plt.grid(False)
plt.axis('off')
plt.savefig('cnf.png', dpi = 1200)
# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=classes,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
#                      title='Normalized confusion matrix')

#plt.savefig("confusion_matrix.png")
#plt.show()