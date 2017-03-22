"""confusion matrix
confusion matrix usage to evaluate the quality of the output of a classifier on the image data set. 
The diagonal elements represent the number of points for which the predicted label is equal to the 
true label, while off-diagonal elements are those that are mislabeled by the classifier. 
The higher the diagonal values of the confusion matrix the better, indicating many correct predictions.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

from sklearn.metrics import confusion_matrix

raise NotImplementedError("Port this source code to python3!")

np.set_printoptions(precision=2)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plotcnf(expectedTags, predictedTags):
  cnf_matrix = confusion_matrix(expectedTags, predictedTags)
  class_names = list(set(expectedTags))

  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

  #plt.figure()
  #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix, with normalization')