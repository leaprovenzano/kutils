import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight


def get_confusion(model, x, y, labels, norm=False, batch_size=32):
    expected_labels = np.argmax(y, axis = 1)
    val_predictions = model.predict(x, verbose=1, batch_size=batch_size)
    if type(val_predictions) is list:
        val_predictions = val_predictions[0]
    vpred_labels =  np.argmax(val_predictions, axis = 1)
    cm = confusion_matrix(expected_labels, vpred_labels)
    plot_confusion_matrix(cm, labels, normalize=norm, title='{} Confusion Matrix'.format(model.name))
    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def hist_concat(h1, h2, e=None):
    for item in h2.history:
        if e:
            h1.history[item] = h1.history[item][:e] +  h2
        else:
            h1.history[item] += h2.history[item]
    return h1


def plot_history(model_history):
    fig, axes = plt.subplots(1, 2, figsize = (20, 10))
    h = model_history.history
    a1, a2 = axes
    # summarize history for accuracy
    try:
        a1.plot(h['acc'])
        a1.plot(h['val_acc'])
        a1.set_title('model accuracy')
    except KeyError:
        pass

    # summarize history for loss
    a2.plot(h['loss'])
    a2.plot(h['val_loss'])
    a2.set_title('model loss')

    plt.ylabel('accuracy/ loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    