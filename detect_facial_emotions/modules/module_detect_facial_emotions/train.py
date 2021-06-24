
# reference https://colab.research.google.com/drive/1V7XMG9CB6zreYzURlE785ZECBob7NX4L#scrollTo=syGYPF_Ygced

from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random
import sys
import os
import warnings 
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import brewer2mpl
from .models.EmotionClassifier import EmotionClassifier
from ..module_utils.data_utils import load_data, save_data, plot_subjects_with_probs, load_train_val_test


def train(data_filepath, save_model_path, num_epochs=1, plot_history = False, verbose = 10):

    ## Los tres conjuntos de datos se los pasa por la función para utilizarlos luego
    emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
            'Sad': 4, 'Surprise': 5, 'Neutral': 6}
    emo     = ['Angry', 'Fear', 'Happy',
            'Sad', 'Surprise', 'Neutral']

    """
    X_train, y_train = load_data(emotion, sample_split=1.0,classes=emo,
    usage= 'Training', filepath=filepath)

    X_val,y_val = load_data(emotion, sample_split=1.0,classes=emo,
    usage= 'PublicTest', filepath=filepath)

    X_test, y_test = load_data(emotion, sample_split=1.0,classes=emo,
    usage='PrivateTest', filepath=filepath)
    """

    X_train, y_train, X_val, y_val, X_test, y_test = load_train_val_test(emotion, data_filepath, 0.1, 0.1, 0.1, emo)

    if verbose > 0:
        print("X_train.shape: ", X_train.shape)
        print("y_train.shape: ", y_train.shape)
        print("X_test.shape: ", X_test.shape)
        print("y_test.shape: ", y_test.shape)
        print("X_val.shape: ", X_val.shape)
        print("y_val.shape: ", y_val.shape)

    save_data(X_test, y_test, save_model_path, "_privatetest6_100pct")
    X_fname = os.path.join(save_model_path, 'X_test_privatetest6_100pct.npy')
    y_fname = os.path.join(save_model_path, 'y_test_privatetest6_100pct.npy')
    X = np.load(X_fname)
    y = np.load(y_fname)

    if verbose > 0:
        print ('Private test set')

    y_labels = [np.argmax(lst) for lst in y]
    counts = np.bincount(y_labels)
    labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']



    classifier = EmotionClassifier(verbose=verbose-1)
    history = classifier.train(save_model_path, X_train, y_train, X_val, y_val, num_epochs=num_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if plot_history:
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()



    # evaluation
    y_train = y_train 
    y_public = y_val 
    y_private = y_test 
    y_train_labels  = [np.argmax(lst) for lst in y_train]
    y_public_labels = [np.argmax(lst) for lst in y_public]
    y_private_labels = [np.argmax(lst) for lst in y_private]

    score = classifier.modelN.evaluate(X, y, verbose=verbose-1)
    print ("model %s: %.2f%%" % (classifier.modelN.metrics_names[1], score[1]*100))


    y_prob = classifier.modelN.predict(X, batch_size=32, verbose=0)
    y_pred = [np.argmax(prob) for prob in y_prob]
    y_true = [np.argmax(true) for true in y]


    # plot_subjects_with_probs(0, 36, y_prob, y_pred, y_true, X)
