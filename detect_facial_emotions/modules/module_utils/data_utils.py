
from keras.utils.np_utils import to_categorical
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import brewer2mpl
import os

def emotion_count(emotion, y_train, classes):
    """
    Esta función reclasifica la etiqueta 'Disgust' como 'Angry'
    """
    emo_classcount = {}
    print ('Disgust classified as Angry')
    y_train.loc[y_train == 1] = 0
    classes.remove('Disgust')
    for new_num, _class in enumerate(classes):
        y_train.loc[(y_train == emotion[_class])] = new_num
        class_count = sum(y_train == (new_num))
        emo_classcount[_class] = (new_num, class_count)
    return y_train.values, emo_classcount


def load_data(emotion, sample_split=0.3, usage='Training',classes=['Angry','Happy'], filepath='/content/drive/MyDrive/Colab Notebooks/Proyecto AI 2.0/fer2013.csv'):
    """
    Esta función carga el dataset en formato csv y realizamos el reshape y rescale de los datos para el feeding del modelo
    """
    df = pd.read_csv(filepath)
    df = df[df.Usage == usage]
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df[df['emotion'] == emotion[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(list(data.index), int(len(data)*sample_split))
    data = data.loc[rows]
    x = list(data["pixels"])
    X = []
    for i in range(len(x)):
        each_pixel = [int(num) for num in x[i].split()]
        X.append(each_pixel)
    ## reshape en las dimensiones 48*48*1 y rescale
    X = np.array(X)
    X = X.reshape(X.shape[0], 48, 48,1)
    X = X.astype("float32")
    X /= 255
    
    y_train, new_dict = emotion_count(emotion, data.emotion, classes)
    y_train = to_categorical(y_train)
    return X, y_train

def load_train_val_test(
    emotions,
    filepath,
    train_sample_split=1.0,
    val_sample_split=1.0,
    test_sample_split=1.0,
    classes=['Angry','Happy']
    ):
    

    """
    Esta función es la versión optimizada de load_data
    carga el dataset en formato csv y realizamos el reshape y rescale de los datos para el feeding del modelo
    """
    df = pd.read_csv(filepath)
    df_train = df[df.Usage == 'Training']
    df_val = df[df.Usage == 'PublicTest']
    df_test = df[df.Usage == 'PrivateTest']

    ### TRAINING
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df_train[df_train['emotion'] == emotions[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(list(data.index), int(len(data)*train_sample_split))
    data = data.loc[rows]
    x = list(data["pixels"])
    X_train = []
    for i in range(len(x)):
        each_pixel = [int(num) for num in x[i].split()]
        X_train.append(each_pixel)
    ## reshape en las dimensiones 48*48*1 y rescale
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], 48, 48,1)
    X_train = X_train.astype("float32")
    X_train /= 255
    
    y_train, new_dict = emotion_count(emotions, data.emotion, classes)
    y_train = to_categorical(y_train)



    ### VALIDATION
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df_val[df_val['emotion'] == emotions[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(list(data.index), int(len(data)*val_sample_split))
    data = data.loc[rows]
    x = list(data["pixels"])
    X_val = []
    for i in range(len(x)):
        each_pixel = [int(num) for num in x[i].split()]
        X_val.append(each_pixel)
    ## reshape en las dimensiones 48*48*1 y rescale
    X_val = np.array(X_val)
    X_val = X_val.reshape(X_val.shape[0], 48, 48,1)
    X_val = X_val.astype("float32")
    X_val /= 255
    
    y_val, new_dict = emotion_count(emotions, data.emotion, classes)
    y_val = to_categorical(y_val)



    ### TEST
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df_test[df_test['emotion'] == emotions[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(list(data.index), int(len(data)*test_sample_split))
    data = data.loc[rows]
    x = list(data["pixels"])
    X_test= []
    for i in range(len(x)):
        each_pixel = [int(num) for num in x[i].split()]
        X_test.append(each_pixel)
    ## reshape en las dimensiones 48*48*1 y rescale
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], 48, 48,1)
    X_test = X_test.astype("float32")
    X_test /= 255
    
    y_test, new_dict = emotion_count(emotions, data.emotion, classes)
    y_test = to_categorical(y_test)


    return X_train, y_train, X_val, y_val, X_test, y_test

def save_data(X_test, y_test, path, fname=''):
    """
    La función almacena los datos cargados en formato numpy para luego realizar el procesamiento
    """
    np.save( os.path.join(path, 'X_test' + fname), X_test)
    np.save( os.path.join(path, 'y_test' + fname), y_test)


def plot_subjects(start, end, y_pred, y_true, X, title=False):
    """
    La función se utiliza para indicar de que emoción trata la imagen
    """
    fig = plt.figure(figsize=(12,12))
    emotion = {0:'Angry', 1:'Fear', 2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral'}
    for i in range(start, end+1):
        input_img = X[i:(i+1),:,:,:]
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(input_img[0,:,:,0], cmap=matplotlib.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        if y_pred[i] != y_true[i]:
            plt.xlabel(emotion[y_true[i]], color='#53b3cb',fontsize=12)
        else:
            plt.xlabel(emotion[y_true[i]], fontsize=12)
        if title:
            plt.title(emotion[y_pred[i]], color='blue')
        plt.tight_layout()
    plt.show()



def plot_probs(start,end, y_prob, X):
    """
    La función se utiliza para graficar la probabilidad en el histograma para seis etiquetas 
    """
    fig = plt.figure(figsize=(12,12))
    for i in range(start, end+1):
        input_img = X[i:(i+1),:,:,:]
        ax = fig.add_subplot(6,6,i+1)
        set3 = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
        ax.bar(np.arange(0,6), y_prob[i], color=set3,alpha=0.5)
        ax.set_xticks(np.arange(0.5,6.5,1))
        labels = ['angry', 'fear', 'happy', 'sad', 'surprise','neutral']
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
        ax.set_yticks(np.arange(0.0,1.1,0.5))
        plt.tight_layout()
    plt.show()

def plot_subjects_with_probs(start, end, y_prob, y_pred, y_true, X):
    """
    Esta función sirve para representar la probabilidad que predice el modelo junto con su imagen
    """
    iter = int((end - start)/6)
    for i in np.arange(0,iter):
        plot_subjects(i*6,(i+1)*6-1, y_pred, y_true, X, title=False)
        plot_probs(i*6,(i+1)*6-1, y_prob, X)