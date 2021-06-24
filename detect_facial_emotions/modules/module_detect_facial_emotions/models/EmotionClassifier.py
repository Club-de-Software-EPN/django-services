from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D,Dropout,Flatten,Dense
from keras.callbacks import ModelCheckpoint
import os

class EmotionClassifier:

    def __init__(self, model=None, verbose=0):

        self.verbose = verbose

        if model is None:
            self.modelN = EmotionClassifier.get_default_model_nn(verbose=verbose)

    def train(self, path_save_model, X_train, y_train, X_val, y_val, num_epochs=50, batch_size=150, optimizer=Adam(lr=0.0001, decay=1e-6)) -> dict:
        self.modelN.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        filepath= os.path.join(path_save_model, "weights_min_loss.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=self.verbose-1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]


        history = self.modelN.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
            validation_data=(X_val, y_val), shuffle=True,callbacks = callbacks_list) # , verbose=self.verbose-1

        self.modelN.save(os.path.join(path_save_model, 'facial_1.h5'))


        return history








    @staticmethod
    def get_default_model_nn(input_shape=(48,48,1), num_classes=6, verbose=0):

        
        modelN = Sequential()

        modelN.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu',name = 'conv1'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))

        modelN.add(layers.Conv2D(128, kernel_size=(3, 3),padding="same", activation='relu', name = 'conv2'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv3'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv4'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))

        
        """

        

        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv5'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv6'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv7'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv8'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))

        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv9'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv10'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv11'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv12'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))

        modelN.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv13'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv14'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu',  name = 'conv16'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv17'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))


        modelN.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv22'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv23'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu',  name = 'conv24'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv25'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        modelN.add(Dropout(0.25))
        """


        modelN.add(Flatten())
        modelN.add(Dense(num_classes, activation='softmax'))

        if verbose > 0:
            print(modelN.summary())

        return modelN

    @staticmethod
    def get_full_model_nn(input_shape=(48,48,1), num_classes=6, verbose=0):

        
        modelN = Sequential()

        modelN.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu',name = 'conv1'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))

        modelN.add(layers.Conv2D(128, kernel_size=(3, 3),padding="same", activation='relu', name = 'conv2'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv3'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv4'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))

        
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv5'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv6'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv7'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv8'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))

        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv9'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv10'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv11'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv12'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))

        modelN.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv13'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv14'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu',  name = 'conv16'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv17'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2)))
        modelN.add(Dropout(0.25))


        modelN.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv22'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv23'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu',  name = 'conv24'))
        modelN.add(BatchNormalization())
        modelN.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu', name = 'conv25'))
        modelN.add(BatchNormalization())
        modelN.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        modelN.add(Dropout(0.25))


        modelN.add(Flatten())
        modelN.add(Dense(num_classes, activation='softmax'))

        if verbose > 0:
            print(modelN.summary())

        return modelN



