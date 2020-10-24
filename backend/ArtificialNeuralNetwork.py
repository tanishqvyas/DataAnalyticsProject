import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



def ArtificialNeuralNetwork(x_train, x_test, y_train, y_test):


    model = Sequential()
    model.add(Dense(64, input_dim=19, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(8, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))


    model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=8)

    plt.plot(history.history['accuracy']) 
    plt.plot(history.history['val_accuracy']) 
    plt.title('model accuracy') 
    plt.ylabel('accuracy')
    plt.xlabel('epoch') 
    plt.legend(['train', 'validation'], loc='upper left') 
    plt.show()