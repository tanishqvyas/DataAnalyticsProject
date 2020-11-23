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
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc



def ArtificialNeuralNetwork(x_train, x_test, y_train, y_test):

    # Parameters
    name_of_model = "ANN_churn1"
    learning_rate = 0.007
    num_of_epochs = 50
    batch_size = 8
    loss_list = ["binary_crossentropy", "categorical_crossentropy"]

    checkpoint = ModelCheckpoint(
        name_of_model,
        # monitor = 'val_loss',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    earlystop = EarlyStopping(

        monitor='val_accuracy',
        min_delta=0,
        patience=7,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.8,
        patience=2,
        verbose=1,
        min_delta=0.0001
    )

    callbacks = [earlystop, checkpoint, reduce_lr]

    # Model
    model = Sequential()
    model.add(Dense(64, input_dim=19, activation='relu',
                    kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(8, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the model
    model.compile(loss=loss_list[0],
                optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(
        x_test, y_test), epochs=num_of_epochs, batch_size=batch_size, verbose=1)


    

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    y_pred = model.predict(x_test)


    print("(Neural Network) Confusion Matrix: \n", confusion_matrix(y_true=y_test, y_pred=y_pred.round()))
    print("(Neural Network) Report: \n",classification_report(y_test,y_pred.round()))
    print("(Neural Network) Accuracy: \n",accuracy_score(y_test, y_pred.round()))



    return y_test, y_pred

    