import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class RandomForestClassifier:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train;
        self.x_test = x_test;
        self.y_train = y_train;
        self.y_test = y_test;

    def trainModel(self):

        rf.fit(self.x_train, self.y_train)
        score = rf.score(self.x_train, self.y_train)
        score2 = rf.score(self.x_test, self.y_test)
        print("Training set accuracy: ", '%.3f' % (score))
        print("Test set accuracy: ", '%.3f' % (score2))

        rf_predictions = rf.predict(self.x_test)
        rf_probs = rf.predict_proba(self.x_test)



        y_pred = rf.predict(self._test)
        print(confusion_matrix(self.y_test, self.y_pred))
        print(classification_report(self.y_test, self.y_pred))
        print(accuracy_score(self.y_test, self.y_pred)