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


def RandomForest(x_train, x_test, y_train, y_test):

    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                              random_state=42)
    rf.fit(x_train, y_train) 
    score = rf.score(x_train, y_train)
    test_score = rf.score(x_test, y_test)

    print("(Random Forest) Training set accuracy: ", '%.3f'%(score))
    print("(Random Forest) Testing set accuracy: ", '%.3f'%(test_score))

    rf_predictions = rf.predict(x_test)
    rf_probs = rf.predict_proba(x_test)

    y_pred = rf.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))