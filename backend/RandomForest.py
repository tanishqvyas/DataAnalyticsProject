import numpy as np
from sklearn.model_selection import train_test_split 
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

    rf_probs = rf.predict_proba(x_test)

    # Getting predictions for test set
    y_pred = rf.predict(x_test)


    # Printing the training results
    print("(Random Forest)Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    print("(Random Forest)Report: \n",classification_report(y_test,y_pred))
    print("(Random Forest)Accuracy: \n",accuracy_score(y_test, y_pred))


    return y_test,np.array(rf_probs[:,1]).reshape(-1,1)