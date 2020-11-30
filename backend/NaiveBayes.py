from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,plot_confusion_matrix
import numpy as np
import pickle
import os


def NaiveBayes(x_train,y_train,x_test,y_test):

    # Path to save model
    path_to_model = os.path.join("model", "NaiveBayes.sav")

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred=gnb.predict(x_test)

    # Printing the training results
    print("(Naive Bayes)Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    print("(Naive Bayes)Report: \n",classification_report(y_test,y_pred))
    print("(Naive Bayes)Accuracy: \n",accuracy_score(y_test, y_pred))

    y_gnb_prob = np.array(gnb.predict_proba(x_test)[:,1]).reshape(-1,1)

    # Saving the Model
    if not os.path.exists(os.path.dirname(path_to_model)):
        try:
            os.makedirs(os.path.dirname(path_to_model))
        except OSError as exc: # Guard against race condition
            print("File does not exist !!!!")

    pickle.dump(gnb, open(path_to_model, 'wb'))

    return y_test,y_gnb_prob

