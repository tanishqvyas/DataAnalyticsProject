from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,plot_confusion_matrix
import numpy as np
def NaiveBayes(x_train,y_train,x_test,y_test):

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred=gnb.predict(x_test)

    #plot_confusion_matrix(gnb, x_test, y_test)
    #plot_confusion_matrix(gnb, x_test, y_test, normalize='true')

    print("(Naive Bayes)Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    print("(Naive Bayes)Report: \n",classification_report(y_test,y_pred))
    print("(Naive Bayes)Accuracy: \n",accuracy_score(y_test, y_pred))

    y_gnb_prob = np.array(gnb.predict_proba(x_test)[:,1]).reshape(-1,1)

    return y_test,y_gnb_prob

