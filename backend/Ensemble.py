# make a prediction with a stacking ensemble
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import pickle
import os



def Stacked_Ensemble(x_train, x_test, y_train, y_test):

    # Path to save model
    path_to_model = os.path.join("model", "StackedEnsemble.sav")

    # define the base models
    level0 = list()
    level0.append(('lr', LinearRegression()))
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor()))
    level0.append(('svm', SVR()))
    level0.append(('adaboost', AdaBoostRegressor()))
    # level0.append(('bayes', ))
    

    # Classifier
    # level0.append(('lr', LogisticRegression()))
    # level0.append(('knn', KNeighborsClassifier()))
    # level0.append(('cart', DecisionTreeClassifier()))
    # level0.append(('svm', SVC()))
    # level0.append(('bayes', GaussianNB()))


    # define meta learner model
    level1 = LinearRegression()

    # Classifier
    # level1 = LogisticRegression()



    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    # model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)


    model.fit(x_train, y_train)

    # Predicting
    y_pred = model.predict(x_test)


    # Printing the training results
    print("\n\n(Stacked Ensemble) Confusion Matrix: \n", confusion_matrix(y_true=y_test, y_pred=y_pred.round()))
    print("(Stacked Ensemble) Report: \n",classification_report(y_test,y_pred.round()))
    print("(Stacked Ensemble) Accuracy: \n",accuracy_score(y_test, y_pred.round()))

    # Saving the Model
    if not os.path.exists(os.path.dirname(path_to_model)):
        try:
            os.makedirs(os.path.dirname(path_to_model))
        except OSError as exc: # Guard against race condition
            print("File does not exist !!!!")
            
    pickle.dump(model, open(path_to_model, 'wb'))

    return y_test, y_pred