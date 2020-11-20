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




def Stacked_Ensemble(x_train, x_test, y_train, y_test):

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

    print("\n\n(Stacked Ensemble) Confusion Matrix: \n", confusion_matrix(y_true=y_test, y_pred=y_pred.round()))
    print("(Stacked Ensemble) Report: \n",classification_report(y_test,y_pred.round()))
    print("(Stacked Ensemble) Accuracy: \n",accuracy_score(y_test, y_pred.round()))

    return y_test, y_pred