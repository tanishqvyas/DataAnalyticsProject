from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
import os




def SupportVectorMachine(x_train, x_test, y_train, y_test):

    # Path to save model
    path_to_model = os.path.join("model", "SVM.sav")

    # Selecting the type of vector
    model = SVC(kernel='poly', degree=2) 
    
    # Training the model
    model.fit(x_train, y_train)

    # Getting the test predictions
    y_pred = model.predict(x_test)

    # Printing the training results
    print("\n\n(Support Vector Machine) Confusion Matrix: \n", confusion_matrix(y_true=y_test, y_pred=y_pred.round()))
    print("(Support Vector Machine) Report: \n",classification_report(y_test,y_pred.round()))
    print("(Support Vector Machine) Accuracy: \n",accuracy_score(y_test, y_pred.round()))

    # Saving the Model
    pickle.dump(model, open(path_to_model, 'wb'))

    return y_test, y_pred