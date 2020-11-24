from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree

# function for fitting trees of various depths on the training data using cross-validation
def run_cross_validation_on_trees(x, y, depths):
    cv_scores = []
    cv_std = []
    cv_mean = []
    cv_accuracy= []
    for depth in depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, x, y, cv=5, scoring='accuracy')
        cv_scores=np.append([],cv_scores)
        cv_mean.append(cv_scores.mean())
        cv_std.append(cv_scores.std())
        cv_accuracy.append(tree_model.fit(x, y).score(x, y))
    return np.array(cv_mean), np.array(cv_std),np.array(cv_accuracy)

    
def find_best_depth(x_train, y_train):
    #To find which is the best depth for the decision tree model
    # fitting trees of depth 1 to 25
    depths = range(1,26)
    cv_mean, cv_std, cv_accuracy = run_cross_validation_on_trees(x_train, y_train, depths)

    id_max = cv_mean.argmax()
    best_depth = depths[id_max]
    best_score = cv_mean[id_max]
    best_std = cv_std[id_max]
    print('The depth',best_depth,' tree achieves the best mean cross-validation accuracy',(best_score*100),' +/- ',(best_std*100),'on training dataset')
    #The depth 5  tree achieves the best mean cross-validation accuracy 79.1581415947335  +/-  0.9010268771897787 on training dataset
    return best_depth

def DecisionTree(x_train, x_test, y_train, y_test):
    
    best_depth=find_best_depth(x_train,y_train) #obtained as 5
    # Create Decision Tree classifer object
    dt = DecisionTreeClassifier(max_depth=best_depth,random_state=42)   #default uses gini index

    # Train Decision Tree Classifer
    dt = dt.fit(x_train,y_train)

    #Predict the response for test dataset
    y_pred = dt.predict(x_test)
    
    print("(Decision Tree)Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    print("(Decision Tree)Report: \n",classification_report(y_test,y_pred))
    print("(Decision Tree)Accuracy: \n",accuracy_score(y_test, y_pred))
    
    #tree.plot_tree(dt,filled=True)
    return y_test,np.array(dt.predict_proba(x_test)[:,1]).reshape(-1,1)
