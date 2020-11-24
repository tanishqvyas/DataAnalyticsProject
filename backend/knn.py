import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



def KNearestNeighbour(x_train,y_train,x_test,y_test):
	# Feature Scaling
	sc_X=StandardScaler() # Standardizes all data between -1 and 1
	sc_X.fit(x_train)
	x_train = sc_X.transform(x_train)
	x_test = sc_X.transform(x_test)
		
	k=int(math.sqrt(len(y_test)))
	if k%2==0:
		k-=1 
	
	classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean') # p=2 because it is a binary classifier
	
	# Fit the model
	classifier.fit(x_train,y_train)
	y_pred=classifier.predict(x_test)
	
	# Printing the training results
	print("KNN Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
	print("KNN Report: \n",classification_report(y_test,y_pred))
	print("KNN Accuracy: \n",accuracy_score(y_test, y_pred))	
	
	return y_test,np.array(classifier.predict_proba(x_test)[:,1]).reshape(-1,1)
	
