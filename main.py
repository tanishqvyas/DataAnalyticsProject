'''
@authors : ["Poojasree D", "Aparna Gopalakrishnan", "Tanishq Vyas"]

'''

### Imports ###

# --External Imports-- #
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn

# --Custom Imports-- #
from backend.plotter import Plotter
from backend.preprocessing import Preprocessor

from backend.RandomForest import RandomForest
from backend.ArtificialNeuralNetwork import ArtificialNeuralNetwork
from backend.DecisionTree import DecisionTree
from backend.NaiveBayes import NaiveBayes
from backend.knn import KNearestNeighbour
from backend.SVM import SupportVectorMachine
from backend.Ensemble import Stacked_Ensemble
from sklearn.metrics import roc_curve, auc

# --Control Variables / Meta-data-- #
meta_data = {

    "wannaPreprocess": True,
    "wannaPlot": True,
    "wannaRunSubmissionCode": True,
    "wannaTrainTest": True
}

# Objects of Custom class imports

# Global Paths

# --Directories-- #
plots_dir_path = os.path.join("backend", "plots")
data_dir_path = os.path.join("backend", "data")

# --files-- #
initial_csv_path = os.path.join(data_dir_path, "initialdata.csv")
processed_csv_path = os.path.join(data_dir_path, "processeddata.csv")


# Getting the dataframe from the initialdata.csv
initialDataFrame = pd.DataFrame(pd.read_csv(initial_csv_path))
print("\nChurn Dataset Sample View (Initial)\n")
print(initialDataFrame.head(), "\n")

#------------------------------ STEPS -----------------------------------------#

# Preprocessing the data
if(meta_data["wannaPreprocess"]):

    # Removing Duplicates
    initialDataFrame = initialDataFrame[~initialDataFrame.duplicated()]

    # Making the object of the Preprocessor
    preprocessorObj = Preprocessor()

    # Mapping Dictionary
    String_to_Num_mapping = {
        "Female": 0,
        "Male": 1,

        "Yes": 1,
        "No": 0,
        "No internet service": 2,
        "No phone service": 2,

        "DSL": 1,
        "Fiber optic": 2,

        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,

        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }

    # Chaneg strings to integers
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "Partner", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "Dependents", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "PhoneService", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "StreamingMovies", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "PaymentMethod", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "Contract", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "PaperlessBilling", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "StreamingTV", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "TechSupport", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "DeviceProtection", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "OnlineBackup", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "OnlineSecurity", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "InternetService", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "MultipleLines", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "gender", String_to_Num_mapping)
    initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(
        initialDataFrame, "Churn", String_to_Num_mapping)

    initialDataFrame.TotalCharges.replace([" "], ["0"], inplace=True)
    initialDataFrame.TotalCharges = initialDataFrame.TotalCharges.astype(float)

    print("Processed Churn Dataset\n")
    print(initialDataFrame.head(), "\n")

    # Saving the preprocessed data
    preprocessorObj.save_file(initialDataFrame, processed_csv_path)


# Loading the preprocessed data file
dataFrame = pd.DataFrame(pd.read_csv(processed_csv_path))
print("Info about the set \n", dataFrame.describe())
print("Number of Unique values  \n", dataFrame.nunique())


# Code regarding the submission of the Project Report 1
if meta_data["wannaRunSubmissionCode"]:

    # Part 1
    seaborn.countplot(dataFrame.Churn)
    plt.title("Customer Churned vs Retained")
    blue_patch = mpatches.Patch(label='Retained')
    orange_patch = mpatches.Patch(color="orange", label='Churned')
    plt.legend(handles=[blue_patch, orange_patch])
    plt.show()

    # Part 2
    plt.hist(dataFrame.MonthlyCharges)
    plt.ylim(0, 3000)
    plt.title('Monthly Charges Distribution')
    plt.xlabel('MonthlyCharges')
    plt.show()

    # Part 3
    plt.title('Gender Distribution for Churning')
    seaborn.countplot(dataFrame.gender, hue=dataFrame.Churn)
    plt.legend(handles=[blue_patch, orange_patch])
    plt.xlabel('Gender | (Female : 0, Male : 1)')
    plt.show()

    # Part 4
    plt.title('Customer has Partner alongside Churning')
    seaborn.countplot(dataFrame.Partner, hue=dataFrame.Churn)
    plt.legend(handles=[blue_patch, orange_patch])
    plt.xlabel('Partner | (No Partner : 0, Has Partner : 1)')
    plt.show()

    # Part 5
    plt.title('Seniority Distribution for Churning')
    seaborn.countplot(initialDataFrame.SeniorCitizen,
                      hue=initialDataFrame.Churn)
    plt.legend(handles=[blue_patch, orange_patch])
    plt.xlabel('Seniority | (0 : Not Senior Citizen, 1 : Is Senior citizen')
    plt.show()

    # Plot 6
    plt.title('Services Distribution')
    first_patch = mpatches.Patch(label='No Service')
    second_patch = mpatches.Patch(color="orange", label='Using Service')
    third_patch = mpatches.Patch(color="green", label='No Internet Service')
    seaborn.countplot(x="variable", hue="value", data=pd.melt(
        dataFrame[['OnlineSecurity', 'DeviceProtection', 'TechSupport', 'OnlineBackup']]))
    plt.legend(handles=[first_patch, second_patch, third_patch])
    plt.xlabel("Services")
    plt.show()

    # Plot 7
    plt.title('Customer has Dependents alongside Churning')
    seaborn.countplot(dataFrame.Dependents, hue=dataFrame.Churn)
    plt.legend(handles=[blue_patch, orange_patch])
    plt.xlabel('Dependents | (No Dependents : 0, Has Dependents : 1)')
    plt.show()

    # Plot 8
    # plt.title('Box Plot for Total Charges')
    # seaborn.boxplot(data=dataFrame, showfliers=True)
    # plt.show()


# Plotting the data
if(meta_data["wannaPlot"]):

    # Creating the object for Plotter class
    plotterObj = Plotter(dataFrame)

    # plotterObj.plot_piechart("Churn", "Churn")
    # plotterObj.plot_piechart("gender", "Gender Distribution")
    # plotterObj.plot_piechart("Partner", "title for the plot")
    # plotterObj.plot_piechart("SeniorCitizen", "Senior")
    # plotterObj.plot_bargraph("MonthlyCharges", "title for the plot", "X labeling", "Y labeling")
    # plotterObj.plot_histogram("CreditScore", "title for plot", "X labeling", "Y labeling")
    # plotterObj.plot_scatterPlot("MonthlyCharges", "TotalCharges", "title for the plot", "X labeling", "Y labeling")
    # plotterObj.plot_boxPlot("MonthlyCharges", "Boxplot", "X labeling", "Y labeling")
    # plotterObj.plot_normalProbabilityPlot(["MonthlyCharges"], "title for the plot", "X labeling", "Y labeling")

    '''
	sns.countplot(dataFrame.MultipleLines,hue=dataFrame.Churn)
	plt.show()
	sns.countplot(dataFrame.Dependents,hue=dataFrame.Churn)
	plt.show()
	sns.countplot(dataFrame.PhoneService,hue=dataFrame.Churn)
	plt.show()
	sns.countplot(dataFrame.InternetService,hue=dataFrame.Churn)
	plt.show()
	sns.countplot(dataFrame.OnlineSecurity,hue=dataFrame.Churn)
	plt.show()
	'''
    plt.figure(figsize=(12, 6))
    corr = dataFrame.corr()
    print(corr)
    sns.heatmap(corr, xticklabels=corr.columns,
                yticklabels=corr.columns, linewidths=.2, cmap="YlGnBu")
    plt.show()
    print("Females left", dataFrame.query("gender ==0 and Churn ==1").shape[0])
    print("Females stayed", dataFrame.query(
        "gender ==0 and Churn ==0").shape[0])
    print("Males left", dataFrame.query("gender ==1 and Churn ==1").shape[0])
    print("Males stayed", dataFrame.query("gender ==1 and Churn ==0").shape[0])
    print("People having partners who left", dataFrame.query(
        "Partner ==1 and Churn ==1").shape[0])
    print("People having partners who stayed", dataFrame.query(
        "Partner ==1 and Churn ==0").shape[0])
    print("People not having partners who left",
          dataFrame.query("Partner ==0 and Churn ==1").shape[0])
    print("People not having partners who stayed",
          dataFrame.query("Partner ==0 and Churn ==0").shape[0])
    print("Multiplelines who left", dataFrame.query("MultipleLines ==0 and Churn ==1").shape[0], dataFrame.query(
        "MultipleLines ==1 and Churn ==1").shape[0], dataFrame.query("MultipleLines ==2 and Churn ==1").shape[0])
    print("Multiplelines who stayed", dataFrame.query("MultipleLines ==0 and Churn ==0").shape[0], dataFrame.query(
        "MultipleLines ==1 and Churn ==0").shape[0], dataFrame.query("MultipleLines ==2 and Churn ==0").shape[0])
    print("InternetSerivces who left", dataFrame.query("InternetService ==0 and Churn ==1").shape[0], dataFrame.query(
        "InternetService==1 and Churn ==1").shape[0], dataFrame.query("InternetService ==2 and Churn ==1").shape[0])
    print("Contract who left", dataFrame.query("Contract ==0 and Churn ==1").shape[0], dataFrame.query(
        "Contract==1 and Churn ==1").shape[0], dataFrame.query("Contract ==2 and Churn ==1").shape[0])
    print("PaymentMethod who left", dataFrame.query("PaymentMethod ==0 and Churn ==1").shape[0], dataFrame.query(
        "PaymentMethod==1 and Churn ==1").shape[0], dataFrame.query("PaymentMethod ==2 and Churn ==1").shape[0], dataFrame.query("PaymentMethod ==3 and Churn ==1").shape[0])
    print("TechSupport who left", dataFrame.query("TechSupport ==0 and Churn ==1").shape[0], dataFrame.query(
        "TechSupport==1 and Churn ==1").shape[0], dataFrame.query("TechSupport ==2 and Churn ==1").shape[0])
    print("People who are independent who left", dataFrame.query(
        "Dependents==0 and Churn ==1").shape[0])
    print("People who are dependent who left", dataFrame.query(
        "Dependents ==1 and Churn ==1").shape[0])

    bins = [0, 12, 24, 36, 48, 72]
    df = dataFrame.loc[dataFrame['Churn'] == 1]
    df = df.groupby(pd.cut(df['tenure'], bins=bins)).tenure.count()
    # Lower bound excluded, upper included in each intweval
    ax = df.plot(kind='bar')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x()
                                          * 1.005, p.get_height() * 1.005))
    plt.show()
    '''
	#Checking if plot is correct
	#Lower bound excluded, upper included
	df=dataFrame.loc[dataFrame['Churn'] == 1]
	res=df.loc[(df['tenure'] >12) & (df['tenure']<=24)]
	print(res)
	'''


############################################## MODEL TRAINING & Tersting #######################################

# Loading the dataset from the cleaned csv
dataSet = pd.read_csv(processed_csv_path)

# Separating Features and Label
features = dataSet.drop(dataSet.columns[[-1]], axis=1)
features = features.drop(dataSet.columns[[0]], axis=1)
label = dataSet[dataSet.columns[-1]]

print("------------------------------------------")
print(features)

print("------------------------------------------")

print("\n\n------------------------------------------")
print(label)

print("------------------------------------------")


# train test split to globally pass the data to models
x_train, x_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, random_state=42)


if(meta_data["wannaTrainTest"]):

    # Random Forest
    y_rf_test, y_rf_prob = RandomForest(x_train, x_test, y_train, y_test)

    # Artificial Neural Network
    nn_y_test, nn_y_pred = ArtificialNeuralNetwork(x_train, x_test, y_train, y_test)

    # DecisionTree
    y_tree_test, y_tree_prob = DecisionTree(x_train, x_test, y_train, y_test)

    # Naive Bayes
    y_nb_test, y_nb_prob = NaiveBayes(x_train, y_train, x_test, y_test)

    # KNN
    y_knn_test, y_knn_prob = KNearestNeighbour(x_train, y_train, x_test, y_test)

    # SVM
    y_svm_test, y_svm_prob = SupportVectorMachine(x_train, x_test, y_train, y_test)

    # ENSEMBLE
    ensemble_y_test, ensemble_y_pred = Stacked_Ensemble(x_train, x_test, y_train, y_test)




    # Plotting ---------------------------------------------------->
    # ROC curve for accuracies
    plt.figure(figsize=(7, 7), dpi=100)

    # for random forest
    rf_fpr, rf_tpr, threshold_rf = roc_curve(y_rf_test, y_rf_prob)
    auc_rf = auc(rf_fpr, rf_tpr)
    plt.plot(rf_fpr, rf_tpr, marker='.',
             label='Random Forest (auc = %0.3f)' % auc_rf)

    # for ann
    nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(
        nn_y_test, nn_y_pred)
    auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
    plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.',
             label='Neural Network (auc = %0.3f)' % auc_keras)

    # for decision tree
    tree_fpr, tree_tpr, threshold_tree = roc_curve(
        y_tree_test, y_tree_prob, drop_intermediate=False)
    auc_tree = auc(tree_fpr, tree_tpr)
    plt.plot(tree_fpr, tree_tpr, marker='.',
             label='Decision Tree (auc = %0.3f)' % auc_tree)

    # for naive bayes
    nb_fpr, nb_tpr, threshold_nb = roc_curve(
        y_nb_test, y_nb_prob, drop_intermediate=False)
    auc_nb = auc(nb_fpr, nb_tpr)
    plt.plot(nb_fpr, nb_tpr, marker='.',
             label='Naive Bayes (auc = %0.3f)' % auc_nb)

    # for KNN
    knn_fpr, knn_tpr, threshold_knn = roc_curve(
        y_knn_test, y_knn_prob, drop_intermediate=False)
    auc_knn = auc(knn_fpr, knn_tpr)
    plt.plot(knn_fpr, knn_tpr, marker='.',
             label='K-Nearest Neighbors (auc = %0.3f)' % auc_knn)


    # for SVM
    svm_fpr, svm_tpr, threshold_svm = roc_curve(
        y_svm_test, y_svm_prob, drop_intermediate=False)
    auc_svm = auc(svm_fpr, svm_tpr)
    plt.plot(svm_fpr, svm_tpr, marker='.',
             label='Support Vector Machine (auc = %0.3f)' % auc_svm)

    
    # for Ensemble
    ens_fpr, ens_tpr, threshold_ens = roc_curve(
        ensemble_y_test, ensemble_y_pred, drop_intermediate=False)
    auc_ens = auc(ens_fpr, ens_tpr)
    plt.plot(ens_fpr, ens_tpr, marker='.',
             label='Ensemble (Stacked) (auc = %0.3f)' % auc_ens)


    plt.xlabel('FPR  (1-specificity)')
    plt.ylabel('TPR  (sensitivity)')
    plt.title('ROC')
    plt.grid(True)
    plt.legend(loc="lower right", prop={'size': 22})
    plt.show()
