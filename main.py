'''
@authors : ["Poojasree D", "Aparna Gopalakrishnan", "Tanishq Vyas"]

'''

### Imports ###

# --External Imports-- #
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --Coustom Imports-- #
from backend.plotter import Plotter
from backend.preprocessing import Preprocessor


# --Control Variables / Meta-data-- #
meta_data = {

	"wannaPreprocess" :True,
	"wannaPlot" : True,
	"wannaTrain" : False,
	"wannaTest" : False
}

# Objects of Custom class imports


### Global Paths

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
		"Female" : 0,
		"Male" : 1,
		
		"Yes" : 1,
		"No" : 0,
		"No internet service": 2,
		"No phone service": 2,

		"DSL" : 1,
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
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "Partner", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "Dependents", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "PhoneService", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "StreamingMovies", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "PaymentMethod", String_to_Num_mapping)
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "Contract", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "PaperlessBilling", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "StreamingTV", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "TechSupport", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "DeviceProtection", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "OnlineBackup", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "OnlineSecurity", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "InternetService", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "MultipleLines", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "gender", String_to_Num_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "Churn", String_to_Num_mapping)	

	
	print("Processed Churn Dataset\n")
	print(initialDataFrame.head(), "\n")


	# Saving the preprocessed data
	preprocessorObj.save_file(initialDataFrame, processed_csv_path)

	

# Loading the preprocessed data file
dataFrame = pd.DataFrame(pd.read_csv(processed_csv_path))
#dataFrame = dataFrame[1:2000]
print("Info about the set \n",dataFrame.describe())
print("Number of Unique values  \n",dataFrame.nunique())

# Plotting the data
if(meta_data["wannaPlot"]):


	# Creating the object for Plotter class
	plotterObj = Plotter(dataFrame)

	plotterObj.plot_piechart("Churn", "Churn")
	plotterObj.plot_piechart("gender", "title for the plot")
	plotterObj.plot_piechart("Partner", "title for the plot")
	plotterObj.plot_piechart("SeniorCitizen", "Senior")
	plotterObj.plot_bargraph("MonthlyCharges", "title for the plot", "X labeling", "Y labeling")
	# plotterObj.plot_histogram("CreditScore", "title for plot", "X labeling", "Y labeling")
	# plotterObj.plot_scatterPlot("MonthlyCharges", "TotalCharges", "title for the plot", "X labeling", "Y labeling")
	plotterObj.plot_boxPlot("MonthlyCharges", "Boxplot", "X labeling", "Y labeling")
	plotterObj.plot_normalProbabilityPlot(["MonthlyCharges"], "title for the plot", "X labeling", "Y labeling")
	
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
	sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2, cmap="YlGnBu")
	plt.show()
	print("Females left",dataFrame.query("gender ==0 and Churn ==1").shape[0])
	print("Females stayed",dataFrame.query("gender ==0 and Churn ==0").shape[0])
	print("Males left",dataFrame.query("gender ==1 and Churn ==1").shape[0])
	print("Males stayed",dataFrame.query("gender ==1 and Churn ==0").shape[0])
	print("People having partners who left",dataFrame.query("Partner ==1 and Churn ==1").shape[0])
	print("People having partners who stayed",dataFrame.query("Partner ==1 and Churn ==0").shape[0])
	print("People not having partners who left",dataFrame.query("Partner ==0 and Churn ==1").shape[0])
	print("People not having partners who stayed",dataFrame.query("Partner ==0 and Churn ==0").shape[0])
	print("Multiplelines who left",dataFrame.query("MultipleLines ==0 and Churn ==1").shape[0],dataFrame.query("MultipleLines ==1 and Churn ==1").shape[0],dataFrame.query("MultipleLines ==2 and Churn ==1").shape[0])
	print("Multiplelines who stayed",dataFrame.query("MultipleLines ==0 and Churn ==0").shape[0],dataFrame.query("MultipleLines ==1 and Churn ==0").shape[0],dataFrame.query("MultipleLines ==2 and Churn ==0").shape[0])
	print("InternetSerivces who left",dataFrame.query("InternetService ==0 and Churn ==1").shape[0],dataFrame.query("InternetService==1 and Churn ==1").shape[0],dataFrame.query("InternetService ==2 and Churn ==1").shape[0])
	print("Contract who left",dataFrame.query("Contract ==0 and Churn ==1").shape[0],dataFrame.query("Contract==1 and Churn ==1").shape[0],dataFrame.query("Contract ==2 and Churn ==1").shape[0])
	print("PaymentMethod who left",dataFrame.query("PaymentMethod ==0 and Churn ==1").shape[0],dataFrame.query("PaymentMethod==1 and Churn ==1").shape[0],dataFrame.query("PaymentMethod ==2 and Churn ==1").shape[0],dataFrame.query("PaymentMethod ==3 and Churn ==1").shape[0])
	print("TechSupport who left",dataFrame.query("TechSupport ==0 and Churn ==1").shape[0],dataFrame.query("TechSupport==1 and Churn ==1").shape[0],dataFrame.query("TechSupport ==2 and Churn ==1").shape[0])
	print("People who are independent who left",dataFrame.query("Dependents==0 and Churn ==1").shape[0])
	print("People who are dependent who left",dataFrame.query("Dependents ==1 and Churn ==1").shape[0])
	

	bins = [0,12,24,36,48,72]
	df=dataFrame.loc[dataFrame['Churn'] == 1]
	df = df.groupby(pd.cut(df['tenure'], bins=bins)).tenure.count()
	#Lower bound excluded, upper included in each intweval
	ax=df.plot(kind='bar')
	for p in ax.patches:
    		ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
	plt.show()
	'''
	#Checking if plot is correct
	#Lower bound excluded, upper included
	df=dataFrame.loc[dataFrame['Churn'] == 1]
	res=df.loc[(df['tenure'] >12) & (df['tenure']<=24)]
	print(res)
	'''
	
	
# Training the model
if(meta_data["wannaTrain"]):
	pass


# Testing the model
if(meta_data["wannaTest"]):
	pass
