'''
@authors : ["Poojasree D", "Aparna Gopalakrishnan", "Tanishq Vyas"]

'''

### Imports ###

# --External Imports-- #
import os
import time
import pandas as pd

# --Coustom Imports-- #
from backend.plotter import Plotter
from backend.preprocessing import Preprocessor


# --Control Variables / Meta-data-- #
meta_data = {

	"wannaPreprocess" : True,
	"wannaPlot" : False,
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
dataFrame = dataFrame[1:2000]


# Plotting the data
if(meta_data["wannaPlot"]):


	
	# Creating the object for Plotter class
	plotterObj = Plotter(dataFrame)

	plotterObj.plot_piechart("Partner", "title for the plot")
	plotterObj.plot_bargraph("MonthlyCharges", "title for the plot", "X labeling", "Y labeling")
	# plotterObj.plot_histogram("CreditScore", "title for plot", "X labeling", "Y labeling")
	# plotterObj.plot_scatterPlot("MonthlyCharges", "TotalCharges", "title for the plot", "X labeling", "Y labeling")
	plotterObj.plot_boxPlot(["MonthlyCharges"], "title for the plot", "X labeling", "Y labeling")
	plotterObj.plot_normalProbabilityPlot(["MonthlyCharges"], "title for the plot", "X labeling", "Y labeling")


# Training the model
if(meta_data["wannaTrain"]):
	pass


# Testing the model
if(meta_data["wannaTest"]):
	pass