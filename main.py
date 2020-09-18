'''
@authors : ["Poojashree D", "Aparna Gopalakrishnan", "Tanishq Vyas"]

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
	"wannaPlot" : True,
	"wannaTrain" : False
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
	
	# Making the object of the Preprocessor
	preprocessorObj = Preprocessor()

	# Mapping Dicts
	Gender_mapping = {
		"Female" : 0,
		"Male" : 1
	}

	Yes_No_mapping = {
		"Yes" : 1,
		"No" : 0,
		"No internet service": 2
	}

	# Chaneg strings to integers
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "Partner", Yes_No_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "Dependents", Yes_No_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "PhoneService", Yes_No_mapping)	
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "StreamingMovies", Yes_No_mapping)	

	
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