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
	"wannaPlot" : False,
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

if(meta_data["wannaPreprocess"]):
	
	# Making the object of the Preprocessor
	preprocessorObj = Preprocessor()

	# Mapping Dicts
	Geography_mapping = {
		"France": 0,
		"Germany" : 1,
		"Spain" : 2
	}

	Gender_mapping = {
		"Female" : 0,
		"Male" : 1
	}

	# Chaneg strings to integers
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "Geography", Geography_mapping)
	initialDataFrame = preprocessorObj.change_col_val_string_to_numeric(initialDataFrame, "Gender", Gender_mapping)
	print("Processed Churn Dataset\n")
	print(initialDataFrame.head(), "\n")



# Loading the preprocessed data file
# dataFrame = pd.DataFrame(pd.read_csv(processed_csv_path))


if(meta_data["wannaPlot"]):
	
	# Creating the object for Plotter class
	plotterObj = Plotter(initialDataFrame)

	plotterObj.plot_piechart("Geography", "title for the plot")
	plotterObj.plot_bargraph("Geography", "title for the plot", "X labeling", "Y labeling")
	# plotterObj.plot_histogram("CreditScore", "title for plot", "X labeling", "Y labeling")
	plotterObj.plot_scatterPlot("CreditScore", "EstimatedSalary", "title for the plot", "X labeling", "Y labeling")
	plotterObj.plot_boxPlot(["CreditScore"], "title for the plot", "X labeling", "Y labeling")
	plotterObj.plot_normalProbabilityPlot(["CreditScore"], "title for the plot", "X labeling", "Y labeling")