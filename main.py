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

}



### Global Paths

# --Directories-- #
plots_dir_path = os.path.join("backend", "plots")
data_dir_path = os.path.join("backend", "data")

# --files-- #
initial_csv_path = os.path.join(data_dir_path, "initialdata.csv")
processed_csv_path = os.path.join(data_dir_path, "processeddata.csv")


# Getting the dataframe from the initialdata.csv
initialDataFrame = pd.read_csv(initial_csv_path)
print("Churn Dataset Sample View (Initial)\n")
print(initialDataFrame.head(), "\n")




#------------------------------ STEPS -----------------------------------------#

if(meta_data["wannaPreprocess"]):
	pass


# Loading the preprocessed data file
# dataFrame = pd.read_csv(processed_csv_path)

if(meta_data["wannaPlot"]):
	pass