import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import zscore, norm, binom, poisson


class Plotter(object):
	
	def __init__(self, dataFrame):
		self.dataFrame = dataFrame

	# This function fetches all the values in a column and returns the list
	def get_column_as_list(self, dataFrame, col_name):
        
		col_list = list(dataFrame[col_name].tolist())

		return col_list

	# Functon to extract entries and their counts from a column
	def structure_data(self, data_list):
		
		# Extract unique fields in the column
		label_set = list(set(data_list))

		# Get count for each field in column
		label_count = [data_list.count(i) for i in label_set]

		return label_set, label_count


	# Function to return Q1 Q2 or Q3
	def get_quartile_value(self, data_list, n, quartile_ratio):

		if (n+1)*quartile_ratio == int((n+1)*quartile_ratio):
			return data_list[(n+1)*quartile_ratio]

		else:
			return ( data_list[math.ceil((n+1)*quartile_ratio)] + data_list[math.floor((n+1)*quartile_ratio)] ) / 2

	# Function to find median
	def get_IQR(self, data_list):

		# finding number of elements
		num_of_elements = len(data_list)

		# fetching Q1, Q2 & Q3 values
		Q1 = self.get_quartile_value(data_list,num_of_elements,0.25)
		Q3 = self.get_quartile_value(data_list,num_of_elements,0.75)

		return Q3 - Q1


	# Function to find bin_size using Freedman-Diaconis formula
	def get_bin_size(self, data_list, n):

		bin_size = (2 * self.get_IQR(data_list)) / (math.pow(n, 1/3))
		return math.ceil(bin_size)


	# Function to calculate num of classes
	def get_count_classes(self, data_list):

		num_of_classes = (max(data_list)-min(data_list)) / self.get_bin_size(data_list, len(data_list))
		return math.ceil(num_of_classes)




	### 
	### Function To plot The different kinds of plots ###
	###

	# Function to plot pie-chart
	def plot_piechart(self, column_title, title):
		print("\n Plotting Pie Chart for : ", title)

		# Pre processing
		data_list = self.get_column_as_list(self.dataFrame, column_title)

		# Extracting labels and respective values
		label_set, label_count = self.structure_data(data_list)

		# Plottng
		plt.pie(label_count, labels=label_set, shadow=False, startangle=90, autopct='%.1f%%')
		plt.title(title)
		# plt.tight_layout()
		plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
		plt.show()


	# Functon to plot bar graph
	def plot_bargraph(self, column_title, title, xlabel, ylabel, isVertical = True):
		print("\n Plotting Bar Chart for : ", title)

		# Pre processing
		data_list = self.get_column_as_list(self.dataFrame ,column_title)

		# Extracting labels and respective values
		label_set, label_count = self.structure_data(data_list)
		
		# To use template styling
		plt.style.use('seaborn')

		if not isVertical:
			plt.barh(label_set, label_count)
			plt.ylabel(xlabel)
			plt.xlabel(ylabel)
		else:
			plt.bar(label_set, label_count)
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
		
		plt.title(title)
		# some padding it seems
		plt.tight_layout()
		plt.show()


	# Function to plot histogram
	def plot_histogram(self, column_title, title, xlabel, ylabel, plotCurve=False):
		print("\n Plotting Histogram for : ", title)

		
		# plotCurve variable is 1 if we wanna plot curve above histogram

		# Pre processing
		data_list = self.get_column_as_list(self.dataFrame, column_title)
		
		# Fnding num of bins
		num_of_bins = self.get_count_classes(data_list)

		# To use template styling
		plt.style.use('seaborn-whitegrid')

		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)

		if plotCurve:
			pass #todo


		plt.hist(data_list, bins = num_of_bins, normed=True)
		plt.show()
	

	# Functon to plot Scatter Plot
	def plot_scatterPlot(self, column_title1, column_title2, title, xlabel, ylabel):
		print("\n Plotting Scatter Plot for : ", title)

		# Pre processing
		data_list1 = self.get_column_as_list(self.dataFrame, column_title1)
		data_list2 = self.get_column_as_list(self.dataFrame, column_title2)


		# Plotting
		plt.scatter(data_list1, data_list2, color='r')
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()
	

	# function to plot box plot
	def plot_boxPlot(self, fieldList, title, xlabel, ylabel, isVertical=False):
		print("\n Plotting Box Plot for : ", title)

		#Test
		#print(plt.style.available)

		# Empty list to hold column data
		dataList = []

		# Loop to fetch column data and append in dataList
		for i in range(len(fieldList)):
			dataList.append(self.get_column_as_list(self.dataFrame, fieldList[i]))
		
		"""
		showfliers : boolean, true to show outliers
		flierprops : styling for outliers markers
		notch : boolean, true to show Notch at Q2
		vert : boolean, true for vertical box plot
		"""

		# Plot related stuff
		plt.style.use("seaborn-dark")
		fig1, myPlot = plt.subplots()
		#myPlot.set_title(title)
		plt.title(title)

		# Label management
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)

		if not isVertical:
			plt.xlabel(ylabel)
			plt.ylabel(xlabel)
		
		# Plotting and showing
		myPlot.boxplot(dataList, notch=False, showfliers=True, vert=isVertical, labels=fieldList)
		plt.show()


	# function to plot Normal Probability Plot
	def plot_normalProbabilityPlot(self,column_list, title, xlabel, ylabel):
		print("\n Plotting Normal Probab Plot for : ", title)

		for column in column_list:

			data = self.get_column_as_list(self.dataFrame, column)

			# Sorting the data
			data = sorted(data);

			# finding mean and standard deviation
			col_mean = np.mean(data)
			col_std = np.std(data)


			# Finding the P value i.e.  P = (position - 0.5)/len of data
			# modified list essentially contains values randing from 0 - 1
			modified_list = [(data.index(i) + 1 -0.5)/ len(data) for i in data]

			# getting zscore list for the list of P
			zscore_list = zscore(modified_list)

			# Finding theoretical Quantile for each value in zscore_list
			theoretical_quantile_list = [ (col_std*z) + col_mean  for z in zscore_list]

			# Plotting labels and stuff
			plt.title(title)
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)

			# plotting scatter plot
			plt.scatter(zscore_list, theoretical_quantile_list)

			plt.show()


if __name__ == '__main__':
    pass