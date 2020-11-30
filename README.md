# Data Analytics Project



**Folder Structure**

```
+
|---- backend
|	|
|	|---- data
|	|	|
|	|	|---- initialdata.csv
|	|       |
|	|       +
|	|
|	|---- plots
|   |   |
    |	|   +
|	|
|	|---- plotter.py
|	|
|	|---- preprocessing.py
|       |
|       |---- ArtificialNeuralNetwork.py
|       |
|       |---- DecisionTree.py
|       |
|       |---- Ensemble.py
|       |
|       |---- knn.py
|       |
|       |---- NaiveBayes.py
|       |
|       |---- RandomForest.py
|       |
|       |---- SVM.py
|       |
|       +
|
|---- main.py
|
|---- .gitignore
|
|---- README.md
|
+

```

# Download Git

Head to git-scm.com and download git on your system before proceeding with the following steps.

# Setup Instructions for Contribution

**Step-1** : Fork the project repository by pressing the fork button on top right. This creates your own fork of this repository. Once created it reads

 your-github-username/DataAnalyticsProject forked from tanishqvyas/DataAnalyticsProject


on the top left.


**Step-2** : Open your own fork i.e. github.com/your-github-username/DataAnalyticsProject and click on the green color code button and copy the link which would look something like

```
https://github.com/<your-github-username>/DataAnalyticsProject.git
```

**Step-3** : Cloning has to be carried out by opening terminal / git bash in the desired folder where you want to have your project folder to be placed. Then type the command

```
git clone "<link-copied-in-step-two>"
```

```
cd DataAnalyticsProject
```


**Step-4** : Setting up the upstream for your cloned repository. In order to do so type the following command

```
git remote add upstream "https://github.com/tanishqvyas/DataAnalyticsProject.git"
```

**Step-5** : In order to check if the upstream is set properly or not type,

```
git remote -v
```

This should show an output similar to the following

```
origin    https://github.com/<your-github-username>/DataAnalyticsProject.git (fetch)
origin    https://github.com/<your-github-username>/DataAnalyticsProject.git (push)
upstream  https://github.com/tanishqvyas/DataAnalyticsProject.git (fetch)
upstream  https://github.com/tanishqvyas/DataAnalyticsProject.git (push)
```

If you are able to see this as the output the you have succesfully forked, cloned and setup upstream for your repository.


# Execution Instructions

**On windows**

Make sure you are in the correct directory, then type the following command

```
python main.py
```

**On Linux**

Make sure you are in the correct directory, then type the following command

```
python3 main.py
```

The **main.py** has a meta-data dictionary to turn off-on the knobs for preprocessing, plotting and training. Change them to view sub part of the code in execution. **Default Settings** makes the whole code run from loading the data till the training and prediction. 
