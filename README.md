#Data Analytics Project



**Folder Structure**

```
+
|---- data
|
|---- utils
|
|---- plots
|
|---- report
|
|---- presentation
|
|---- main.py
|
|---- .gitignore
|
|---- README.md
|
+

```

**Setup Instructions**

Step-1 : fork the project repository by pressing the fork button on top right. This creates your own fork of this repository. Once created it reads

<github-username>/DataAnalyticsProject
	forked from tanishqvyas/DataAnalyticsProject


on the top left.


Step-2 : Open your own fork i.e. github.com/<your-github-username>/DataAnalyticsProject and click on the green color code button and copy the link which would look something like

```
https://github.com/<your-github-username>/DataAnalyticsProject.git
```

Step-3 : Cloning has to be carried out by opening terminal / git bash in the desired folder where you want to have your project folder to be placed. Then type the command

```
git clone "<link-copied-in-step-two>"
```

```
cd DataAnalyticsProject
```


Step-4 : Setting up the upstream for your cloned repository. In order to do so type the following command

```
git remote add upstream "https://github.com/tanishqvyas/DataAnalyticsProject.git"
```

Step-5 : In order to check if the upstream is set properly or not type,

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

