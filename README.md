# Group 63
## Sebastian Stephens, Greyson McReynolds, Reuben Covey, Sofia Varmeziar, Sam Deckbar 
### Machine learning class project to use different ML techniques and models in order to try and predict the winner of NFL games. Currently have a win rate of about 60%.
### **Helpful Git commands**
To Clone Repo(enter in command line): 
```
git clone https://github.com/step021/ml-sports-predictions-63.git
```
Create your own brance when working:
```
git checkout -b <your_name>
```
Git workflow to push to remote repo:
```
git add <file_name> #adds changes to staging
git status #verify changes made are in staging area
git commit -m "message" #commits to local repo
git push -u origin <branch_name> #first time
git push origin <branch_name> #any other time
```

### Description of Directories and Files
##### /ml-sports-predictions-63/awayModel.pth: Path of where the away neural net model will be written to.
##### /ml-sports-predictions-63/awayModel.py: Python file which defines a neural network to predict the scores of the away team. The name of the file where the data gets read to can be changed to determine which dataset the model is trained on.
##### /ml-sports-predictions-63/environment.yaml: File that is used for the creation of the conda environment to make sure that your computer has all of the necessary dependencies.
##### /ml-sports-predictions-63/homeModel.pth: Path of where the home neural net model will be written to.
##### /ml-sports-predictions-63/index.html: Stores the HTML code for the Github Pages.
##### /ml-sports-predictions-63/modelGuide.txt: Text file to help users create and run the model.
##### /ml-sports-predictions-63/newPredict.py: Python file which uses the models that were previously trained to predict scores for NFl games from the past two seasons. It then loops through the results and sees how well it did. 
##### /ml-sports-predictions-63/NORMALIZED_teamdata.csv: Dataset that contains the adjusted data that has been transformed into a form that is ready to be used by the model.
##### /ml-sports-predictions-63/PCA.py: Python file which performs PCA on the dataset and only keeps the most relevant features as well as standardizing the data. 
##### /ml-sports-predictions-63/Preprocessing: Directory which contains important files used to preprocess the data and get it ready for training/testing.
##### /ml-sports-predictions-63/Preprocessing/Data-Normalization.ipynb: Python notebook used to normalize the dataset.
##### /ml-sports-predictions-63/Preprocessing/Data-Preprocessing.ipynb: Python notebook used to get the dataset in the correct format for training and predictions.
##### /ml-sports-predictions-63/Preprocessing/ORIGINAL_teamdata.csv: Original dataset
##### /ml-sports-predictions-63/Preprocessing/TRAINABLE_teamdata.csv: Dataset after it has been altered to be in the right format.
##### /ml-sports-predictions-63/TEST_PCA_STANDARDIZED.csv: Test dataset that has had PCA ran on it and has been standardized.
##### /ml-sports-predictions-63/TRAIN_PCA_STANDARDIZED.csv: Train dataset that has had PCA ran on it and has been standardized.
#####/ml-sports-predictions-63/midterm.html: More html code for github pages
#####/ml-sports-predictions-63/final.html: More html code for github pages
#####/ml-sports-predictions-63/proposal.html: More html code for github pages
#####/ml-sports-predictions-63/RandomForest/Notes.md: Notes on the model
#####/ml-sports-predictions-63/RandomForest/RFmodel.py: Python file which defines the random forest model
#####/ml-sports-predictions-63/RandomForest/TEST_2.csv: Data for random forest
#####/ml-sports-predictions-63/RandomForest/TRAINING_2.csv: Training data for random forest
#####/ml-sports-predictions-63/RandomForest/TESTRF.py: Tests the random forest model.
#####/ml-sports-predictions-63/GradientBoost/gbModel.py: the gradient boost model python file we implemented.
#####/ml-sports-predictions-63/GradientBoost/TRAINING_PCA_STANDARDIZED.csv:training data for gradient boost.
#####/ml-sports-predictions-63/GradientBoost/TEST_PCA_STANDARDIZED.csv:Test data for gradient boost.

