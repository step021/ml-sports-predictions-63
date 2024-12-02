The only data preprocessing we chose to employ was the removal of the most recent season from the dataset to use as a test set.
This is because Random Forest doesn't require any scaling or normalization because of the nature of the algorithm, as it instead splits the data at a specific value for each feature.
Random Forest also does not require a lot of feature reduction.
The only reason we would have applied, say PCA, is to speed up computational time, but this was not an issue.

First model iteration was 56.88% accurate in predicting winner.
It used the following hyperparameters:

max_depth = None
n_estimators = 100
min_samples_leaf = 1
min_samples_split = 2

The best combination of all the hyperparamters according to Grid Search
(searched 3 folds for each of 81 candidates) is:

max_depth = 10
min_samples_leaf = 4
min_samples_split = 10
n_esimtators = 300

The model went up to 57.62% accuracy.

After this, we implemented Adaboost.
We knew that Adaboost is sensitive to outliers (which were plentiful in our dataset), but we tried anyways.
Using the following parameters:

n_estimators = 300
learning_rate = .1

The model went down to 57.25% accuracy.

So, the best iteration of the model was the Random Forest with optimized hyperparameters (57.62%).