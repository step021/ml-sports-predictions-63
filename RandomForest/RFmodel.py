import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib

TRAINING_DATA_FILE = "TRAINING_2.csv"
MODEL_SAVE_PATH = "multi_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    features = data.drop(columns=["score_home", "score_away"])
    targets = data[["score_home", "score_away"]]
    return features, targets

def train_random_forest(xTrain, yTrain):
    rf_model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
    rf_model.fit(xTrain, yTrain)
    return rf_model

def train_adaboost(xTrain, yTrain):
    base_learner = DecisionTreeRegressor(max_depth=1, random_state=RANDOM_STATE)
    adaboost_model = MultiOutputRegressor(AdaBoostRegressor(
        estimator=base_learner,
        n_estimators=100,
        learning_rate=.1,
        random_state=RANDOM_STATE
    ))
    adaboost_model.fit(xTrain, yTrain)
    return adaboost_model

def save_model(model, path):
    print(f"Saving model to {path}...")
    joblib.dump(model, path)

if __name__ == "__main__":
    X, y = load_and_preprocess_data(TRAINING_DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Choose a model to train:")
    print("1: Random Forest")
    print("2: AdaBoost")
    model_choice = input("Enter your choice (1/2): ").strip()

    if model_choice == "1":
        model = train_random_forest(X_train, y_train)
    elif model_choice == "2":
        model = train_adaboost(X_train, y_train)
    else:
        print("Invalid choice. Exiting.")
        exit()

    save_model(model, MODEL_SAVE_PATH)