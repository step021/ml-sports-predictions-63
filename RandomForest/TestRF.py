import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TESTING_DATA_FILE = "TEST_2.csv"
MODEL_LOAD_PATH = "multi_model.pkl"
NUM_GAMES_TO_PRINT = 200

def load_model(path):
    print(f"Loading model from {path}...")
    return joblib.load(path)

def load_and_preprocess_test_data(file_path, feature_columns):
    data = pd.read_csv(file_path)
    features = data.drop(columns=["score_home", "score_away"])
    features = features.reindex(columns=feature_columns, fill_value=0)
    targets = data[["score_home", "score_away"]]
    return features, targets

def evaluate_model(model, X_test, y_test):
    print("Evaluating the model...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions, multioutput="raw_values")
    mse = mean_squared_error(y_test, predictions, multioutput="raw_values")
    r2 = r2_score(y_test, predictions, multioutput="raw_values")
    print(f"Home Scores - MAE: {mae[0]:.2f}, MSE: {mse[0]:.2f}, R²: {r2[0]:.2f}")
    print(f"Away Scores - MAE: {mae[1]:.2f}, MSE: {mse[1]:.2f}, R²: {r2[1]:.2f}")

    return predictions

def calculate_winner_accuracy(predictions, y_test):
    pred_winners = ["home" if home > away else "away" for home, away in predictions]
    actual_winners = ["home" if row["score_home"] > row["score_away"] else "away" for _, row in y_test.iterrows()]

    correct_predictions = sum([pred == actual for pred, actual in zip(pred_winners, actual_winners)])
    accuracy = (correct_predictions / len(y_test)) * 100
    print(f"Winner Prediction Accuracy: {accuracy:.2f}%")
    return accuracy

def print_predictions_vs_actual(predictions, y_test, num_games):
    print(f"\nPredicted vs Actual Scores (First {num_games} Games):")
    print("Game\tPredicted Home\tActual Home\tPredicted Away\tActual Away")
    for i in range(min(num_games, len(y_test))):
        pred_home, pred_away = predictions[i]
        actual_home, actual_away = y_test.iloc[i]
        print(f"{i+1}\t{pred_home:.1f}\t\t{actual_home:.1f}\t\t{pred_away:.1f}\t\t{actual_away:.1f}")

if __name__ == "__main__":
    model = load_model(MODEL_LOAD_PATH)

    dummy_train_data = pd.read_csv("TRAINING_2.csv")
    X_latest, y_latest = load_and_preprocess_test_data(TESTING_DATA_FILE, feature_columns=dummy_train_data.drop(columns=["score_home", "score_away"]).columns)

    predictions = model.predict(X_latest)
    evaluate_model(model, X_latest, y_latest)

    print_predictions_vs_actual(predictions, y_latest, NUM_GAMES_TO_PRINT)
    calculate_winner_accuracy(predictions, y_latest)