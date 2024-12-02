import torch
import pandas as pd
from homeModel import HomeScoreNN  # Import the model class
from awayModel import AwayScoreNN # Import the model class
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
def load_away_model(model_path="awayModel.pth"):
    model = AwayScoreNN()  # Initialize model structure
    model.load_state_dict(torch.load(model_path))  # Load trained weights
    model.eval()  # Set model to evaluation mode
    return model

def load_home_model(model_path="homeModel.pth"):
    model = HomeScoreNN()  # Initialize model structure
    model.load_state_dict(torch.load(model_path))  # Load trained weights
    model.eval()  # Set model to evaluation mode
    return model

# Main prediction function
def main():
    # Load the test data

    # test_data = pd.read_csv("TEST_DATA_NotNormalized.csv") # non normalized data
    # test_data = pd.read_csv("TEST_PCA.csv") # loads pca reduced data
    test_data = pd.read_csv("TEST_PCA_STANDARDIZED.csv") # reduced and standardized data

    # Separate features and true scores (for reference)
    X_test = test_data.drop(["score_away", "score_home"], axis=1).values
    y_away_test = test_data["score_away"].values  # Actual scores
    y_home_test = test_data["score_home"].values

    # Convert to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_away_test = torch.tensor(y_away_test, dtype=torch.float32)
    y_home_test = torch.tensor(y_home_test, dtype=torch.float32)

    # Load model and make predictions
    awayModel = load_away_model("awayModel.pth")
    homeModel = load_home_model("homeModel.pth")
    with torch.no_grad():
        awayPredictions = awayModel(X_test)
        homePredictions = homeModel(X_test)

    # Calculate and display the test loss
    criterion = torch.nn.MSELoss()
    test_away_loss = criterion(awayPredictions, y_away_test)
    test_home_loss = criterion(homePredictions, y_home_test)
    print(f"Test Loss Away (MSE): {test_away_loss.item():.4f}")
    print(f"Test Loss Home (MSE): {test_home_loss.item():.4f}")
    print("----------------\n")
    total = 0
    wins = 0
    # Display some predictions alongside actual values
    for i in range(500):  # Display first 20 predictions
        awayPrediction = awayPredictions[i].item()
        homePrediction = homePredictions[i].item()
        awayScore = y_away_test[i].item()
        homeScore = y_home_test[i].item()
        if homePrediction > awayPrediction and homeScore > awayScore:
            wins += 1
        if awayPrediction > homePrediction and awayScore > homeScore:
            wins += 1
        total += 1
        print(f"Predicted: (away: {awayPredictions[i].item()}, home: {homePredictions[i].item()})",
              f"Actual: (away: {y_away_test[i].item()}, home: {y_home_test[i].item()})")
        
    predicted_away_scores = awayPredictions.numpy().flatten()
    actual_away_scores = y_away_test.numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_away_scores, predicted_away_scores, alpha=0.6)
    plt.plot([actual_away_scores.min(), actual_away_scores.max()],[actual_away_scores.min(), actual_away_scores.max()], 'r--', lw=2)  # Diagonal reference line
    plt.xlabel("Actual Away Scores")
    plt.ylabel("Predicted Away Scores")
    plt.title("Predicted vs. Actual Away Scores")
    plt.show()

    predicted_home_scores = homePredictions.numpy().flatten()
    actual_home_scores = y_home_test.numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_home_scores, predicted_home_scores, alpha=0.6)
    plt.plot([actual_home_scores.min(), actual_home_scores.max()],[actual_home_scores.min(), actual_home_scores.max()], 'r--', lw=2)  # Diagonal reference line
    plt.xlabel("Actual Home Scores")
    plt.ylabel("Predicted Home Scores")
    plt.title("Predicted vs. Actual Home Scores")
    plt.show()

    predicted_totals = predicted_home_scores + predicted_away_scores
    TP = (predicted_home_scores > predicted_away_scores) & (actual_home_scores > actual_away_scores)
    FP = (predicted_home_scores > predicted_away_scores) & (actual_home_scores < actual_away_scores)
    TN = (predicted_home_scores < predicted_away_scores) & (actual_home_scores < actual_away_scores)
    FN = (predicted_home_scores < predicted_away_scores) & (actual_home_scores > actual_away_scores)
    colors = np.full(actual_home_scores.shape, 'yellow')
    colors[TP] = 'green'
    colors[TN] = 'blue'
    colors[FP] = 'red'
    colors[FN] = 'orange'
    actual_totals = actual_away_scores + actual_home_scores
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_totals, predicted_totals, c=colors, alpha=0.6)
    plt.xlabel("Actual Total Scores")
    plt.ylabel("Predicted Total Scores")
    plt.title("Predicted vs. Actual Total Scores")
    plt.scatter([], [], c='green', label='TP (True Positive)', alpha=0.6)
    plt.scatter([], [], c='red', label='FP (False Positive)', alpha=0.6)
    plt.scatter([], [], c='blue', label='TN (True Negative)', alpha=0.6)
    plt.scatter([], [], c='orange', label='FN (False Negative)', alpha=0.6)
    plt.legend()
    plt.show()

    residuals_home = actual_home_scores - predicted_home_scores
    residuals_away = actual_away_scores - predicted_away_scores
    plt.clf()
    plt.figure(figsize=(12, 5))
    # Residual plot for home scores
    plt.subplot(1, 2, 1)
    plt.scatter(predicted_home_scores, residuals_home, alpha=0.6, color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Home Scores')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot for Home Team Scores')

    # Residual plot for away scores
    plt.subplot(1, 2, 2)
    plt.scatter(predicted_away_scores, residuals_away, alpha=0.6, color='green')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Away Scores')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot for Away Team Scores')
    plt.tight_layout()
    plt.show()
        
    print(f"Total Win Percentage: {round(wins / total, 3) * 100}")
# Run the main function when this script is executed
if __name__ == "__main__":
    main()