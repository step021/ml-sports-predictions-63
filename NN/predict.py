# predict.py

import torch
import pandas as pd
from model import MultiOutputNN  # Import the model class

# Load the trained model
def load_model(model_path="model.pth"):
    model = MultiOutputNN()  # Initialize model structure
    model.load_state_dict(torch.load(model_path))  # Load trained weights
    model.eval()  # Set model to evaluation mode
    return model

# Main prediction function
def main():
    # Load the test data (already scaled)
    test_data = pd.read_csv("TEST_2.csv")

    # Separate features and true scores (for reference)
    X_test = test_data.drop(["score_away", "score_home"], axis=1).values
    y_test = test_data[["score_away", "score_home"]].values  # Actual scores

    # Convert to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Load model and make predictions
    model = load_model("model.pth")
    with torch.no_grad():
        predictions = model(X_test)

    # Calculate and display the test loss
    criterion = torch.nn.MSELoss()
    test_loss = criterion(predictions, y_test)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")

    # Display some predictions alongside actual values
    predNum = 100
    successes = 0
    for i in range(predNum):  # Display first 20 predictions
        predicted_away_score = predictions[i][0].item()
        predicted_home_score = predictions[i][1].item()
        actual_away_score = y_test[i][0].item()
        actual_home_score = y_test[i][1].item()
        
        print(f"Predicted: (away: {predicted_away_score}, home: {predicted_home_score})",
            f"Actual: (away: {actual_away_score}, home: {actual_home_score})")
        
        # Check if prediction matches the actual outcome
        if (predicted_away_score > predicted_home_score and actual_away_score > actual_home_score) or \
        (predicted_away_score < predicted_home_score and actual_away_score < actual_home_score):
            successes += 1

    print(f"Probability of successfully predicting the winner: {successes / predNum}")


# Run the main function when this script is executed
if __name__ == "__main__":
    main()
