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
    test_data = pd.read_csv("TEST_DATA.csv")

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
    for i in range(20):  # Display first 20 predictions
        print(f"Predicted: (away: {predictions[i][0].item()}, home: {predictions[i][1].item()})",
              f"Actual: (away: {y_test[i][0].item()}, home: {y_test[i][1].item()})")

# Run the main function when this script is executed
if __name__ == "__main__":
    main()
