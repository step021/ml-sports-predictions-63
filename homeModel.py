import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the neural network model for home team score prediction
class HomeScoreNN(nn.Module):
    def __init__(self):
        super(HomeScoreNN, self).__init__()
        self.fc1 = nn.Linear(66, 128)  # Adjusted input size to 90
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output layer with 1 neuron (only score_home)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Linear activation for regression
        return x

# Load and prepare the training data
    
# data = pd.read_csv("TRAINING_DATA_NotNormalized.csv")  # Non-Scaled data
# data = pd.read_csv("TRAIN_PCA.csv") # Reduced features from PCA version
data = pd.read_csv("TRAIN_PCA_STANDARDIZED.csv") # Reduced and standardized features

# Separate features and target variables
X = data.drop(["score_away", "score_home"], axis=1).values  # Drop target columns
y = data["score_home"].values  # Select only score_home as the target

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Reshape to match output dimension
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)  # Reshape to match output dimension

# Initialize the model, loss function, and optimizer
model = HomeScoreNN()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training the model
losses = []
epochs = 150
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    losses.append(loss.item())

plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Home Loss')
plt.title('Home Loss Over Epochs')
plt.show()

# Evaluate the model on validation data
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val)
    print(f"Validation Loss (MSE): {val_loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "homeModel.pth")
print("Model saved as homeModel.pth")
