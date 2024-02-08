import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn.functional as F
import pandas as pd


# Define the neural network model
class BikeShifterNet(nn.Module):
    def __init__(self):
        super(BikeShifterNet, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # 6 input features
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)  # Assuming 3 output classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# Function to handle training and evaluation
def train_and_evaluate(features_train, labels_train, features_test, labels_test, epochs=100, learning_rate=0.01,
                       momentum=0.9):
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(features_train)
    y_train = torch.LongTensor(labels_train)
    X_test = torch.FloatTensor(features_test)
    y_test = torch.LongTensor(labels_test)

    # Initialize the model
    model = BikeShifterNet()

    # Set the optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / len(y_test)

    return model, accuracy


# Training_ML function

def Training_MLGrades(powers, cadences, speeds, grades, labels, timelastshift, min_accuracy=101):
    # Convert labels to numeric if they are not
    if isinstance(labels[0], str):
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

    # Combine features and preprocess
    # Make sure all lists are of the same length and aligned
    features = list(zip(powers, cadences, speeds, grades, labels, timelastshift))
    features = [list(feature) for feature in features if None not in feature]  # Filter out any data points with None
    labels = [labels[i] for i in range(len(labels)) if None not in features[i]]

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split the data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.75,
                                                                                random_state=42)

    # Train the model and evaluate
    model, accuracy = train_and_evaluate(features_train, labels_train, features_test, labels_test)

    # Ensure model meets minimum accuracy requirement
    while accuracy < min_accuracy:
        model = BikeShifterNet()  # Create a new instance of the model in each iteration
        model, accuracy = train_and_evaluate(features_train, labels_train, features_test, labels_test)
        print(f"Retraining model. Current accuracy: {accuracy}%")

    print(f"Model trained with accuracy: {accuracy}%")

    # Extract model parameters for serialization or further use
    model_data = {
        'weights_fc1': model.fc1.weight.data.numpy().tolist(),
        'biases_fc1': model.fc1.bias.data.numpy().tolist(),
        'weights_fc2': model.fc2.weight.data.numpy().tolist(),
        'biases_fc2': model.fc2.bias.data.numpy().tolist(),
        'weights_fc3': model.fc3.weight.data.numpy().tolist(),
        'biases_fc3': model.fc3.bias.data.numpy().tolist(),
    }
    return model_data, scaler
