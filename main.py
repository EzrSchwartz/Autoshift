import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
# Load data
data = pd.read_csv("data.csv")


features = data[['power', 'cadence', 'hr']].values
labels = data['label'].values
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

X_train = torch.FloatTensor(features_train)
y_train = torch.LongTensor(labels_train)
X_test = torch.FloatTensor(features_test)
y_test = torch.LongTensor(labels_test)

class BikeShifterNet(nn.Module):
    def __init__(self):
        super(BikeShifterNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model = BikeShifterNet()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    correct = (predicted == y_test).sum().item()
    print(f"Accuracy: {100 * correct / len(y_test)}%")


while True:
    start = input("Would you like to enter your own data or let the system make some and print out training data (Training)(Own Data):").lower()

    if start == 'own data':
        while True:
            try:
                power = float(input("Enter power (W): "))
                cadence = float(input("Enter cadence (rpm): "))
                hr = float(input("Enter heart rate (bpm): "))

                feature = scaler.transform([[power, cadence, hr]])
                feature_tensor = torch.FloatTensor(feature)

                # Predict
                with torch.no_grad():
                    output = model(feature_tensor)
                    _, prediction = torch.max(output, 1)
                    predicted_action = prediction.item()

                # Decide action
                if predicted_action == 0:
                    print("Stay in the current gear.")
                elif predicted_action == 1:
                    print("Shift up. (Harder)")
                elif predicted_action == 2:
                    print("Shift down. (Easier)")
                else:
                    print("Unexpected prediction.")

                cont = input("Do you want to input another set of values? (yes/no) ").lower()
                if cont != 'yes':
                    break

            except ValueError:
                print("Invalid input. Please enter valid numbers.")
    elif start == 'training':

        Dataproduced = 0

        while True:

            Dataproduced += 1

            try:

                power = random.randint(100, 400)

                cadence = random.randint(60, 110)

                hr = random.randint(100, 220)

                feature = scaler.transform([[power, cadence, hr]])

                feature_tensor = torch.FloatTensor(feature)

                # Predict

                with torch.no_grad():

                    output = model(feature_tensor)

                    _, prediction = torch.max(output, 1)

                    predicted_action = prediction.item()

                # Print generated data and prediction

                print(str(power) + "," + str(cadence) + "," + str(hr) + "," + str(predicted_action))

                if Dataproduced == 10000:
                    break

            except ValueError:
                print("Error occurred while generating data.")



