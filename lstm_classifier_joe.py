import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Load the dataset from CSV
df = pd.read_csv("out.csv")

# Drop non-numeric columns
# Shift the mood column forward by 1

df['mood_shifted'] = df.groupby('id')['mood'].shift(-1)  # Assuming you want to predict the next instance
df.dropna(subset=['mood_shifted'], inplace=True)
df.sort_values("date", inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# Assuming test_targets and test_predictions are the expected and predicted quantiles

# Define features and target



# Normalize features
scaler = StandardScaler()
non_numeric_columns = ['id', 'date','screen','mood' ,'Unnamed: 0',]
df.drop(columns=non_numeric_columns, inplace=True)
features = df.drop(columns=['mood_quantiles']).values.astype(np.float32)
features = scaler.fit_transform(features)

# Map mood quantiles to integer labels
quantile_mapping = {'Q1': 0, 'Q2': 1, 'Q3': 2}
target_quantiles = df['mood_quantiles'].map(quantile_mapping).values.astype(np.int64)



# Define new output size for classification
output_size = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]



# Modify the model's last layer for classification
class LSTMClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassification, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define hyperparameters
input_size = features.shape[1]
hidden_size = 64
num_layers = 3
learning_rate = 0.001
num_epochs = 30
batch_size = 64

# Split data using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
accuracies = []
for train_index, test_index in tscv.split(features):
    train_features, test_features = features[train_index], features[test_index]
    train_target, test_target = target_quantiles[train_index], target_quantiles[test_index]
    
    # Create DataLoader
    train_dataset = CustomDataset(train_features, train_target)
    test_dataset = CustomDataset(test_features, test_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = LSTMClassification(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    accuracies.append(accuracy)
    print(f'Test Accuracy: {accuracy:.4f}')

# Print average accuracy
print(f'Average Accuracy: {np.mean(accuracies):.4f}')

def predict(model, dataloader):
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return predictions, targets

# Make predictions on the test set
test_predictions, test_targets = predict(model, test_loader)
conf_matrix = confusion_matrix(test_targets, test_predictions)
# Plot real quantiles vs predicted quantiles
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(quantile_mapping.keys()), 
            yticklabels=list(quantile_mapping.keys()))
plt.title('Confusion Matrix')
plt.ylabel('Actual Quantiles')
plt.xlabel('Predicted Quantiles')
plt.show()