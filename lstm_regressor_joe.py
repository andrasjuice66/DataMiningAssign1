import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Load the dataset from CSV
df = pd.read_csv("out.csv")

# Drop non-numeric columns
df.sort_values("date", inplace=True)
non_numeric_columns = ['id', 'date', 'mood_quantiles','screen', 'Unnamed: 0',]
df.drop(columns=non_numeric_columns, inplace=True)

# Shift the mood column forward by 1

# Drop NaN values resulting from shifting
df.dropna(inplace=True)
target = df['mood'].values.astype(np.float32)
df['mood'] = df['mood'].shift(1)

# Define features and target
features = df.drop(columns=['mood']).values.astype(np.float32)



# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Define PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
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
hidden_size = 128
num_layers = 3
output_size = 1
learning_rate = 0.001
num_epochs = 30
batch_size = 128

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_train_losses = []
all_test_losses = []
fold_losses = []
# Split data using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_index, test_index) in enumerate(tscv.split(features)):
    print(f"Starting fold {fold+1}")
    train_losses = []
    test_losses = []
    train_features, test_features = features[train_index], features[test_index]
    train_target, test_target = target[train_index], target[test_index]

    
    # Create DataLoader
    train_dataset = CustomDataset(train_features, train_target)
    test_dataset = CustomDataset(test_features, test_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        all_train_losses.append(train_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')
    
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs.squeeze(), labels)
                test_loss += loss.item() * inputs.size(0)
            test_loss /= len(test_loader.dataset)
            all_test_losses.append(test_loss)

        
    
    fold_losses.append((train_losses, test_losses))
print(f'Test Loss: {test_loss:.4f}')
    # Plot losses for this fold


# Function to make predictions
def predict(model, dataloader):
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1))
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return predictions, targets

# Make predictions on the test set
test_predictions, test_targets = predict(model, test_loader)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(test_targets, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Mood')
plt.title('Predicted vs Actual Mood')
plt.legend()
plt.show()


# Plot training loss vs test loss over epochs
epochs_per_fold = num_epochs

# Plot losses for each fold
for fold in range(tscv.n_splits):
    fold_start = fold * epochs_per_fold
    fold_end = (fold + 1) * epochs_per_fold

    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses[fold_start:fold_end], label='Training Loss')
    plt.plot(all_test_losses[fold_start:fold_end], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Test Loss During Fold {fold + 1}')
    plt.legend()
    plt.show()