import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from dataset import TimeSeriesDataset
from model import MultiChannelModel
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one epoch and return the average training loss.
    
    Parameters:
    - model: The neural network model to train.
    - train_loader: DataLoader for the training data.
    - criterion: Loss function.
    - optimizer: Optimizer for updating model parameters.
    
    Returns:
    - avg_train_loss: Average training loss for the epoch.
    """
    model.train()
    train_loss = 0
    for context, target in train_loader:
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss


def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the model on the given data loader and return the average loss.
    
    Parameters:
    - model: The neural network model to evaluate.
    - data_loader: DataLoader for the validation or test data.
    - criterion: Loss function.
    
    Returns:
    - avg_loss: Average loss over the dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for context, target in data_loader:
            output = model(context)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train the model over a specified number of epochs and return training and validation losses.
    
    Parameters:
    - model: The neural network model to train.
    - train_loader: DataLoader for the training data.
    - val_loader: DataLoader for the validation data.
    - criterion: Loss function.
    - optimizer: Optimizer for updating model parameters.
    - num_epochs: Number of epochs to train the model.
    
    Returns:
    - train_losses: List of average training losses for each epoch.
    - val_losses: List of average validation losses for each epoch.
    """
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        avg_val_loss = evaluate_model(model, val_loader, criterion)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses


# Plotting function
def plot_losses(train_losses, val_losses, save_str):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_str)

def generate_predictions(model, data_loader, context_length, horizon):
    model.eval()
    predictions_list = []
    with torch.no_grad():
        print('Generating predictions')
        for i, (context, target) in enumerate(tqdm(data_loader)):
            output = model(context)
            # Get the first value predicted from every sliding window
            # predictions_list.append(output[:,:,0].unsqueeze(2))
            if i%horizon==0:
                predictions_list.append(output)
    predictions = torch.cat(predictions_list, dim=2)
    return predictions

# Parameters
context_length = 1440  # Length of the historical data to consider
horizon = 360         # How far ahead we want to predict
batch_size = 64
num_epochs = 5
learning_rate = 0.005

# Load data
data_path = 'data/data.csv'  # Replace with the path to your CSV file
data = pd.read_csv(data_path)
colnames = list(data.columns)
# target_columns = ['0_requests', '8_requests', '13_requests', '13_cpu', '13_pods']
target_columns = ['13_requests', '13_cpu', '13_pods']
colnames = [item for item in colnames if item in target_columns]

data = data[colnames]

data_values = data.values

# data_values = data.iloc[:, 1:].values  # Assuming first column is time, skip it for values

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Prepare data
dataset = TimeSeriesDataset(data_values, context_length, horizon)
total_size = len(dataset)

# Define sizes for each split
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size  # Ensures it adds up to total_size

# Split dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed)
)

# DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Model, loss, optimizer
input_size = data_values.shape[1]
model = MultiChannelModel(input_size, context_length, horizon)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
plot_losses(train_losses, val_losses, 'losses')


avg_test_loss = evaluate_model(model, test_loader, criterion)
print(avg_test_loss)

# Generate predictions on the test set
# test_data = data_values[-(len(test_loader.dataset) + context_length + horizon):]  # Adjust to get enough data
predictions = generate_predictions(model, test_loader, context_length, horizon)
predictions = predictions[:,:,:test_size]

print('Plotting')

num_series = data_values.shape[1]

fig, ax = plt.subplots(num_series,1,figsize=(15,8))

time_arr = np.arange(0,data_values.shape[0])

for i in range(num_series):

    ax[i].scatter(time_arr[:train_size], data_values[:train_size,i],s=1,c='b', label='train')
    ax[i].scatter(time_arr[train_size:train_size+val_size], data_values[train_size:train_size+val_size,i],s=1,c='g', label='val')
    ax[i].scatter(time_arr[train_size+val_size:], data_values[train_size+val_size:,i],s=1,c='r', label='test')
    
    # ax[i].scatter(time_arr[-test_size:], predictions[:,i],s=1,c='purple', label='test_pred')
    ax[i].scatter(time_arr[-test_size:], predictions[0,i,:],s=1,c='purple', label='test_pred')
    # ax[i].set_title()

    plt.legend(loc='upper left')

    save_str = 'results.png'
    plt.tight_layout()
    plt.savefig(save_str)

# Adjust predictions to match the shape of the original target
# predictions_reshaped = predictions.reshape(-1, horizon, predictions.shape[2])  # Shape: (num_samples, horizon, input_size)

# # Convert predictions to DataFrame with the same columns
# predictions_df = pd.DataFrame(predictions_reshaped.reshape(-1, predictions_reshaped.shape[2]), columns=data.columns[1:])  # Exclude time column
# predictions_df.insert(0, 'time', data['time'].iloc[context_length + horizon:context_length + horizon + len(predictions_df)])  # Add time column

# # Save predictions
# predictions_df.to_csv('predictions.csv', index=False)
# print("Predictions saved to predictions.csv")

