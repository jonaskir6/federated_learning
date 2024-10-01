import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import lstm_ae


def train(device, model, train_dl, n_epochs=10):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch_losses = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_dl:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl) 
        epoch_losses.append(train_loss) 
        print(f'Epoch {epoch+1}, Loss: {train_loss}')

    # training loss
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, epoch_losses, marker='o', label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.show()


# save table with anomalies in csv? or show as plot?
def detect_anomalies_without_threshold(model, test_dl, device, return_num_anomalies=False):
    model.eval()
    criterion = nn.MSELoss()
    anomalies = []
    scores = []
    reconstruction_errors = []

    with torch.no_grad():
        for batch_X, batch_Y in test_dl:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            output = model(batch_X)
            loss = criterion(output, batch_Y)
            reconstruction_error = loss.item()
            reconstruction_errors.append(reconstruction_error)

            anomalies.append((batch_X.cpu().numpy(), output.cpu().numpy(), reconstruction_error))

    # normalize based on the max reconstruction error (0-100)
    max_error = max(reconstruction_errors)
    for error in reconstruction_errors:
        score = (error / max_error) * 100 
        scores.append(score)

    # sort by reconstruction error (desc)
    sorted_anomalies = sorted(zip(anomalies, scores), key=lambda x: x[1], reverse=True)

    # for federated learning
    if return_num_anomalies:
        # maybe look where the biggest gap is?
        sorted_anomalies = [anomaly for anomaly, score in sorted_anomalies if score > 25]
        return len(sorted_anomalies)
    
    else:
        return sorted_anomalies

