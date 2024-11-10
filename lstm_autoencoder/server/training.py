import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import os
import numpy as np

factor = int(os.getenv('FACTOR'))

threshold = 0
factor = 3

def train(device, model, train_dl, n_epochs=2):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    epoch_losses = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for X, Y in train_dl:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, Y)
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


def detect(model, test_dl, device, supervised_mode=False):
    model.eval()
    criterion = nn.MSELoss()
    reconstruction_errors = []
    actual_data = []
    pred_data = []

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            reconstruction_error = loss.item()
            reconstruction_errors.append(reconstruction_error)

            actual_data.extend(y[:, -100:].cpu().numpy().flatten())
            pred_data.extend(output[:, -100:].cpu().numpy().flatten())


    actual_data = np.array(actual_data)
    pred_data = np.array(pred_data)
    counter = np.arange(1, len(actual_data) + 1)

    plt.figure(figsize=(14, 7))
    sns.lineplot(x=counter, y=actual_data, color='blue', label='Actual Data')
    sns.lineplot(x=counter, y=pred_data, color='red', label='Predicted Data', alpha=0.6)
    plt.xlabel('Counter')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Data Points')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(x=counter[:5000], y=actual_data[:5000], color='blue', label='Actual Data')
    sns.lineplot(x=counter[:5000], y=pred_data[:5000], color='red', label='Predicted Data')
    plt.xlabel('Counter')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Data Points')
    plt.legend()
    plt.grid(True)
    plt.show()


    if supervised_mode:
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        os.environ['threshold'] = str(mean_error + factor * std_error)
        return mean_error + factor * std_error
    else:
        anomalies = []
        for err in reconstruction_errors:
            if err > threshold:
                anomalies.append(err)
        return len(anomalies)