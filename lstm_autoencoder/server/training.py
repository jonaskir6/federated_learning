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

def train(device, model, train_dl, n_epochs=5):

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
            # print("out: ", output.shape)
            # print("y: ", y.shape)
            loss = criterion(output, y)
            reconstruction_error = loss.item()
            reconstruction_errors.append(reconstruction_error)

            actual_data.append(y.cpu().numpy())
            pred_data.append(output.cpu().numpy())

    actual_data = np.concatenate(actual_data, axis=0)
    pred_data = np.concatenate(pred_data, axis=0)
    counter = np.arange(1, actual_data.shape[0] + 1)

    # print("ad: ", actual_data.shape)
    # print("pd: ", pred_data.shape)

    actual_data = actual_data[-50:]
    pred_data = pred_data[-50:]

    x = np.arange(1, len(reconstruction_errors) + 1)

    plt.hist(reconstruction_errors, bins=50, color='blue')

    plt.figure(figsize=(14, 7))
    sns.lineplot(x=x, y=reconstruction_errors, color='blue', label=f'Reconstruction Error')
    plt.xlabel('Sequence')
    plt.ylabel('Reconstruction Error')
    plt.title(f'Reconstruction Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    # seq along the time axis
    actual_data_concat = actual_data.reshape(-1, actual_data.shape[2])
    pred_data_concat = pred_data.reshape(-1, pred_data.shape[2])

    counter = np.arange(actual_data_concat.shape[0])

    for feature in range(actual_data.shape[2]):
        plt.figure(figsize=(15, 6))
        plt.plot(counter, actual_data_concat[:, feature], color='green', alpha=1)
        plt.plot(counter, pred_data_concat[:, feature], color='red', alpha=0.6)
        plt.title(f'Feature {feature + 1} - All Sequences')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.legend(['Actual', 'Predicted'], loc='upper right')
        plt.show()

    # threshold = upper bound for 90% confidence interval
    if supervised_mode:
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        upper_bound = mean_error + 1.645 * std_error
        os.environ['threshold'] = str(upper_bound)
        return upper_bound
    else:
        anomalies = []
        for err in reconstruction_errors:
            if err > threshold:
                anomalies.append(err)
        return len(anomalies)
