import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import os


data_path = 'all_data/client_0.csv'
data_sensor = 'dms1'
supervised_mode = True
threshold = 0
seq_size = 100
factor = 3
sensors = 8
offset = 10
train_loss_tot = 0

dataframe = pd.read_csv(data_path, delimiter=';')
df = dataframe[['counter', 'dms1', 'dms2', 'dms3', 'dms4', 'dms5', 'dms6', 'dms7', 'dms8']]

df = df.groupby(df.index // 10).agg({
    'counter': 'first',
    'dms1': 'mean',
    'dms2': 'mean',
    'dms3': 'mean',
    'dms4': 'mean',
    'dms5': 'mean',
    'dms6': 'mean',
    'dms7': 'mean',
    'dms8': 'mean'
}).reset_index(drop=True)

# df = df.iloc[1800:-1200]


# print("Start: ", df['counter'].min())
# print("End: ", df['counter'].max())

# print(f"Shape of df after slicing: {df.shape}")

# approx 80/20 split
train_test_ratio = 0.8
breakpoint = int(len(df) * train_test_ratio)
train, test = df.iloc[:breakpoint], df.iloc[breakpoint:]

# print(train.shape)

scaler = StandardScaler()
dms_sensors = [f"dms{i}" for i in range(1, sensors + 1)]
scaler = scaler.fit(train[dms_sensors])

train[dms_sensors] = scaler.transform(train[dms_sensors])
test[dms_sensors] = scaler.transform(test[dms_sensors])

def to_sequences(x, seq_size=1):
    x_values = []
    y_values = []
    # print('lenx: ', len(x))

    for i in range(len(x) - seq_size - offset):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(x.iloc[i + offset:(i + offset + seq_size)].values)
    return np.array(x_values), np.array(y_values)

num_train_sequences = len(train) - seq_size
num_test_sequences = len(test) - seq_size

# shape: (samples, seq_size, n_features)
# shape: (samples, seq_size, n_features)
trainX, trainY = [], []
testX, testY = [], []

for sensor in dms_sensors:
    trainX_tmp, trainY_tmp = to_sequences(train[sensor], seq_size)
    testX_tmp, testY_tmp = to_sequences(test[sensor], seq_size)
    trainX.append(trainX_tmp)
    trainY.append(trainY_tmp)
    testX.append(testX_tmp)
    testY.append(testY_tmp)

trainX = np.concatenate(trainX, axis=0).reshape(-1, seq_size, sensors)
trainY = np.concatenate(trainY, axis=0).reshape(-1, seq_size, sensors)
testX = np.concatenate(testX, axis=0).reshape(-1, seq_size, sensors)
testY = np.concatenate(testY, axis=0).reshape(-1, seq_size, sensors)

# print('trainX shape:', trainX.shape)
# print('trainY shape:', trainY.shape)
# print('testX shape:', testX.shape)
# print('testY shape:', testY.shape)

# print(f"Number of train sequences: {num_train_sequences}")
# print(f"Number of test sequences: {num_test_sequences}")


# print(trainX.shape)
# print(trainY.shape)

trainX = torch.tensor(trainX, dtype=torch.float32)
trainY = torch.tensor(trainY, dtype=torch.float32)
testX = torch.tensor(testX, dtype=torch.float32)
testY = torch.tensor(testY, dtype=torch.float32)

train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=128):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            # dropout=0.5,
            batch_first=True
        )

    def forward(self, x):
        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=128, n_features=3):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=2,
            # dropout=0.5,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.dense_layers = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)
        x = self.dense_layers(x)
        return x

class LSTMAutoencoder(nn.Module):
    def __init__(self, device, seq_len, n_features, embedding_dim=128):
        super(LSTMAutoencoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # print(x.shape)
        return x

def train(device, model, train_dl, n_epochs=100):

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
        # print(f'Epoch {epoch+1}, Loss: {train_loss}')

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

    train_loss_tot =  epoch_losses[-1]


def get_train_loss():
    return train_loss_tot


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(device, seq_len=trainX.shape[1], n_features=trainX.shape[2]).to(device)

    train(device, model, train_dl)

    torch.save(model.state_dict(), "results/models/model_1.pth")
