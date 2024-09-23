import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
dataframe = pd.read_csv('data/fl_data.csv')
df = dataframe[['counter', 'weg1', 'dms1']]
df['Time'] = pd.to_datetime(df['counter'])

sns.lineplot(x=df['Time'], y=df['dms1'])

print("Start: ", df['Time'].min())
print("End: ", df['Time'].max())

train, test = df.loc[df['Time'] <= '100000'], df.loc[df['Time'] > '100000']

scaler = StandardScaler()
scaler = scaler.fit(train[['dms1']])

train['dms1'] = scaler.transform(train[['dms1']])
test['dms1'] = scaler.transform(test[['dms1']])

seq_size = 100

def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

# Shape??
trainX, trainY = to_sequences(train[['dms1']], train['dms1'], seq_size)
testX, testY = to_sequences(test[['dms1']], test['dms1'], seq_size)

trainX = torch.tensor(trainX, dtype=torch.float32)
trainY = torch.tensor(trainY, dtype=torch.float32)
testX = torch.tensor(testX, dtype=torch.float32)
testY = torch.tensor(testY, dtype=torch.float32)

train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
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
          num_layers=1,
          batch_first=True
        )

    def forward(self, x):
        x = x.reshape(1, self.seq_len, self.n_features)

        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)

        return x, hidden_n

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, output_dim=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, output_dim

        self.lstm1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )

        self.lstm2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.dense_layers = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        x = x.repeat(self.seq_len, 1)
        x = x.reshape(1, self.seq_len, self.input_dim)

        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.dense_layers(x)
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

model = LSTMAutoencoder(seq_len=trainX.shape[1], n_features=trainX.shape[2])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train (model, train_ds, test_ds, n_epochs = 10):
    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {train_loss}')

    # Plot training loss
    plt.plot(range(n_epochs), train_loss, label='Training loss')
    plt.legend()

# Anomaly detection
model.eval()
with torch.no_grad():
    trainPredict = model(trainX).numpy()
    trainMAE = np.mean(np.abs(trainPredict - trainX.numpy()), axis=1)
    plt.hist(trainMAE, bins=30)
    max_trainMAE = 0.3

    testPredict = model(testX).numpy()
    testMAE = np.mean(np.abs(testPredict - testX.numpy()), axis=1)
    plt.hist(testMAE, bins=30)

anomaly_df = pd.DataFrame(test[seq_size:])
anomaly_df['testMAE'] = testMAE
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
anomaly_df['dms1'] = test[seq_size:]['dms1']

sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['testMAE'])
sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['max_trainMAE'])

anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

sns.lineplot(x=anomaly_df['Time'], y=scaler.inverse_transform(anomaly_df['dms1'].values.reshape(-1, 1)).flatten())
sns.scatterplot(x=anomalies['Time'], y=scaler.inverse_transform(anomalies['dms1'].values.reshape(-1, 1)).flatten(), color='r')