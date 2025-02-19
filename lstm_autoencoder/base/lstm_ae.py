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

# Fragen: 
# - n_features: Werden wir auch pro Modell mehrere Sensoren haben? Oder immer pro Modell einen Sensor => Implementationskomplexität

# - env variables
# - output dim
# - batch_size etc.
# - supervised mode, find avg reconstruction loss of model

# Read env variables
# data_path = os.getenv('DATA_FILE')
# data_sensor = os.getenv('DATA_SENSOR')
# supervised_mode = os.getenv('SUPERVISED_MODE')
# target_rec_loss = float(os.getenv('TARGET_REC_LOSS'))
# seq_size = int(os.getenv('SEQ_SIZE'))

data_path = 'data/fl_data.csv'
data_sensor = 'dms1'
supervised_mode = True
target_rec_loss = 0.20818762120509904
seq_size = 500

# How do I get the data? env variable from docker compose?
dataframe = pd.read_csv(data_path, delimiter=';')
df = dataframe[['counter', data_sensor]]

df = df.groupby(df.index // 10).agg({
    'counter': 'first',
    data_sensor: 'mean',
}).reset_index(drop=True)

sns.lineplot(x=df['counter'], y=df[data_sensor])

print("Start: ", df['counter'].min())
print("End: ", df['counter'].max())

# approx 80/20 split
train, test = df.loc[df['counter'] <= 178000], df.loc[df['counter'] > 178000]

# print(train.shape)

scaler = StandardScaler()
scaler = scaler.fit(train[[data_sensor]])

train[[data_sensor]] = scaler.transform(train[[data_sensor]])
test[[data_sensor]] = scaler.transform(test[[data_sensor]])

def to_sequences(x, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(x.iloc[i + 1:(i + 1 + seq_size)].values)
        
    return np.array(x_values), np.array(y_values)

# shape: (samples, seq_size, n_features)
trainX, trainY = to_sequences(train[[data_sensor]], seq_size)
# shape: (samples, n_features)
testX, testY = to_sequences(test[[data_sensor]], seq_size)

num_train_sequences = len(train) - seq_size
num_test_sequences = len(test) - seq_size

print(f"Number of train sequences: {num_train_sequences}")
print(f"Number of test sequences: {num_test_sequences}")

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
          dropout=0.5,
          batch_first=True
        )

    def forward(self, x):
        #print('enc:', x.shape)

        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=128, output_dim=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, output_dim

        self.lstm1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=2,
          dropout=0.5,
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
        #print('dec1:', x.shape)
        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)
        x = x[:,-1,:]
        #print('dec2:', x.shape)

        return self.dense_layers(x)
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, device, seq_len, n_features, embedding_dim=128, output_dim=1):
        super(LSTMAutoencoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, output_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.unsqueeze(-1)
        # print('LSTMAE: ', x.shape)

        return x
    

def train(device, model, train_dl, n_epochs=20):

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

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            reconstruction_error = loss.item()
            reconstruction_errors.append(reconstruction_error)

    if supervised_mode:
        return sum(reconstruction_errors) / len(reconstruction_errors)
    else:
        return max(reconstruction_errors) - target_rec_loss
    

# deprecated for now (function to calculate anomaly score based on reconstruction error)
# def detect_anomalies(model, test_dl, device, return_num_anomalies=False):
#     model.eval()
#     criterion = nn.MSELoss()
#     anomalies = []
#     scores = []
#     reconstruction_errors = []

#     with torch.no_grad():
#         for batch_X, batch_Y in test_dl:
#             batch_X = batch_X.to(device)
#             batch_Y = batch_Y.to(device)

#             output = model(batch_X)
#             loss = criterion(output, batch_Y)
#             reconstruction_error = loss.item()
#             reconstruction_errors.append(reconstruction_error)

#             anomalies.append((batch_X.cpu().numpy(), output.cpu().numpy(), reconstruction_error))

#     # normalize based on the max reconstruction error (0-100)
#     max_error = max(reconstruction_errors)
#     for error in reconstruction_errors:
#         score = (error / max_error) * 100 
#         scores.append(score)

#     # sort by reconstruction error (desc)
#     sorted_anomalies = sorted(zip(anomalies, scores), key=lambda x: x[1], reverse=True)

#     # for federated learning
#     if return_num_anomalies:
#         # maybe look where the biggest gap is?
#         sorted_anomalies = [anomaly for anomaly, score in sorted_anomalies if score > 25]
#         return len(sorted_anomalies)
    
#     else:
#         return sorted_anomalies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMAutoencoder(device, seq_len=trainX.shape[1], n_features=trainX.shape[2], output_dim=seq_size).to(device)

train(device, model, train_dl)

if supervised_mode:
    avg = detect(model, test_dl, device, supervised_mode)
    print(f'Average reconstruction error: {avg}')
else:
    anomaly_score = detect(model, test_dl, device, supervised_mode)
    print(f'Anomaly score: {anomaly_score}')
