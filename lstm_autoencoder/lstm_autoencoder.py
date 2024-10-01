import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


dataframe = pd.read_csv('data/fl_data.csv', delimiter=';')
df = dataframe[['counter', 'dms1']]

df = df.groupby(df.index // 10).agg({
    'counter': 'first',
    'dms1': 'mean',
}).reset_index(drop=True)

sns.lineplot(x=df['counter'], y=df['dms1'])

print("Start: ", df['counter'].min())
print("End: ", df['counter'].max())

train, test = df.loc[df['counter'] <= 140000], df.loc[df['counter'] > 140000]

print(train.shape)

scaler = StandardScaler()
scaler = scaler.fit(train[['dms1']])

train[['dms1']] = scaler.transform(train[['dms1']])
test[['dms1']] = scaler.transform(test[['dms1']])

seq_size=500

def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

# shape: (samples, seq_size, n_features)
trainX, trainY = to_sequences(train[['dms1']], train['dms1'], seq_size)
# shape: (samples, n_features)
testX, testY = to_sequences(test[['dms1']], test['dms1'], seq_size)

# print(trainX.shape)
# print(trainY.shape)

trainX = torch.tensor(trainX, dtype=torch.float32)
trainY = torch.tensor(trainY, dtype=torch.float32)
testX = torch.tensor(testX, dtype=torch.float32)
testY = torch.tensor(testY, dtype=torch.float32)

train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
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
        #print('enc:', x.shape)

        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)

        return x

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
        #print('dec1:', x.shape)
        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)
        x = x[:,-1,:]
        #print('dec2:', x.shape)

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
        #print('LSTMAE: ', x.shape)

        return x

model = LSTMAutoencoder(seq_len=trainX.shape[1], n_features=trainX.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_dl, n_epochs=10):
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


train(model, train_loader)


# save table with anomalies in csv? or show as plot?
def detect_anomalies_without_threshold(model, test_dl):
    model.eval()
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
    return sorted_anomalies

# example
sorted_anomalies = detect_anomalies_without_threshold(model, test_loader)
print(f'Number of anomalies detected: {len(sorted_anomalies)}')
for i, (anomaly, score) in enumerate(sorted_anomalies):
    print(f'Anomaly {i+1}: Score = {score}')