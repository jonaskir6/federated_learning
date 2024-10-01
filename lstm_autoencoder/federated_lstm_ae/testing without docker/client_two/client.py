import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from typing import OrderedDict
import sys
import lstm_ae, training

# How do I get the data? env variable from docker compose?
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

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = lstm_ae.LSTMAutoencoder(device, seq_len=trainX.shape[1], n_features=trainX.shape[2]).to(device)

# train.train(model, train_dl, device)

# sorted_anomalies = train.detect_anomalies_without_threshold(model, test_dl, device)
# print(f'Number of anomalies detected: {len(sorted_anomalies)}')
# for i, (anomaly, score) in enumerate(sorted_anomalies):
#     print(f'Anomaly {i+1}: Score = {score}')



#########

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        training.train(device, model, train_dl)
        return self.get_parameters(config={}), len(trainX), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        anomalies = training.detect_anomalies(model, test_dl, device, return_num_anomalies=True)
        # what is going to be returned? what is the format/score? Right now it tells you loss as anomalies (round one loss = anomalies)
        return float(anomalies), len(testX), {"accuracy:": 0.0}

# start flower client
fl.client.start_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient().to_client(),
        grpc_max_message_length=2 * 1024 * 1024 *1024 -1
)