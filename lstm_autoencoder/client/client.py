import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from typing import OrderedDict
import os
import lstm_ae, training

# TODO input from console in script before running client and server for test, change index averaging, fix fluctuation and input data 

# Read env variables
# data_path = os.getenv('DATA_FILE')
# data_sensor = os.getenv('DATA_SENSOR')
# supervised_mode = os.getenv('SUPERVISED_MODE')
# target_rec_loss = float(os.getenv('TARGET_REC_LOSS'))
# seq_size = int(os.getenv('SEQ_SIZE'))

# Hardcoded for now
# data_path = input('Data Path:')
# data_sensor = input('Data Sensor:')
# supervised_mode = bool(input('Supervised Mode(True/False):'))
# seq_size = int(input('Sequence Size:'))
# offset = int(input('Prediction Offset:'))

data_path = 'fl_data.csv'
data_sensor = 'dms1'
supervised_mode = True
threshhold = 0
seq_size = 500
offset = 100

dataframe = pd.read_csv(data_path, delimiter=';')
df = dataframe[['counter', data_sensor]]

df = df.groupby(df.index // 10).agg({
    'counter': 'first',
    data_sensor: 'mean',
}).reset_index(drop=True)

df = df.iloc[1800:-1200]

sns.lineplot(x=df['counter'], y=df[data_sensor]) 

print("Start: ", df['counter'].min())
print("End: ", df['counter'].max())

print(f"Shape of df after slicing: {df.shape}")

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

    for i in range(len(x) - seq_size - offset):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(x.iloc[i + offset:(i + offset + seq_size)].values)
        
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = lstm_ae.LSTMAutoencoder(device, seq_len=trainX.shape[1], n_features=trainX.shape[2], output_dim=seq_size).to(device)


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
        value = training.detect(model, test_dl, device, supervised_mode=supervised_mode)
        # what is going to be returned? what is the format/score?
        if supervised_mode:
            return float(value), len(testX), {"Threshold:": value} 
        else:  
            return float(value), len(testX), {"Number of anomalies:": value}

# start flower client
fl.client.start_client(
        server_address="server:5002", 
        client=FlowerClient().to_client(),
        grpc_max_message_length=2 * 1024 * 1024 *1024 -1
)