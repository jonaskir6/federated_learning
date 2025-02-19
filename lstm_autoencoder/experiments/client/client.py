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
# data_sensor = 'dms1'
supervised_mode = True
threshhold = 0
seq_size = 100
offset = 10
sensors = 8

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

df = df.iloc[1800:-1200]


print("Start: ", df['counter'].min())
print("End: ", df['counter'].max())

print(f"Shape of df after slicing: {df.shape}")

# approx 80/20 split
train_test_ratio = 0.8
breakpoint = int(len(df) * train_test_ratio)
train, test = df.iloc[:breakpoint], df.iloc[breakpoint:]

print(train.shape)

scaler = StandardScaler()
dms_sensors = [f"dms{i}" for i in range(1, sensors + 1)]
scaler = scaler.fit(train[dms_sensors])

train[dms_sensors] = scaler.transform(train[dms_sensors])
test[dms_sensors] = scaler.transform(test[dms_sensors])

def to_sequences(x, seq_size=1):
    x_values = []
    y_values = []
    print('lenx: ', len(x))

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

print('trainX shape:', trainX.shape)
print('trainY shape:', trainY.shape)
print('testX shape:', testX.shape)
print('testY shape:', testY.shape)

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

train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = lstm_ae.LSTMAutoencoder(device, seq_len=trainX.shape[1], n_features=trainX.shape[2]).to(device)


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
        loss = training.evaluate(model, test_dl, device)
        return float(loss), len(testX), {"Loss": loss}
    
        # evaluate threshold instead of loss:
        # value = training.detect(model, test_dl, device, supervised_mode=supervised_mode)
        # if supervised_mode:
        #     return float(value), len(testX), {"Threshold:": value} 
        # else:  
        #     return float(value), len(testX), {"Number of anomalies:": value}

# start flower client
fl.client.start_client(
        server_address="server:5002", 
        client=FlowerClient().to_client(),
        grpc_max_message_length=2 * 1024 * 1024 *1024 -1
)