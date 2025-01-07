from typing import List, OrderedDict
import flwr as fl
import sys
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import lstm_ae
import os
import training
import pandas as pd

# need seq_len and n_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'fl_data.csv'
### Change to env ###
n_features = 4
seq_size = 100
offset=10
### Change to env ###
model = lstm_ae.LSTMAutoencoder(device, seq_len=seq_size, n_features=n_features)

dataframe = pd.read_csv(data_path, delimiter=';')
df = dataframe[['counter', 'dms1', 'dms2', 'dms3', 'dms4']]

df = df.groupby(df.index // 10).agg({
    'counter': 'first',
    'dms1': 'mean',
    'dms2': 'mean',
    'dms3': 'mean',
    'dms4': 'mean'
}).reset_index(drop=True)

df = df.iloc[1800:-1200]


train_test_ratio = 0.8
breakpoint = int(len(df) * train_test_ratio)
test = df.iloc[breakpoint:]

scaler = StandardScaler()
dms_sensors = [f"dms{i}" for i in range(1, n_features + 1)]

test[dms_sensors] = scaler.transform(test[dms_sensors])

def to_sequences(x, seq_size=1):
    x_values = []
    y_values = []
    print('lenx: ', len(x))

    for i in range(len(x) - seq_size - offset):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(x.iloc[i + offset:(i + offset + seq_size)].values)
    return np.array(x_values), np.array(y_values)

num_test_sequences = len(test) - seq_size

# shape: (samples, seq_size, n_features)
# shape: (samples, seq_size, n_features)
trainX, trainY = [], []
testX, testY = [], []

for sensor in dms_sensors:
    testX_tmp, testY_tmp = to_sequences(test[sensor], seq_size)
    testX.append(testX_tmp)
    testY.append(testY_tmp)

testX = np.concatenate(testX, axis=0).reshape(-1, seq_size, n_features)
testY = np.concatenate(testY, axis=0).reshape(-1, seq_size, n_features)

testX = torch.tensor(testX, dtype=torch.float32)
testY = torch.tensor(testY, dtype=torch.float32)

test_dataset = TensorDataset(testX, testY)

test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)

def get_evaluate_fn(num_classes: int):

    def evaluate_fn(server_round: int, parameters, config):

        model = lstm_ae.LSTMAutoencoder(device, seq_len=seq_size, n_features=n_features)     
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        test_loss = training.evaluate(model, test_dl, device)

        return test_loss

    return evaluate_fn

class SaveModelStrategy(fl.server.strategy.FedAvg(
    evaluate_fn=get_evaluate_fn(num_classes=n_features)
)):
    def aggregate_fit(
        self, rnd, results, failures
    ):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")

            os.makedirs("models", exist_ok=True)

            # convert `parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            torch.save(model.state_dict(), f"models/model_round_{rnd}.pth")    

        return aggregated_parameters, aggregated_metrics
    


strategy = SaveModelStrategy()

fl.server.start_server(
    server_address = 'server:5002',
    config = fl.server.ServerConfig(num_rounds=4),
    strategy = strategy,
    grpc_max_message_length=2 * 1024 * 1024 *1024 -1
)