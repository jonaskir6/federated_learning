from typing import List, OrderedDict
import flwr as fl
import sys
import numpy as np
import torch
import lstm_ae
import os

# need seq_len and n_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### Change to env ###
n_features = 4
seq_size = 100
### Change to env ###
model = lstm_ae.LSTMAutoencoder(device, seq_len=seq_size, n_features=n_features, output_dim=seq_size)

class SaveModelStrategy(fl.server.strategy.FedAvg):
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
    config = fl.server.ServerConfig(num_rounds=3),
    strategy = strategy,
    grpc_max_message_length=2 * 1024 * 1024 *1024 -1
)