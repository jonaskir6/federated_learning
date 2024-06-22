from typing import List, OrderedDict
import flwr as fl
import sys
import numpy as np
import torch
import network, train


model = network.MADE(num_layer=4, num_units=4000, input_feat=784, ordering=range(1,785))

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self, rnd, results, failures
    ):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            samples = train.sample(model, num_samples=100, device='cuda')
            train.save_samples(samples, f"model_samples/model_{rnd}_samples.png")

            # Save the model
            torch.save(model.state_dict(), f"model_round_{rnd}.pth")    

        return aggregated_parameters, aggregated_metrics
    

strategy = SaveModelStrategy()

fl.server.start_server(
    server_address = 'localhost:' + str(sys.argv[1]),
    config = fl.server.ServerConfig(num_rounds=7),
    strategy = strategy,
    grpc_max_message_length=2 * 1024 * 1024 *1024 -1
)