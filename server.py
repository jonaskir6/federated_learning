import flwr as fl
import sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self, rnd, results, failures
    ):
        weights = super().aggregate_fit(rnd, results, failures)

        if weights is not None:
            print(f"Saving model to model_{rnd}.npz")
            np.savez(f"model_{rnd}.npz", *weights)

        return weights
    

strategy = SaveModelStrategy()

fl.server.start_server(
    server_address = 'localhost:' + str(sys.argv[1]),
    config = fl.server.ServerConfig(num_rounds=3),
    grpc_max_message_length = 4 * 1024 * 1024,
    strategy = strategy
)