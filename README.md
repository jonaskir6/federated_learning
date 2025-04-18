# Federated Learning Experiments for the Fraunhofer CCIT Edge Cloud Project

Using LSTM-Autoencoders and Federated Learning to train models on industry time series data

## Requirements

- run on linux
- nvidia toolkit
- docker with nvidia runtime
- python3

## LSTM-Autoencoder for Time Series Data
### Running the Project

For example vaulues and data look at the example at `lstm_autoencoder/base/lstm_ae.ipynb`

Add your data folder to the `experiments` folder and folder and name the data files "client_x.csv" for all clients "x" you want to specify (Warning: Number of data files needs to be at least the number of clients)

Specify your data source path and number of clients in the `lstm_autoencoder/experiments/config.json` file

Navigate to the `lstm_autoencoder/experiments` folder and execute the following command to start the services:

```sh
python3 script.py
```

- You can find the final models in `lstm_autoencoder/experiments/results/models`
