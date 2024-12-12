# Federated Learning with MADE and LSTM-Autoencoders

## Requirements

- nvidia toolkit
- docker with nvidia runtime


## MADE
### Custom Settings (missing)

### Federated Learning Tests

This project uses PyTorch and Flower to implement Federated Learning on the MNIST dataset, utilizing the Masked Autoencoder for Distribution Estimation (MADE) model. For more details, refer to the [original paper](https://arxiv.org/abs/1502.03509).

### Running the Project

Navigate to the `fedavg_made` folder and execute the following command to start the services:

```sh
docker-compose up -d
```

- You can find the samples in the client and server folders
- Your final aggregated model will be called model_round{num_epochs}.pth

## LSTM-Autoencoder for Time Series Data
### Custom Settings (missing)

### Running the Project

For example vaulues and data look at the example lstm_ae.ipynb

Add your data to the `lstm_autoencoder/client/data` folder (right now it will train all on the same data, custom data for each container feature missing)

Navigate to the `fedavg_made` folder and execute the following command to start the services:

```sh
docker-compose up -d
```

- You can find the samples in `lstm_autoencoder/server/models`
- Your final aggregated model will be called model_round{num_epochs}.pth
