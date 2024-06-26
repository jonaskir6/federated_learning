# FedAvg with MADE
Federated Learning Tests with Pytorch &amp; Flower on MNIST Dataset with Masked Autoencoder for Distribution Estimation Model 

## Usage
- (Conda) Environment needs: torch, numpy, flwr, torchvision, sys, matplotlib
- In 3 different terminals start client_one, client_two, server (all on the same port):
```bash
./server {port}
./client_one {port}
./client_two {port}
```
- You can find the samples in the client and server folders
- Your final aggregated model will be called model_round{num_epochs}.pth
