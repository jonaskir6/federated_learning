import flwr as fl
from torchvision import datasets, transforms
import numpy as np
import network, train
import torch
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# from MNIST repo
class Binarize():
    def __call__(self, tensor):
        return (tensor > 0.5).float()
    
# Print distribution of data classes
def getDist(y):
    ax = sns.countplot(y)
    ax.set(title="Count of data classes")
    plt.savefig("count.png")
    
# Get data based on distribution
def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]]<dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
        
    return np.array(dx), np.array(dy)

model = network.MADE(num_layer=4, num_units=4000, input_feat=784, ordering=range(1,785))

# Binarize MNIST
transform = transforms.Compose([transforms.ToTensor(), 
                Binarize(), 
                torch.nn.Flatten(0)])

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, transform=transform)


# ### Convert to numpy arrays for getData function ###
train_x_data = np.array([train_data[i][0].numpy() for i in range(len(train_data))])
train_y_data = np.array([train_data[i][1] for i in range(len(train_data))])

dist = [4000, 10, 3000, 30, 4500, 70, 3500, 20, 5000, 10]

train_x, train_y = getData(dist, train_x_data, train_y_data)

getDist(train_y)
# ### Convert to numpy arrays for getData function ###

filtered_train_x_tensor = torch.tensor(train_x).float()
filtered_train_y_tensor = torch.tensor(train_y).long()

filtered_train_dataset = TensorDataset(filtered_train_x_tensor, filtered_train_y_tensor)

train_dl = DataLoader(filtered_train_dataset, batch_size=64, shuffle=True)
test_dl = DataLoader(test_data, batch_size=64, shuffle=False)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.state_dict()

    # returns weights of MNIST netowrk after training
    def fit(self):
        train.train_model(model, train_dl, test_dl)
        return model.state_dict(), len(train_x), {}
    
    def evaluate(self):
        return train.evaluate(test_dl, model)

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 4 * 1024 * 1024 *1024
)