import time
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from torch.optim import Adam, AdamW, RMSprop, SGD 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def evaluate(test_data_loader, model):
    num_correct = 0
    num_total = 0

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

  # Disable gradient computation.
    with torch.no_grad():
        # Switch to evaluation mode.
        model.eval()
        loss=[]
        # Iterate over the entire test dataset.
        for images, labels in test_data_loader:
       
            images = images.to(device)
            output = model(images)
            loss.append(nn.functional.binary_cross_entropy(output,images))

    loss=torch.Tensor(loss)   
    return torch.mean(loss)

def sample(model, num_samples=100, device='cuda'):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    image_size = 28*28 # MNIST dataset image size
    
    with torch.no_grad():  # Disable gradient computation
        samples = torch.zeros((num_samples, image_size), device=device)
        for i in range(image_size):
            logits = model(samples)  # Get the logits for the current samples
            probs = logits[:, i]  # Convert logits to probabilities (already done in netwoks.py as a layer)
            samples[:, i] = torch.bernoulli(probs)  # Sample from the Bernoulli distribution

    return samples.cpu().numpy().reshape(-1, 28, 28) # Reshape to (num_samples, 28, 28)

def save_samples(samples, filename='samples.png'):
    # Get the number of samples
    num_samples = samples.shape[0]

    if(num_samples<=5):
        grid_size_1 = 1
        grid_size_2 = num_samples
    else:
        # We use the ceiling of the square root of the number of samples to ensure all samples fit
        grid_size_1 = int(np.ceil(np.sqrt(num_samples)))
        grid_size_2 = grid_size_1
    
    fig, axs = plt.subplots(grid_size_1, grid_size_2, figsize=(28, 28))
    axs = axs.flatten()

    for img, ax in zip(samples, axs):
        ax.imshow(img, cmap='gray', interpolation='none')
        ax.axis('off')

    # Turn off remaining empty subplots
    for i in range(num_samples, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def train_model(model, training_data, epochs):
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    print('=========================================')

    model.to(device)

    learning_rates = []
    train_loss_curve = []
    test_loss_curve = []
    train_loss_epochs = []
    test_loss_epochs = []


    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)

    overall_start_time = time.time()

    # training loop
    for epoch in range(epochs):

        print("Starting epoch")

        epoch_start_time = time.time()
        # Switch to training mode.
        model.train()

        losses = []
        batch_idx = 0

        for images, labels in training_data:

            images = images.to(device)

            output = model(images)
            loss = nn.functional.binary_cross_entropy(output, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
                optimizer.zero_grad(set_to_none=True)

                losses.append(loss.detach().clone())

                if batch_idx % 100 == 0:
                    average_loss = torch.stack(losses).mean().item()
                    train_loss_curve.append(average_loss)
                    train_loss_epochs.append(epoch + 1)
                    losses = []
                    # print(f'Epoch: {epoch + 1:3d}/{7:3d}, Batch {batch_idx + 1:5d}, Loss: {average_loss:.4f}')
                    batch_idx += 1

        scheduler.step()
        epoch_end_time = time.time()
        # print('-----------------------------------------')
        # print(f'Epoch: {epoch + 1:3d} took {epoch_end_time - epoch_start_time:.2f}s')
        # test_loss = evaluate(model=model, test_data_loader=test_data, device=device)
        # test_loss_curve.append(test_loss)
        # test_loss_epochs.append(epoch + 1)
        # print(f'Epoch: {epoch + 1:3d}, Test Loss: {test_loss:.4f}')
        # print('-----------------------------------------')
    

    samples = sample(model, num_samples=100)
    save_samples(samples)

    # overall_end_time = time.time()
    # print('=========================================')
    # print(f'Training took {overall_end_time - overall_start_time:.2f}s')

    # # Loss Curve Plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_loss_epochs, train_loss_curve, label='Train Loss')
    # plt.scatter(test_loss_epochs, test_loss_curve, color='red', label='Test Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.legend()
    # plt.show()

    # # Learning Rate Plot
    # num_batches = len(training_data)
    # learning_rates_res = [sum(learning_rates[i * num_batches:(i + 1) * num_batches]) / num_batches for i in range(num_epochs)]
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(num_epochs), learning_rates_res)
    # plt.xlabel('Epochs')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate over Time')
    # plt.show()