import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from STFTdataloader import train_iter
from STFTdataloader import test_iter
from autoencoder import Autoencoder
import matplotlib.pyplot as plt

AE = Autoencoder()
SAVE_PATH = r'D:\Develop\Code\CAE\trained.ae'
loss_func = nn.MSELoss()
lr = 0.01
optimizer = optim.Adam(AE.parameters(), lr=lr)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def train(train_iterator, epochs):
    train_loss = []
    for e in range(epochs):
        running_loss = 0.0
        for _, data in enumerate(train_iterator):
            clean, noisy = data
            clean = clean.to(device).unsqueeze(1).float()
            noisy = noisy.to(device).unsqueeze(1).float()
            encoded, decoded = AE(noisy)
            loss = loss_func(decoded, clean)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iteration {} of {}'.format(_ + 1, len(train_iterator)))

        loss = running_loss / len(train_iterator)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(e + 1, epochs, loss))
        if loss <= train_loss[-1]:
            torch.save(AE.state_dict(), SAVE_PATH)
            pass

    return train_loss


if __name__ == '__main__':
    device = get_device()
    print(device)
    AE.to(device)
    for m in AE.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = torch.nn.init.calculate_gain('relu'))
    train_loss = train(train_iter, epochs=10)
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
