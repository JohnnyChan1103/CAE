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
import librosa
import numpy as np

AE = Autoencoder()
SAVE_PATH = r'D:\Develop\Code\CAE\trained.ae'
loss_func = nn.MSELoss()


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def test(test_iterator):
    AE.load_state_dict(torch.load(SAVE_PATH))
    test_loss = 0.0
    data = next(iter(test_iterator))
    clean, noisy = data
    clean = clean.to(device).unsqueeze(1).float()
    noisy = noisy.to(device).unsqueeze(1).float()
    encoded, decoded = AE(noisy)
    print(clean.shape, noisy.shape, decoded.shape)
    for i in range(10):
        Dc = librosa.amplitude_to_db(clean[i][0].detach().numpy(), ref=np.max)
        Dn = librosa.amplitude_to_db(noisy[i][0].detach().numpy(), ref=np.max)
        Dd = librosa.amplitude_to_db(decoded[i][0].detach().numpy(), ref=np.max)
        print('Figure '+str(i+1))
        print(Dc.shape)
        plt.subplot(10, 3, 3*i+1)
        librosa.display.specshow(Dc, sr=16000, hop_length=128, y_axis='log')
        plt.title('Clean frame '+str(i))
        plt.subplot(10, 3, 3*i+2)
        librosa.display.specshow(Dn, sr=16000, hop_length=128, y_axis='log')
        plt.title('Noisy frame '+str(i))
        plt.subplot(10, 3, 3*i+3)
        librosa.display.specshow(Dd, sr=16000, hop_length=128, y_axis='log')
        plt.title('Denoised frame '+str(i))
    plt.show()

if __name__ == '__main__':
    device = 'cpu'
    print(device)
    AE.to(device)
    test(test_iter)
