import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn


# 定义自编码器结构
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(1, 15), stride=7  # 16*1*115
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=16, out_channels=8, kernel_size=(1, 15), stride=7  # 8*1*101
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8, out_channels=16, kernel_size=(1, 15), stride=7  # 16*1*115
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=16, kernel_size=(1, 15), stride=7    # 16*1*129
            ),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=1, kernel_size=1  # 1*1*129
            ),
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded
