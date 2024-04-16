import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class STFTDataset(Dataset):

    def __init__(self, clean_wav_dir, noisy_wav_dir, file_index=0, samples=128, overlap=64):
        self.file_index = file_index
        self.samples = samples
        self.overlap = overlap
        self.total_frames = 0

        # 读文件夹下每个数据文件的名称
        self.clean_file_name = os.listdir(clean_wav_dir)
        self.noisy_file_name = os.listdir(noisy_wav_dir)

        self.clean_data_path = []
        self.noisy_data_path = []
        # 把每一个文件的路径拼接起来
        for index in range(len(self.clean_file_name)):
            self.clean_data_path.append(os.path.join(clean_wav_dir, self.clean_file_name[index]))
        for index in range(len(self.noisy_file_name)):
            self.noisy_data_path.append(os.path.join(noisy_wav_dir, self.noisy_file_name[index]))
        # 文件读入缓存，迭代时直接取用
        self.clean_file_cache = pd.read_csv(self.clean_data_path[self.file_index], header=None)
        self.noisy_file_cache = pd.read_csv(self.noisy_data_path[self.file_index], header=None)
        self.clean_file_cache = torch.tensor(self.clean_file_cache.values)
        self.noisy_file_cache = torch.tensor(self.noisy_file_cache.values)

    def __len__(self):
        # return len(self.clean_file_name)
        return 8192

    def __getitem__(self, frame_index):
        if (frame_index - self.total_frames) * (
                self.samples - self.overlap) + self.samples <= self.noisy_file_cache.size(dim=1):
            clean_data = self.clean_file_cache[0:128, (frame_index - self.total_frames) * (self.samples - self.overlap):
                                                  (frame_index - self.total_frames) * (
                                                          self.samples - self.overlap) + self.samples]
            noisy_data = self.noisy_file_cache[0:128, (frame_index - self.total_frames) * (self.samples - self.overlap):
                                                  (frame_index - self.total_frames) * (
                                                          self.samples - self.overlap) + self.samples]
        else:
            clean_data = self.clean_file_cache[0:128,
                         self.clean_file_cache.size(dim=1) - self.samples:self.clean_file_cache.size(dim=1)]
            noisy_data = self.noisy_file_cache[0:128,
                         self.noisy_file_cache.size(dim=1) - self.samples:self.noisy_file_cache.size(dim=1)]
            self.total_frames = frame_index + 1
            self.file_index += 1
            self.clean_file_cache = pd.read_csv(self.clean_data_path[self.file_index], header=None)
            self.noisy_file_cache = pd.read_csv(self.noisy_data_path[self.file_index], header=None)
            self.clean_file_cache = torch.tensor(self.clean_file_cache.values)
            self.noisy_file_cache = torch.tensor(self.noisy_file_cache.values)
        return clean_data, noisy_data


clean_train_dir = r"D:\Develop\Dataset\TRAINSET_CLEAN_STFT\Amp"
noisy_train_dir = r"D:\Develop\Dataset\TRAINSET_NOISY_STFT\Amp"

# 读取数据集
train_dataset = STFTDataset(clean_wav_dir=clean_train_dir, noisy_wav_dir=noisy_train_dir)
# 加载数据集
train_iter = DataLoader(train_dataset, batch_size=256, pin_memory=True, prefetch_factor=True, num_workers=8)

clean_test_dir = r"D:\Develop\Dataset\TESTSET_CLEAN_STFT\Amp"
noisy_test_dir = r"D:\Develop\Dataset\TESTSET_NOISY_STFT\Amp"

# 读取测试集
test_dataset = STFTDataset(clean_wav_dir=clean_test_dir, noisy_wav_dir=noisy_test_dir, overlap=0)
# 加载测试集
test_iter = DataLoader(train_dataset, batch_size=10, pin_memory=True, prefetch_factor=True, num_workers=8)

#调试用
'''
for _, data in enumerate(train_iter):
    print('Iteration {}'.format(_+1))
    cd, nd = data
    #cd = torch.tensor(cd)
    #nd = torch.tensor(nd)
    print(type(cd), type(nd))
    print(cd.shape, nd.shape)
    #print(cd[0])
    #plt.imshow(20*nd[0].numpy(), cmap=plt.cm.gray)
    #plt.show()
    #print(type(x))
    pass
'''
