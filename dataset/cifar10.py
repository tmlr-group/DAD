from torchvision.datasets import CIFAR10 as DATA
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import random
import os
import pickle
from utils import *

class CIFAR10():
    def __init__(self, seed: int = 1, train_batch_size: int = 512, test_batch_size: int = 100, path: str = './data/'):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.path = path
        self.seed = seed

    def transform_train(self):
        return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    def transform_test(self):
        return transforms.Compose([
        transforms.ToTensor(),
        ])
    
    def train_data(self):
        torch.manual_seed(self.seed)
        train_dataset = DATA(self.path, train=True, download=True, transform=self.transform_train())
        return DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def test_data(self):
        test_dataset = DATA(self.path, train=False, download=False, transform=self.transform_test())
        return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)


    
def main():
    train_dataset = DATA('../data/', train=True, download=True, 
                                         transform=transforms.Compose([
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             ]))
    print('load data successfully!')
    sampler = CustomImageNetSampler(train_dataset, sample_per_class=50)
    print('initialize sampler successfully!')
    mmd_indices = sampler.get_sampled_indices()
    print('get sampled indices successfully!')
    random.shuffle(mmd_indices)
    if mmd_indices is not None:
        print("Number of sampled indices:", len(mmd_indices))
    else:
        print("Failed to get sampled indices.")
    directory = '../data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('../data/cifar10_mmd_indices.pkl', 'wb') as f:
        pickle.dump(mmd_indices, f)

    all_indices = set(range(len(train_dataset))) 
    training_indices = list(all_indices - set(mmd_indices))
    random.shuffle(training_indices)
    with open('../data/cifar10_train_indices.pkl', 'wb') as f:
        pickle.dump(training_indices, f)

if __name__ == '__main__':
    main()