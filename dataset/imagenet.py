from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import random
import pickle
import os
from utils import *

class ImageNet():
    def __init__(self, seed: int = 1, train_batch_size: int = 128, test_batch_size: int = 100, num_workers: int = 6):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
        self.num_workers = num_workers

    def transform_train(self):
        return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        ])

    def transform_test(self):
        return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
    
    def train_data(self):
        torch.manual_seed(self.seed)
        train_dataset = ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/train', 
                                    transform=self.transform_train())
        
        return DataLoader(train_dataset, 
                          batch_size=self.train_batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)

    def test_data(self):
        test_dataset = ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/val', 
                                   transform=self.transform_test())
        return DataLoader(test_dataset, 
                          batch_size=self.test_batch_size, 
                          shuffle=False,
                          num_workers=self.num_workers)
    
def main():

    train_dataset = ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/train', 
                                    transform=transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        ]))
    print('load data successfully!')
    sampler = CustomImageNetSampler(train_dataset, sample_per_class=100)
    print('initialize sampler successfully!')
    indices = sampler.get_sampled_indices()
    print('get sampled indices successfully!')
    random.shuffle(indices)
    if indices is not None:
        print("Number of sampled indices:", len(indices))
    else:
        print("Failed to get sampled indices.")
    directory = '../data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('../data/imagenet_indices.pkl', 'wb') as f:
        pickle.dump(indices, f)

    with open('../data/imagenet_indices.pkl', 'rb') as f:
        all_indices = pickle.load(f)
    assert isinstance(all_indices, list)

    selected_indices = random.sample(all_indices, 1000)

    remaining_indices = [idx for idx in all_indices if idx not in selected_indices]

    with open('../data/imagenet_mmd_indices.pkl', 'wb') as f:
        pickle.dump(selected_indices, f)

    with open('../data/imagenet_train_indices.pkl', 'wb') as f:
        pickle.dump(remaining_indices, f)

if __name__ == '__main__':
    main()