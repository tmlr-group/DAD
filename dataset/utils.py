from torch.utils.data import Dataset
import random

class CustomDataset(Dataset):
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
class CustomImageNetSampler():
    def __init__(self, dataset, sample_per_class):
        self.dataset = dataset
        self.sample_per_class = sample_per_class

    def get_sampled_indices(self):
        # Find indices for sampling
        indices_per_class = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in indices_per_class:
                indices_per_class[label] = []
            indices_per_class[label].append(idx)

        # Randomly sample 'sample_per_class' indices from each class
        sampled_indices = []
        for label, indices in indices_per_class.items():
            sampled_indices.extend(random.sample(indices, min(self.sample_per_class, len(indices))))

        return sampled_indices