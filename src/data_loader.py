import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

'''
    Dataset class 
'''

class Dataset(Dataset):

    def __init__(self, data, img_dir, alphabet, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        self.alphabet = alphabet
        pass


    def __len__(self) -> int:
        ''' Returns the total number of samples'''
        return len(self.data)

    def __getitem__(self, index):
        ''' Generates one sample of data'''

        # Create the images path
        img_path = os.path.join(self.img_dir, self.data.iloc[index,0])

        # Load image in grayscale
        image = Image.open(img_path).convert('L')

        # Read thelabels label
        label = self.data.iloc[index,1]
        
        # Apply transformations to image
        if self.transform:
            image, label = self.transform(image, label, self.alphabet)
        
        return transforms.ToTensor()(image), torch.tensor(label)


def get_dataloader(df, img_path, alphabet, transform=None, batch_size = 64):
    ''' Creates a dataset with the given df and transformation
        Returns a dataloader object with the given dataset'''
    dataset = Dataset(df, img_path, alphabet, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True)