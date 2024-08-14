import os

import torch
import torchvision.transforms as transforms

from golden_retrieval.dataloader.traindataset import TupleDataset

# TODO: args parser
## args
sfm120k_root = '/home/jo/develop/golden_retrieval/dataset'
mode = 'train'
dataset_name = 'retrieval-SfM-120k'
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
batch_size = 32
num_workers = 4
b_pin_memory = True

nomalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([
    transforms.ToTensor(),
    nomalize
    ])
train_dataset = TupleDataset(
    sfm120k_root, mode, dataset_name, transform=transform
    )
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=num_workers,
    pin_memory=True, 
    )
