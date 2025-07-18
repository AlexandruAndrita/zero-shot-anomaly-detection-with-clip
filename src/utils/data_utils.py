from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.utils.data import TensorDataset, DataLoader
import torch

normal_transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize((0.4814,0.4578,0.4082),
                                                (0.2686,0.2613,0.2758))
])

def mvtec_train_loader(root='data', split='train', batch_size=8):
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    all_imgs = []
    all_labels = []
    for idx, folder in enumerate(folders):
        path = os.path.join(root, folder, split)
        if os.path.isdir(path):
            ds = datasets.ImageFolder(path, normal_transform)
            for img, label in ds:
                all_imgs.append(img)
                all_labels.append(idx)
    dataset = TensorDataset(torch.stack(all_imgs), torch.tensor(all_labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def mvtec_test_loader(root='data', split='test', batch_size=8):
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    all_imgs = []
    all_labels = []
    for idx, folder in enumerate(folders):
        path = os.path.join(root, folder, split)
        if os.path.isdir(path):
            ds = datasets.ImageFolder(path, normal_transform)
            for img, label in ds:
                all_imgs.append(img)
                all_labels.append(idx)
    dataset = TensorDataset(torch.stack(all_imgs), torch.tensor(all_labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
