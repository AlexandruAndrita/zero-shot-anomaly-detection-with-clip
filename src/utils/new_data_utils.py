from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import torch
from PIL import Image
import glob

# Enhanced transforms
normal_transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))
])

class MVTecDataset(Dataset):
    """Enhanced MVTec dataset with proper anomaly labeling"""
    
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        categories = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        
        for category in categories:
            cat_path = os.path.join(root, category, split)
            
            if not os.path.exists(cat_path):
                continue
                
            if split == 'train':
                # Only normal images in training
                good_path = os.path.join(cat_path, 'good')
                if os.path.exists(good_path):
                    for img_path in glob.glob(os.path.join(good_path, '*.png')):
                        self.images.append(img_path)
                        self.labels.append(0)  # Normal = 0
            else:
                # Both normal and anomalous in test
                good_path = os.path.join(cat_path, 'good')
                if os.path.exists(good_path):
                    for img_path in glob.glob(os.path.join(good_path, '*.png')):
                        self.images.append(img_path)
                        self.labels.append(0)  # Normal = 0
                
                # Anomalous images
                for anomaly_type in os.listdir(cat_path):
                    if anomaly_type == 'good':
                        continue
                    anomaly_path = os.path.join(cat_path, anomaly_type)
                    if os.path.isdir(anomaly_path):
                        for img_path in glob.glob(os.path.join(anomaly_path, '*.png')):
                            self.images.append(img_path)
                            self.labels.append(1)  # Anomaly = 1
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def mvtec_train_loader(root='data', split='train', batch_size=8):
    """Enhanced MVTec train loader with proper anomaly handling"""
    dataset = MVTecDataset(root, split, normal_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def mvtec_test_loader(root='data', split='test', batch_size=8):
    """Enhanced MVTec test loader with proper anomaly handling"""
    dataset = MVTecDataset(root, split, normal_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
