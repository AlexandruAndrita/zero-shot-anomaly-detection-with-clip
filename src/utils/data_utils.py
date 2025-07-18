from torchvision import datasets, transforms
from torch.utils.data import DataLoader

normal_transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize((0.4814,0.4578,0.4082),
                                                (0.2686,0.2613,0.2758))
])

def mvtec_loader(root='data/mvtec', cls='transistor', split='train', bs=8):
    path = f"{root}/{cls}/{split}"
    ds = datasets.ImageFolder(path, normal_transform)
    return DataLoader(ds, batch_size=bs, shuffle=True)
