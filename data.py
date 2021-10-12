import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torchvision.transforms import ToTensor

transform = transforms.Compose([ToTensor()])

train_set = torchvision.datasets.CIFAR10(root='./CIFAR10',
                                         download=True,
                                         train=True,
                                         transform=transform)

test_set = torchvision.datasets.CIFAR10(root='./CIFAR10',
                                         download=True,
                                         train=False,
                                         transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False)

if __name__ == "__main__":
    print("Train Set : {}|Test Set : {}".format(len(train_set), len(test_set)))
