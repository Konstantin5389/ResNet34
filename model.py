import torch.nn as nn
import torch
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=1, kernel_size=3, bias=False)
    
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class Res34Net(nn.Module):

    def __init__(self, layers, block=BasicBlock, num_class=10):
        super(Res34Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self.make_layer(in_channels=64, out_channels=64, block=block, num_block=layers[0],stride=2)
        self.layer2 = self.make_layer(in_channels=64, out_channels=128, block=block, num_block=layers[1], stride=2)
        self.layer3 = self.make_layer(in_channels=128, out_channels=256, block=block, num_block=layers[2], stride=2)
        self.layer4 = self.make_layer(in_channels=256, out_channels=512, block=block, num_block=layers[3], stride=2)
        self.fc = nn.Linear(512, num_class)
    def make_layer(self, block, in_channels, out_channels, num_block, stride=1, downsample=None):
        if in_channels != out_channels or stride != 1:
            downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels=in_channels, out_channels=out_channels, downsample=downsample, stride=stride))
        for _ in range(num_block - 1):
            layers.append(block(in_channels=out_channels, out_channels=out_channels, downsample=None, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = Res34Net([3, 4, 6, 3])
    print(model)