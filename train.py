import torch
import torch.nn as nn
from data import train_loader, test_loader
from model import Res34Net

model = Res34Net([3, 4, 6, 3]).cuda()
num_epoch = 20
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
if __name__ == "__main__":
    for epoch in range(num_epoch):
        train_loss = 0
        correct = 0
        total = 0
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += len(targets)
            print("Epoch: {}|Loss: {:.6f}|Acc: {:.2f}%|".format(epoch+1, train_loss/total, 100*correct/total))

    torch.save(model, "./model.pth")
