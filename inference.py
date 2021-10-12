import torch 
import torch.nn as nn
from model import Res34Net
from data import test_loader

model = torch.load('./model.pth')
model.eval()
model.cuda()
criterion = nn.CrossEntropyLoss()

train_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += len(targets)

print("Loss: {:.6f}|Acc: {:.2f}|".format(train_loss/total, 100*correct/total))
