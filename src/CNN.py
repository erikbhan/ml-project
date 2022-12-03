import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 26)

    def forward(self, x):
        if type(x) == np.ndarray: x = torch.from_numpy(x)
        x = x.float()
        x = x.to(device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 0, -1)
        x = F.softmax(self.fc1(x))
        return x

    def avg_accuracy(self, x, y):
        avg = 0
        for i in range(len(x)):
            avg += torch.mean(torch.eq(self.forward(x[i]).argmax(), y[i].argmax()).float())
        return (avg/len(x))

# utility methods for importing and exporting models, defined outside class
def export_model(cnn, name="cnn"):
    torch.save(cnn.state_dict(), "cnns/" + name + ".pth")

def import_model(cnn, name="cnn"):
    cnn.load_state_dict(torch.load("cnns/" + name + ".pth"))