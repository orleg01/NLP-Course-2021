
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SmallClassiffier(nn.Module):

    def __init__(self):
        super(SmallClassiffier, self).__init__()

        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


