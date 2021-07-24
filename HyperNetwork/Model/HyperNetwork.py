
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.base import BaseNetwork, BookBaseNetwork, EmotionBaseNetwork
from Model.Classifier import SmallClassiffier

import numpy as np

class HyperNetwork(nn.Module):

    def __init__(self, start_size):
        super(HyperNetwork, self).__init__()

        self.base = BaseNetwork(start_size)
        self.classifier = SmallClassiffier()

    def forward(self, x, number):

        x = self.base(x)
        self.x = torch.clone(x)

        fcw1 = x[:10].view(10, 1)
        fcb1 = x[10:20 ]  # .view(10,1)
        fcw2 = x[20:120].view(10 ,10)
        fcb2 = x[120:130]
        fcw3 = x[130:150].view(2 ,10)
        fcb3 = x[150:152]

        self.classifier.fc1.weight = torch.nn.Parameter(fcw1, requires_grad =True)
        self.classifier.fc1.bias = torch.nn.Parameter(fcb1, requires_grad =True)
        self.classifier.fc2.weight = torch.nn.Parameter(fcw2, requires_grad =True)
        self.classifier.fc2.bias = torch.nn.Parameter(fcb2, requires_grad =True)
        self.classifier.fc3.weight = torch.nn.Parameter(fcw3, requires_grad =True)
        self.classifier.fc3.bias = torch.nn.Parameter(fcb3, requires_grad =True)

        optim = torch.optim.Adam(self.classifier.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        #optim.zero_grad()

        return self.classifier(number), optim



class HyperNetworkBooking(nn.Module):

    def __init__(self, max_id, start_size=0):
        super(HyperNetworkBooking, self).__init__()

        self.base = BookBaseNetwork(max_id)
        self.classifier = SmallClassiffier()

    def forward(self, x, number, y):

        x = self.base(x)
        self.x = torch.clone(x)

        fcw1 = x[:30].view(10, 3)
        fcb1 = x[30:40 ]  # .view(10,1)
        fcw2 = x[40:140].view(10 ,10)
        fcb2 = x[140:150]
        fcw3 = x[150:160].view(1 ,10)
        fcb3 = x[160:161]

        self.classifier.fc1.weight = torch.nn.Parameter(fcw1, requires_grad =True)
        self.classifier.fc1.bias = torch.nn.Parameter(fcb1, requires_grad =True)
        self.classifier.fc2.weight = torch.nn.Parameter(fcw2, requires_grad =True)
        self.classifier.fc2.bias = torch.nn.Parameter(fcb2, requires_grad =True)
        self.classifier.fc3.weight = torch.nn.Parameter(fcw3, requires_grad =True)
        self.classifier.fc3.bias = torch.nn.Parameter(fcb3, requires_grad =True)

        optim = torch.optim.Adam(self.classifier.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.MSELoss()

        if y is None:
            return [self.classifier(number)]

        results = []

        for i in range(30):

            optim.zero_grad()

            result = self.classifier(number)

            results.append(result)

            result = result.view(3)
            y = y.view(3)
            loss = criterion(result, y)
            loss.backward()

            optim.step()

        return results


class HyperNetworkEmotion(nn.Module):

    def __init__(self, max_id, start_size=62):
        super(HyperNetworkEmotion, self).__init__()

        self.base = EmotionBaseNetwork(max_id)
        self.classifier = SmallClassiffier()

    def forward(self, x, number, y):

        x = self.base(x)
        self.x = torch.clone(x)

        fcw1 = x[:60].view(10, 6)
        fcb1 = x[60:70 ]  # .view(10,1)
        fcw2 = x[70:170].view(10 ,10)
        fcb2 = x[170:180]
        fcw3 = x[180:190].view(1 ,10)
        fcb3 = x[190:191]

        self.classifier.fc1.weight = torch.nn.Parameter(fcw1, requires_grad =True)
        self.classifier.fc1.bias = torch.nn.Parameter(fcb1, requires_grad =True)
        self.classifier.fc2.weight = torch.nn.Parameter(fcw2, requires_grad =True)
        self.classifier.fc2.bias = torch.nn.Parameter(fcb2, requires_grad =True)
        self.classifier.fc3.weight = torch.nn.Parameter(fcw3, requires_grad =True)
        self.classifier.fc3.bias = torch.nn.Parameter(fcb3, requires_grad =True)

        optim = torch.optim.Adam(self.classifier.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.MSELoss()

        if y is None:
            return [self.classifier(number)]

        results = []

        for i in range(100):

            optim.zero_grad()

            result = self.classifier(number)

            results.append(result)

            result = result.view(6)
            y = y.view(6)
            loss = criterion(result, y)
            loss.backward()

            optim.step()

        return results