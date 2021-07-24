
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import numpy as np


class BaseNetwork(nn.Module):

    def __init__(self, init_size, result_size=152):
        super(BaseNetwork, self).__init__()

        self.fc1 = nn.Linear(init_size, init_size * 2)
        self.fc2 = nn.Linear(init_size * 2, init_size * 2)
        self.fc3 = nn.Linear(init_size * 2, result_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class BookBaseNetwork(nn.Module):

    def __init__(self, max_id, init_size=1000, result_size=161):
        super(BookBaseNetwork, self).__init__()

        self.init_size = init_size

        self.embedding = nn.Embedding(max_id, 20)
        self.linear1 = nn.Linear(20, 100)
        self.linear2 = nn.Linear(100, 5)

        self.linear3 = nn.Linear(5000, 1000)
        self.linear4 = nn.Linear(1000, result_size)

        """
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 3, 5)

        self.max_pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(3, 5, 5)
        self.conv4 = nn.Conv2d(5, 5, 5)

        self.max_pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(5, 10, (10, 3))
        self.conv6 = nn.Conv2d(10, 10, (10, 3))

        self.max_pool3 = nn.MaxPool2d((4,2))

        self.linear3 = nn.Linear(3920, 1000)
        self.linear4 = nn.Linear(1000, 1000)
        self.linear5 = nn.Linear(1000, 1000)
        self.linear6 = nn.Linear(1000, result_size)
        """

    def forward(self, x):

        x = F.normalize(self.embedding(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        """
        x = x.view(1, 1, self.init_size, 100)

        x = F.relu(self.conv1(x))
        x = self.conv2(x)

        x = self.max_pool1(x)

        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        x = self.max_pool2(x)

        x = F.relu(self.conv5(x))
        x = self.conv6(x)

        x = self.max_pool3(x)

        x = x.view(-1)
        x = self.linear3(x)
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = self.linear6(x)
        """

        x = x.view(-1)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x



class EmotionBaseNetwork(nn.Module):

    def __init__(self, max_id, result_size=191):
        super(EmotionBaseNetwork, self).__init__()

        self.embedding = nn.Embedding(max_id, 100)
        self.linear1 = nn.Linear(20, 100)
        self.linear2 = nn.Linear(100, 1000)
        self.linear3 = nn.Linear(1000, 1000)
        self.linear4 = nn.Linear(1000, 1000)
        self.linear5 = nn.Linear(1000, 100)

        self.linear6 = nn.Linear(6200, 1000)
        self.linear7 = nn.Linear(1000, result_size)


    def forward(self, x):

        x = F.normalize(self.embedding(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)

        x = x.view(-1)
        x = F.relu(self.linear6(x))
        x = self.linear7(x)

        return x

#bookClassifier = BookBaseNetwork(20)
#data = torch.from_numpy(np.array([[1] for i in range(1000)], np.long))
#data = data.type(torch.LongTensor)
#res = bookClassifier(data)

#print(res.shape)
