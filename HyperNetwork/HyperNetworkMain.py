
from Model.HyperNetwork import *

network = HyperNetwork(1000)
optimBase = torch.optim.Adam(network.base.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()

import torch.utils.data as data_utils

epoches = 100
for epoch in range(epoches):

    print(epoch)
    optimBase.zero_grad()

    if epoch % 2 == 1:
        x = torch.from_numpy(np.array([1.0 for i in range(1000)], dtype=np.float32))
        y = torch.from_numpy(np.array([1], dtype=np.int64))
    else:
        x = torch.from_numpy(np.array([0.0 for i in range(1000)], dtype=np.float32))
        y = torch.from_numpy(np.array([0], dtype=np.int64))
    number = torch.from_numpy(np.array([8], dtype=np.float32))

    outputs, optimClassifier = network(x, number)
    print(outputs)

    loss = criterion(outputs.view(1, 2), y.view(1))
    loss.backward()

    optimClassifier.step()

    ##optimBase.zero_grad()
    classifier = network.classifier
    arr = torch.cat((classifier.fc1.weight.data.view(-1),
                     classifier.fc1.bias.data.view(-1),
                     classifier.fc2.weight.data.view(-1),
                     classifier.fc2.bias.data.view(-1),
                     classifier.fc3.weight.data.view(-1),
                     classifier.fc3.bias.data.view(-1)))

    arr = arr.detach().numpy()
    arr = torch.from_numpy(arr)

    #print(network.x.shape, arr.shape)
    loss = mse(network.x, arr)
    #print(network.x, arr)

    loss.backward()
    optimBase.step()

    #network.classifier = SmallClassiffier()

    #optim2 = torch.optim.Adam(network.classifier.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    #optim1 = torch.optim.Adam(network.base.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

print('Finished Training')