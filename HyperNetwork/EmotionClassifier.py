

from DataBaseCreation import DataBaseCreation
from datasets import load_dataset

dataset = load_dataset('emotion')

train = dataset["train"]

text = train["text"]
label = train["label"]

database = DataBaseCreation(text, label)


from Model.HyperNetwork import *
import random


network = HyperNetworkEmotion(len(database.word_to_id)).cuda()
optimBase = torch.optim.Adam(network.base.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

mse = nn.MSELoss()

import torch.utils.data as data_utils

epoches = 1000
for epoch in range(epoches):

    print(epoch)
    indexes_rand = [_ for _ in range(len(database.x))]
    random.shuffle(indexes_rand)

    true_result = 0
    total_result = 0

    for index in indexes_rand:

        optimBase.zero_grad()

        x = database.x[index]
        y = database.y[index]

        x = torch.from_numpy(np.array(x, dtype=np.long))
        x = x.type(torch.LongTensor).cuda()
        _y = torch.from_numpy(np.array([[10 if i == y else 0] for i in range(6)], dtype=np.float32)).cuda()

        _x = torch.from_numpy(np.array( [[1 if j == i else 0 for j in range(6)] for i in range(6)], dtype=np.float32)).cuda()

        outputs = network(x, _x, _y)
        #print(outputs)

        #loss = criterion(outputs.view(1, 2), _y)
        #loss.backward()

        #optimClassifier.step()

        ##optimBase.zero_grad()
        classifier = network.classifier
        arr = torch.cat((classifier.fc1.weight.data.view(-1),
                         classifier.fc1.bias.data.view(-1),
                         classifier.fc2.weight.data.view(-1),
                         classifier.fc2.bias.data.view(-1),
                         classifier.fc3.weight.data.view(-1),
                         classifier.fc3.bias.data.view(-1)))

        arr = arr.cpu().detach().numpy()
        arr = torch.from_numpy(arr).cuda()

        #print(network.x.shape, arr.shape)
        loss = mse(network.x, arr)
        #print(network.x, arr)

        loss.backward()
        optimBase.step()

        #network.classifier = SmallClassiffier()

        #optim2 = torch.optim.Adam(network.classifier.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        #optim1 = torch.optim.Adam(network.base.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

        x = database.x[index]
        y = database.y[index]

        x = torch.from_numpy(np.array(x, dtype=np.long))
        x = x.type(torch.LongTensor).cuda()

        number_1 = torch.from_numpy(np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)).cuda()
        number_2 = torch.from_numpy(np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)).cuda()
        number_3 = torch.from_numpy(np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)).cuda()
        number_4 = torch.from_numpy(np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)).cuda()
        number_5 = torch.from_numpy(np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)).cuda()
        number_6 = torch.from_numpy(np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)).cuda()

        total_result += 1

        inf_1 = network(x, number_1, None)[0]
        inf_2 = network(x, number_2, None)[0]
        inf_3 = network(x, number_3, None)[0]
        inf_4 = network(x, number_4, None)[0]
        inf_5 = network(x, number_5, None)[0]
        inf_6 = network(x, number_6, None)[0]

        l = [inf_1, inf_2, inf_3, inf_4, inf_5, inf_6]
        max_index = l.index(max(l))

        if y == max_index:

            true_result += 1

    print("Epoch train ", epoch, " total inference = ", (true_result / total_result))

    test = dataset["train"]

    text = test["text"]
    label = test["label"]

    x = database.get_x(text)

    total_result = 0
    true_result = 0
    for i in range(len(x)):

       x = database.x[i]
       y = database.y[i]

       x = torch.from_numpy(np.array(x, dtype=np.long))
       x = x.type(torch.LongTensor).cuda()

       number_1 = torch.from_numpy(np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)).cuda()
       number_2 = torch.from_numpy(np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)).cuda()
       number_3 = torch.from_numpy(np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)).cuda()
       number_4 = torch.from_numpy(np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)).cuda()
       number_5 = torch.from_numpy(np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)).cuda()
       number_6 = torch.from_numpy(np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)).cuda()

       total_result += 1

       inf_1 = network(x, number_1, None)[0]
       inf_2 = network(x, number_2, None)[0]
       inf_3 = network(x, number_3, None)[0]
       inf_4 = network(x, number_4, None)[0]
       inf_5 = network(x, number_5, None)[0]
       inf_6 = network(x, number_6, None)[0]

       l = [inf_1, inf_2, inf_3, inf_4, inf_5, inf_6]
       max_index = l.index(max(l))

       if y == max_index:
          true_result += 1

    print("Epoch test ", epoch, " total inference = ", (true_result / total_result))

    torch.save(network.state_dict(), "emotion_network.oy")

print('Finished Training')