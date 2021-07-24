

from Model.HyperNetwork import *
from CreateDataSet import CreateDataSet
import random

create_data_set = CreateDataSet()
network = HyperNetworkBooking(len(create_data_set.data_director.word_to_id)).cuda()
optimBase = torch.optim.Adam(network.base.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

mse = nn.MSELoss()

import torch.utils.data as data_utils

epoches = 1000
for epoch in range(epoches):

    print(epoch)
    indexes_rand = [_ for _ in range(len(create_data_set.x))]
    random.shuffle(indexes_rand)

    true_result = 0
    total_result = 0

    for index in indexes_rand:

        optimBase.zero_grad()

        x = create_data_set.x[index]
        y = create_data_set.y[index]

        x = torch.from_numpy(np.array(x, dtype=np.long))
        x = x.type(torch.LongTensor).cuda()
        _y = torch.from_numpy(np.array([[10 if i == y else 0] for i in range(1,4)], dtype=np.float32)).cuda()



        _x = torch.from_numpy(np.array( [[1 if j == i else 0 for j in range(3)] for i in range(3)], dtype=np.float32)).cuda()

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

        x = create_data_set.x[index]
        y = create_data_set.y[index]

        x = torch.from_numpy(np.array(x, dtype=np.long))
        x = x.type(torch.LongTensor).cuda()

        number_1 = torch.from_numpy(np.array([1, 0, 0], dtype=np.float32)).cuda()
        number_2 = torch.from_numpy(np.array([0, 1, 0], dtype=np.float32)).cuda()
        number_3 = torch.from_numpy(np.array([0, 0, 1], dtype=np.float32)).cuda()

        total_result += 1

        inf_1 = network(x, number_1, None)[0]
        inf_2 = network(x, number_2, None)[0]
        inf_3 = network(x, number_3, None)[0]

        if y == 1 and inf_1 > inf_2 and inf_1 > inf_3:
            true_result += 1
        if y == 2 and inf_2 > inf_1 and inf_2 > inf_3:
            true_result += 1
        if y == 3 and inf_3 > inf_1 and inf_3 > inf_2:
            true_result += 1

    print("Epoch ", epoch, " total inference = ", (true_result / total_result))

    torch.save(network.state_dict(), "book_network.oy")

print('Finished Training')