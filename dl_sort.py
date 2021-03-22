"""
    Written on January 9, 2021
    Code by Junmo Kim

    Using DATA:
        AI_DATA_SORT.csv

    Data Description:
        Size of Raw data: (51187, 46)
        Columns w/ different type: 5, 6, 14, 15
        Features selected if the proportion of null values is less than 20%
        After feature selection, every row with at least 1 null values is excluded.
        Size of preprocessed data: (37494, 15)

"""


import numpy as np
import pandas as pd
from import_data import DATA_PREPROCESS, DATA_PREPROCESS_v2, NORMALIZATION
import torch
import torch.optim as optim
import torch.nn as nn
from network import DNN, DNN_v1
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.model_selection import KFold


if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
    cuda = torch.device('cuda')
else:
    cuda = torch.device('cpu')



np.random.seed(777)
torch.manual_seed(777)

data, xdata, ydata = DATA_PREPROCESS()
size = data.shape
data.to_csv('kky22.csv', index=None)


# 0~6 features, 7~10 labels
data = np.array(data)
data = data.astype(float)

# Train / Test split
np.random.shuffle(data)
thres = int(0.8 * len(data))
train = data[:thres]
test = data[thres:]


def _TData(train, y):
    """
     0~8 features, 9~13 labels
    """
    train_dataset = TensorDataset(torch.Tensor(train[:, :9]).to(cuda),
                                  torch.Tensor(train[:, y]).to(cuda))

    return train_dataset


def Parameters(net):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss(reduction='mean')
    # optimizer = optim.SGD(net.parameters(), lr=0.000001)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    return criterion, optimizer


def Learning(train_loader, val_loader, nodes, num):
    """
        train_loader: 학습시킬 데이터셋 (DataLoader 형식)
        val_loader: 평가할 데이터셋 (DataLoader 형식)
        num: Epochs

    """
    net = DNN_v1(xdata.shape[1], 1, nodes)
    criterion, optimizer = Parameters(net)
    net.to(cuda)
    net.train()

    for epoch in range(num):
        running_loss = 0.
        lcnt = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)

            outputs = net(inputs)
            loss = criterion(outputs, torch.unsqueeze(labels, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            lcnt += len(labels)

    print('Epoch: %d / Final Loss: %.3f' % (num, running_loss / lcnt))
    net.eval()
    net.to(cuda)

    total = 0
    acc_1 = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            acc = 0
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)

            outputs = net(inputs)
            for j in range(len(outputs)):
                acc += 1 - (abs(outputs[j] - labels[j]) / labels[j])

            acc_1 += acc
            total += labels.size(0)

    return acc_1 / total


def RUN(batch_size, kfold, ep_list, y_value, nodes):
    print('y_value: %d' % y_value)
    acc_box = []
    for num in ep_list:
        kfold_acc = 0.
        kf = KFold(n_splits=kfold)
        for train_index, test_index in kf.split(train):
            training, validation = train[train_index], train[test_index]

            train_loader = DataLoader(_TData(training, y_value), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(_TData(validation, y_value), batch_size=batch_size, shuffle=True)

            kfold_acc += Learning(train_loader, val_loader, nodes, num)

        print('Epoch: %5d \nValidation Accuracy: %5.2f' % (num, kfold_acc / kfold))
        acc_box.append((kfold_acc / kfold).item())

    ep = int(acc_box[acc_box.index(max(acc_box))])
    train_loader = DataLoader(_TData(train, y_value), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(_TData(test, y_value), batch_size=batch_size, shuffle=False)

    accuracy = Learning(train_loader, test_loader, nodes, ep)

    print('Epoch: %5d \nTest Accuracy: %5.2f' % (ep, accuracy))



batch_size = 512
kfold = 5
ep_list = range(200, 320, 20)

RUN(batch_size, kfold, ep_list, y_value=15, nodes=3)  # y_value: 15, 16, 17, 18














