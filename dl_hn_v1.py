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
from sklearn.model_selection import KFold
from config import args



if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
    cuda = torch.device('cuda')
else:
    print("No CUDA,,,")
    cuda = torch.device('cpu')

y_value = args.y
nodes = args.n
dataset = args.d

print('NORMALIZATION ON')
if dataset == 1:    # hn_test
    data, xdata, ydata = DATA_PREPROCESS_v2()
    size = data.shape
    data = np.concatenate((NORMALIZATION(xdata, 'standard'), ydata), axis=1)
    # data.to_csv('kky1.csv', index=None)

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

elif dataset == 2:      # Sort data
    data, xdata, ydata = DATA_PREPROCESS()
    size = data.shape
    data = np.concatenate((NORMALIZATION(xdata, 'standard'), ydata), axis=1)
    # data.to_csv('kky1.csv', index=None)

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
         0~14 features, 15~18 labels
        """
        train_dataset = TensorDataset(torch.Tensor(train[:, :17]).to(cuda),
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
    print('\n\n\n가즈아~!~!~!~!~!')
    print('y_value: %2d' % y_value)
    print('Nodes: %2d' % nodes)
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

    ep = ep_list[acc_box.index(max(acc_box))]
    train_loader = DataLoader(_TData(train, y_value), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(_TData(test, y_value), batch_size=batch_size, shuffle=False)

    accuracy = Learning(train_loader, test_loader, nodes, ep)

    print('Epoch: %5d \nTest Accuracy: %5.2f' % (ep, accuracy))

    return accuracy.item()


batch_size = 512
kfold = 5
ep_list = range(100, 300, 50)


# y_list = [9, 10, 11, 12]
n_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

racc = []
for node in n_list:
    racc.append(RUN(batch_size, kfold, ep_list, y_value=y_value, nodes=node))  # y_value: 9, 10, 11, 12

bestnode = int(racc[racc.index(max(racc))])









