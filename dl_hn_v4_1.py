"""
    Written on January 9, 2021
    Code by Junmo Kim

    추가: 범주화해서 CrossEntropyLoss 쓸거임 / 범주화를 정상, 비정상 두개로만 분류했음

    Counter({0: 8360, 1: 6914})
    Counter({0: 13716, 1: 1558})
    Counter({0: 7042, 1: 8232})
    Counter({0: 4724, 1: 10550})
"""


##
import numpy as np
import pandas as pd
from import_data import DATA_PREPROCESS, DATA_PREPROCESS_SH, NORMALIZATION, CATEGORIZATION_v2
import torch
import torch.optim as optim
import torch.nn as nn
from network import DNN, DNN_v1, DNN_cat
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from utils import EarlyStopping, KFOLD_GCN
# from matplotlib import pyplot as plt
# from config import args
from collections import Counter



if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
    cuda = torch.device('cuda')
else:
    print("No CUDA,,,")
    cuda = torch.device('cpu')

np.random.seed(777)
kfold = 5
# y_value = args.y



data, xdata, ydata = DATA_PREPROCESS_SH(log=False)
xdata = NORMALIZATION(xdata, 'standard')
ydata = CATEGORIZATION_v2(ydata)
for i in range(len(ydata[0])):
    print(Counter(ydata[:, i]))


def _ychoice(y):
    this = np.transpose(np.concatenate((np.transpose(xdata), [ydata[:, y]]), axis=0))
    this = np.array(this)
    # this = this.astype(float)
    np.random.shuffle(this)
    return this
# data.to_csv('kky1.csv', index=None)





# def _TData(train, y):
#     """
#      0~14 features, 15~18 labels
#     """
#     til = size[1] - 4
#     train_dataset = TensorDataset(torch.Tensor(train[:, :til]).to(cuda),
#                                   torch.Tensor(train[:, y]).to(cuda))
#
#     return train_dataset

def _TData_v2(train):
    train_dataset = TensorDataset(torch.Tensor(train[:, :-1]).to(cuda),
                                  torch.LongTensor(train[:, -1]).to(cuda))

    return train_dataset



def Parameters(net):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss(reduction='mean')
    # optimizer = optim.SGD(net.parameters(), lr=0.000001)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    return criterion, optimizer

def Parameters_cat(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    return criterion, optimizer


##

def Learning(train_loader, val_loader, nodes, num, patience=10, num_classes=None):
    """
        train_loader: 학습시킬 데이터셋 (DataLoader 형식)
        val_loader: 평가할 데이터셋 (DataLoader 형식)
        num: Epochs

    """
    net = DNN_cat(xdata.shape[1], num_classes, nodes)
    criterion, optimizer = Parameters_cat(net)
    net.to(cuda)

    """Early Stopping Template"""
    for epoch in range(num):
        torch.cuda.empty_cache()
        if epoch == 0:
            early_stopping = EarlyStopping(patience=patience, verbose=False)
        else:
            early_stopping = EarlyStopping(patience=patience, best_score=best_score, counter=counter, verbose=False)

        # Training
        net.train()
        running_loss = 0.
        lcnt = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            lcnt += len(labels)

        # Validation
        net.eval()
        net.to(cuda)

        val_loss = 0
        vcnt = 0
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            val_loss += loss.item()
            vcnt += len(labels)

        # Template
        best_score, counter, finish = early_stopping(val_loss / vcnt, net)
        # if epoch % 50 == 0:
        #     print('Epoch Now: %d' % epoch)

        if finish:
            break

    net1 = DNN_cat(xdata.shape[1], num_classes, nodes)
    net1.load_state_dict(torch.load('checkpoint.pt'))
    net1.eval()

    # output save
    val_loss = 0
    correct = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    precision_c = list(0. for i in range(num_classes))
    precision_t = list(0. for i in range(num_classes))

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)

            outputs = net1(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            val_loss += loss.item()

            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            for i in range(len(predicted)):
                output = predicted[i].item()
                precision_c[output] += c[i].item()
                precision_t[output] += 1

        print('Best Epoch: %5d / Last Validation Loss: %5f / Accuracy: %.3f'
              % (epoch - patience, val_loss / vcnt, 100 * correct / vcnt))

        for cls in range(num_classes):
            print('Recall of %5s : %2d %%' % (
                cls, 100 * class_correct[cls] / class_total[cls]))

        for cls in range(num_classes):
            if precision_t[cls] == 0:
                print('No prediction for %s' % cls)
            else:
                print('Precision of %5s : %2d %%' % (
                    cls, 100 * precision_c[cls] / precision_t[cls]))

    return correct / vcnt


def RUN(batch_size, kfold, max_epoch, y_value, node_list, patience):
    # Train / Test split
    data = _ychoice(y_value)
    num_classes = len(set(data[:, -1]))
    tmp = KFOLD_GCN(data, KFOLD=kfold)
    train = np.vstack(tmp[:-1])
    test = tmp[-1]


    print('\n\n\n가즈아~!~!~!~!~!')
    print('y_value: %2d' % y_value)

    acc_box = []
    for nodes in node_list:

        acc = 0
        print('Nodes: %2d' % nodes)

        kfold_all = KFOLD_GCN(train, kfold)
        for k in range(kfold):
            validation = kfold_all[k]
            training = []
            for exp in range(kfold):
                if exp != k:
                    training.append(kfold_all[exp])
            training = np.vstack(training)

            train_loader = DataLoader(_TData_v2(training), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(_TData_v2(validation), batch_size=batch_size, shuffle=False)
            acc += Learning(train_loader, val_loader, nodes, max_epoch, patience, num_classes)


        print('Validation Accuracy: %f' % (acc / kfold))
        acc_box.append(acc / kfold)

    best_node = node_list[acc_box.index(max(acc_box))]
    train_loader = DataLoader(_TData_v2(train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(_TData_v2(test), batch_size=batch_size, shuffle=False)

    acc = Learning(train_loader, test_loader, best_node, max_epoch, patience, num_classes)

    print('# of node: %d \nTest Accuracy: %f' %
          (best_node, acc))

    return acc


batch_size = 512
kfold = 5
max_epoch = 1000

n_list = [6, 9, 12, 15, 18]


RUN(batch_size, kfold, max_epoch, y_value=-4, node_list=n_list, patience=100)
RUN(batch_size, kfold, max_epoch, y_value=-3, node_list=n_list, patience=100)
RUN(batch_size, kfold, max_epoch, y_value=-2, node_list=n_list, patience=100)
RUN(batch_size, kfold, max_epoch, y_value=-1, node_list=n_list, patience=100)

# bestnode = int(racc[racc.index(max(racc))])









