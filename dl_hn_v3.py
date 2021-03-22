"""
    Written on January 9, 2021
    Code by Junmo Kim

    추가내용: 변수추가, R스퀘어 계산

"""

##
import numpy as np
import pandas as pd
from import_data import DATA_PREPROCESS, DATA_PREPROCESS_SH, NORMALIZATION
import torch
import torch.optim as optim
import torch.nn as nn
from network import DNN, DNN_v1
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from utils import EarlyStopping
# from matplotlib import pyplot as plt
# from config import args



if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
    cuda = torch.device('cuda')
else:
    print("No CUDA,,,")
    cuda = torch.device('cpu')

# y_value = args.y

print('NORMALIZATION ON')
data, xdata, ydata = DATA_PREPROCESS_SH()
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
    til = size[1] - 4
    train_dataset = TensorDataset(torch.Tensor(train[:, :til]).to(cuda),
                                  torch.Tensor(train[:, y]).to(cuda))

    return train_dataset


def Parameters(net):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss(reduction='mean')
    # optimizer = optim.SGD(net.parameters(), lr=0.000001)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    return criterion, optimizer

##

def Learning(train_loader, val_loader, nodes, num, patience=10):
    """
        train_loader: 학습시킬 데이터셋 (DataLoader 형식)
        val_loader: 평가할 데이터셋 (DataLoader 형식)
        num: Epochs

    """
    net = DNN_v1(xdata.shape[1], 1, nodes)
    criterion, optimizer = Parameters(net)
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
            loss = criterion(outputs, torch.unsqueeze(labels, 1))

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

            loss = criterion(outputs, torch.unsqueeze(labels, 1))
            val_loss += loss.item()
            vcnt += len(labels)

        # Template
        best_score, counter, finish = early_stopping(val_loss / vcnt, net)
        # if epoch % 50 == 0:
        #     print('Epoch Now: %d' % epoch)

        if finish:
            break


    net1 = DNN_v1(xdata.shape[1], 1, nodes)
    net1.load_state_dict(torch.load('checkpoint.pt'))
    net1.eval()

    # output save
    r_label = []
    r_output = []
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)

            outputs = net1(inputs)
            loss = criterion(outputs, torch.unsqueeze(labels, 1))
            val_loss += loss.item()
            outputs = torch.flatten(outputs)

            r_label.append(labels)
            r_output.append(outputs)

        print('Best Epoch: %5d / Last Validation Loss: %5f' % (epoch - patience, val_loss / vcnt))

    return r_label, r_output


def RUN(batch_size, kfold, max_epoch, y_value, node_list, patience):
    print('\n\n\n가즈아~!~!~!~!~!')
    print('y_value: %2d' % y_value)

    rsq_box = []
    for nodes in node_list:
        kf = KFold(n_splits=kfold)
        r_final_label = []
        r_final_output = []
        print('Nodes: %2d' % nodes)
        for train_index, test_index in kf.split(train):
            training, validation = train[train_index], train[test_index]

            train_loader = DataLoader(_TData(training, y_value), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(_TData(validation, y_value), batch_size=batch_size, shuffle=False)
            r_label, r_output = Learning(train_loader, val_loader, nodes, max_epoch, patience)

            r_final_label += r_label
            r_final_output += r_output

        r_final_label = np.hstack([data.cpu().numpy() for data in r_final_label])
        r_final_output = np.hstack([data.cpu().numpy() for data in r_final_output])
        print('Validation R-squared: %f' % (r2_score(r_final_label, r_final_output)))
        rsq_box.append(r2_score(r_final_label, r_final_output))

    best_node = node_list[rsq_box.index(max(rsq_box))]
    train_loader = DataLoader(_TData(train, y_value), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(_TData(test, y_value), batch_size=batch_size, shuffle=False)

    labels, outputs = Learning(train_loader, test_loader, best_node, max_epoch, patience)
    labels = np.hstack([data.cpu().numpy() for data in labels])
    outputs = np.hstack([data.cpu().numpy() for data in outputs])

    print('# of node: %d \nTest R-squared: %f' %
          (best_node, r2_score(labels, outputs)))

    return r2_score(labels, outputs)


batch_size = 512
kfold = 5
max_epoch = 1000

n_list = [6, 9, 12, 15, 18, 21, 24, 27, 30]


RUN(batch_size, kfold, max_epoch, y_value=-4, node_list=n_list, patience=50)
RUN(batch_size, kfold, max_epoch, y_value=-3, node_list=n_list, patience=50)
RUN(batch_size, kfold, max_epoch, y_value=-2, node_list=n_list, patience=50)
RUN(batch_size, kfold, max_epoch, y_value=-1, node_list=n_list, patience=50)

# bestnode = int(racc[racc.index(max(racc))])









