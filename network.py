import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

from torchvision import transforms


class DNN(nn.Module):

    def __init__(self, num_genes, num_classes):
        super(DNN, self).__init__()

        h1 = nn.Linear(num_genes, 5)
        h2 = nn.Linear(5, 5)
        h3 = nn.Linear(5, num_classes)
        bn = nn.BatchNorm1d(5)
        relu = nn.ReLU()

        self.hid = nn.Sequential(
            h1, bn, relu,
            h2, bn, relu,
            h2, bn, relu,
            h2, bn, relu,
            h2, bn, relu,
            h3
        )
        if torch.cuda.is_available():
            self.hid = self.hid.cuda()

    def forward(self, x):
        x = self.hid(x)

        return x


class DNN_decreasing(nn.Module):

    def __init__(self, num_genes, num_classes):
        super(DNN_decreasing, self).__init__()

        h1 = nn.Linear(num_genes, 4096)
        h2 = nn.Linear(4096, 2048)
        h3 = nn.Linear(2048, 1024)
        h4 = nn.Linear(1024, 512)
        h5 = nn.Linear(512, 256)
        h6 = nn.Linear(256, 128)
        h7 = nn.Linear(128, 64)
        dropout = nn.Dropout(p=0.5)

        bn = []
        for i in range(7):
            bn.append(nn.BatchNorm1d(2 ** (12 - i)))

        hfinal = nn.Linear(64, num_classes)
        relu = nn.ReLU()

        self.hid = nn.Sequential(
            h1, bn[0], relu, dropout,
            h2, bn[1], relu, dropout,
            h3, bn[2], relu, dropout,
            h4, bn[3], relu, dropout,
            h5, bn[4], relu, dropout,
            h6, bn[5], relu, dropout,
            h7, bn[6], relu, dropout,
            hfinal
        )

        self.hid = self.hid.cuda()

    def forward(self, x):
        x = self.hid(x)

        return x





class DNN_v1(nn.Module):

    def __init__(self, num_genes, num_classes, hidden):
        super(DNN_v1, self).__init__()

        h1 = nn.Linear(num_genes, hidden)
        h2 = nn.Linear(hidden, hidden)
        h3 = nn.Linear(hidden, num_classes)
        bn = nn.BatchNorm1d(hidden)
        relu = nn.ReLU()

        self.hid = nn.Sequential(
            h1, bn, relu,
            h3
        )
        if torch.cuda.is_available():
            self.hid = self.hid.cuda()

    def forward(self, x):
        # print('before')
        # print(x.size())
        x = self.hid(x)
        # print('after')
        # print(x.size())

        return x


class DNN_cat(nn.Module):

    def __init__(self, num_genes, num_classes, hidden):
        super(DNN_cat, self).__init__()

        h1 = nn.Linear(num_genes, hidden)
        h2 = nn.Linear(hidden, hidden)
        h3 = nn.Linear(hidden, num_classes)
        bn = nn.BatchNorm1d(hidden)
        relu = nn.ReLU()

        self.hid = nn.Sequential(
            h1, bn, relu,
            # h2, bn, relu,
            h3
        )
        if torch.cuda.is_available():
            self.hid = self.hid.cuda()

    def forward(self, x):
        # print('before')
        # print(x.size())
        x = self.hid(x)
        # print('after')
        # print(x.size())

        return x






















