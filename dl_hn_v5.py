import numpy as np
import pandas as pd
from import_data import FINAL_DATA, NORMALIZATION, CATEGORIZATION_v2, CATEGORIZATION, REMOVE_AMBIGUOUS
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


# Configuration Setting
if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
    cuda = torch.device('cuda')
else:
    print("No CUDA,,,")
    cuda = torch.device('cpu')

seednum = 777
np.random.seed(seednum)
torch.manual_seed(seednum)
kfold = 5


# Data Loading
data, xdata, ydata = FINAL_DATA()
xnorm = NORMALIZATION(xdata, 'standard')
label = CATEGORIZATION_v2(ydata)


# Parameters
batch_size = 128
max_epoch = 1000
n_list = [6, 9, 12, 15, 18, 21, 24]








