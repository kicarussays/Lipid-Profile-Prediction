"""
    범주 숫자로 바꿔야된다

"""


from scipy.special import softmax
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.special import softmax
import os

lipid_list = ['TCH', 'TG', 'HDL', 'LDL']

for i in range(4):
    filename = 'roc_curve/' + lipid_list[i] + '_cat.pkl'
    with open(filename, 'rb') as f:
        cat = pickle.load(f)
    filename = 'roc_curve/' + lipid_list[i] + '_TF.pkl'
    with open(filename, 'rb') as f:
        tf = pickle.load(f)
    filename = 'roc_curve/' + lipid_list[i] + '_remove.pkl'
    with open(filename, 'rb') as f:
        remove = pickle.load(f)
    filename = 'roc_curve/' + lipid_list[i] + '_remove_nonambig.pkl'
    with open(filename, 'rb') as f:
        remove_nonambig = pickle.load(f)

    plt.figure(figsize=(10, 10))
    x1, x2, _ = roc_curve(tf[0], np.array([softmax(i)[1] for i in tf[1]]))
    xscore = round(roc_auc_score(tf[0], np.array([softmax(i)[1] for i in tf[1]])), 2)
    y1, y2, _ = roc_curve(remove[0], np.array([softmax(i)[1] for i in remove[1]]))
    yscore = round(roc_auc_score(remove[0], np.array([softmax(i)[1] for i in remove[1]])), 2)
    z1, z2, _ = roc_curve(remove_nonambig[0], np.array([softmax(i)[1] for i in remove_nonambig[1]]))
    zscore = round(roc_auc_score(remove_nonambig[0], np.array([softmax(i)[1] for i in remove_nonambig[1]])), 2)
    label = 'Normal / Abnormal: AUC {}'.format(xscore)
    plt.plot(x1, x2, linestyle='--', label=label)
    label = 'Ambiguous Data Removed: AUC {}'.format(yscore)
    plt.plot(y1, y2, linestyle='--', label=label)
    label = 'Ambiguous Data Included: AUC {}'.format(zscore)
    plt.plot(z1, z2, linestyle='--', label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Standard: AUC 0.5')
    plt.legend(loc='lower right')
    plt.title(lipid_list[i] + ' ROC Graph')
    figname = lipid_list[i] + '_Graph.png'
    plt.savefig(figname)








print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])








