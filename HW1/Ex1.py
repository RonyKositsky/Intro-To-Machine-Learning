#205817893 - Rony Kositsky

import numpy.random
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


mnist = fetch_openml('mnist_784')
data = mnist['data'].values
labels = mnist['target'].values
idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def calculate_distances(train_set, train_labels_set, image):
    distances = []
    for i in range(len(train_set)):
        dist = np.linalg.norm(train_set[i] - image)
        distances.append((dist, train_labels_set[i]))
    return distances


def find_best_match(m_labels):
    dic = {}
    for label in m_labels:
        dic[label] = dic[label] + 1 if label in dic else 1
    return max(dic, key=dic.get)


def predict(train_set, labels_set, image, k):
    dist = sorted(calculate_distances(train_set, labels_set, image))
    res = [label[1] for label in dist[:k]]
    return find_best_match(res)


def accuracy_calculation(n, k):
    n_train = train[:n]
    n_labels = train_labels[:n]
    total = 0
    for i in range(len(test)):
        if predict(n_train, n_labels, test[i], k) == test_labels[i]:
            total += 1
    return total / len(test)


def calculate_accuracy():
    acc = accuracy_calculation(1000, 10)
    print(f'Accuracy is {acc}')


def section_c():
    x = np.arange(1, 101)
    y = []
    for i in range(100):
        y.append(accuracy_calculation(1000, x[i]))
    plt.plot(x, y)
    plt.show()


def section_d():
    x = np.arange(100, 5100, 100)
    y = []
    for i in range(50):
        y.append(accuracy_calculation(x[i], 1))
    plt.plot(x, y)
    plt.show()
