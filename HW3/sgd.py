#################################
# Your name: Rony Kositsky
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = [], [], [], [], [], []


def helper_hinge():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labelss = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labelss = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labelss = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_dataa = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_dataa = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_dataa = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_dataa, train_labelss, validation_dataa, validation_labelss, test_dataa, test_labelss


def helper_ce():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labelss = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labelss = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labelss = labels[8000 + test_idx]

    # Preprocessing
    train_dataa = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_dataa = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_dataa = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_dataa, train_labelss, validation_dataa, validation_labelss, test_dataa, test_labelss


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    t = 1
    indexes = np.random.randint(0, len(data), T)
    w = data[0]  # taking the first one as default
    while t <= T:
        index = indexes[t - 1]
        eta = eta_0 / t
        w = (1 - eta) * w + eta * C * labels[index] * data[index] if (np.dot(w, data[index]) * labels[index] < 1) \
            else (1 - eta) * w
        t += 1
    return w


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    number_of_labels = 10
    number_of_data = 784
    random_indexes = np.random.randint(0, len(data), T)

    # w = np.zeros((number_of_labels, number_of_data))
    #
    # for i in range(number_of_labels):
    #     w[i] = data[0]
    w = np.random.randint(-300, 300, size=(number_of_labels, number_of_data))
    t = 1

    while t <= T:
        index = random_indexes[t - 1]
        eta = eta_0 / t
        x = data[index]
        y = int(labels[index])

        # We are using the log sum exp trick to calculate the gradient.
        denominator = log_sum_exp_trick(w, x, number_of_labels)
        for i in range(number_of_labels):
            numerator = np.dot(w[i], x)
            gradient = np.exp(numerator - denominator)
            delta = eta * (gradient - the_same_row(y, i)) * x
            w[i] = np.subtract(w[i], delta)

        t += 1

    return w


def log_sum_exp_trick(w, x, number_of_labels):
    exponents = [np.dot(w[i], x) for i in range(number_of_labels)]
    max_exp = np.max(exponents)
    return max_exp + np.log(np.sum([np.exp(exp - max_exp) for exp in exponents]))


def the_same_row(y, k):
    return True if y == k else False


def modified_sign(x):
    return 1 if x > 0 else -1


def get_classifier_accuracy(classifier, multi_class):
    sample_size = len(validation_data)
    success = 0
    for i in range(sample_size):
        x = validation_data[i]
        prediction = np.dot(classifier, x)
        success += np.argmax(prediction) == int(validation_labels[i]) if multi_class else \
            (modified_sign(prediction) == validation_labels[i])
    return success / sample_size


def hinge_find_best_eta():
    eta_value = np.linspace(0.1, 100, 100)
    # eta_value = np.logspace(-5, 3, 8)

    C = 1
    T = 1000
    number_of_runs = 10
    accuracy_list = []
    for eta in eta_value:
        print(eta)
        res_lst = np.zeros(number_of_runs, dtype=numpy.float64)
        for i in range(number_of_runs):
            classifier = SGD_hinge(train_data, train_labels, C, eta, T)
            res_lst[i] = get_classifier_accuracy(classifier, False)
        accuracy_list.append(np.average(res_lst))
    max_acc = np.argmax(accuracy_list)
    print("Max eta is " + str(eta_value[max_acc]))
    plt.plot(eta_value, accuracy_list)
    plt.title("Best eta")
    # plt.xscale('log', base=10)
    plt.xlabel("Eta")
    plt.ylabel("Accuracy")
    plt.show()


def hinge_find_best_C():
    c_values = np.logspace(-5, 5, 11)
    c_values = np.linspace(0.1, 10e3, 1000)
    eta = 1.1  # from the last run
    T = 1000
    number_of_runs = 10
    accuracy_list = []
    for c in c_values:
        print(c)
        res_lst = np.zeros(number_of_runs, dtype=numpy.float64)
        for i in range(number_of_runs):
            classifier = SGD_hinge(train_data, train_labels, c, eta, T)
            res_lst[i] = get_classifier_accuracy(classifier, False)
        accuracy_list.append(np.average(res_lst))
    max_acc = np.argmax(accuracy_list)
    print("Max C is " + str(c_values[max_acc]))
    plt.plot(c_values, accuracy_list)
    plt.title("Best C")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    # plt.xscale('log', base=10)
    plt.show()


def hinge_plot_image():
    eta = 1.1
    C = 5435.48
    T = 20000
    w = SGD_hinge(train_data, train_labels, C, eta, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()


def hinge_train_test_data():
    eta = 1.1
    C = 5435.48
    T = 20000
    w = SGD_hinge(test_data, test_labels, C, eta, T)
    print(get_classifier_accuracy(w, False))


def hinge_loss():
    hinge_find_best_eta()
    hinge_find_best_C()
    hinge_plot_image()
    hinge_train_test_data()


def entropy_find_best_eta():
    #eta_value = np.logspace(-10, 7, 18)
    eta_value = np.linspace(10, 1e6, 100)
    T = 1000
    number_of_runs = 10
    accuracy_list = []
    for eta in eta_value:
        res_lst = np.zeros(number_of_runs, dtype=numpy.float64)
        print(eta)
        for i in range(number_of_runs):
            classifier = SGD_ce(train_data, train_labels, eta, T)
            res_lst[i] = get_classifier_accuracy(classifier, True)
        accuracy_list.append(np.average(res_lst))
    max_acc = np.argmax(accuracy_list)
    print("Max eta is " + str(eta_value[max_acc]))
    plt.plot(eta_value, accuracy_list)
    plt.title("Best eta")
    plt.xlabel("Eta")
    plt.ylabel("Accuracy")
    #plt.xscale('log', base=10)
    plt.show()


def entropy_draw_pictures():
    eta = 818183.6363636364
    T = 20000
    w = SGD_ce(train_data, train_labels, eta, T)

    rows = 5
    columns = 2
    fig = plt.figure(figsize=(2, 5))

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.reshape(w[i - 1], (28, 28)), interpolation='nearest')

    plt.show()


def entropy_test():
    eta = 818183.6363636364
    T = 20000
    w = SGD_ce(test_data, test_labels, eta, T)
    print(get_classifier_accuracy(w, True))


def entropy_loss():
    #entropy_find_best_eta()
    #entropy_draw_pictures()
    entropy_test()


if __name__ == '__main__':
    # train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    # hinge_loss()
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    entropy_loss()
    print("Finished")
