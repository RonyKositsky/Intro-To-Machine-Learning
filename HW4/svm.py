#################################
# Your name: Rony Kositsky
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""


# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    c = 1000
    lin_mod = svm.SVC(C=c, kernel='linear')
    quad_mod = svm.SVC(C=c, kernel='poly', degree=2)
    rbf_mod = svm.SVC(C=c, kernel='rbf')
    sv_per_class = []

    # linear case:
    trained_lin_mod = lin_mod.fit(X_train, y_train)
    create_plot(X_train, y_train, trained_lin_mod)
    lin_sv = trained_lin_mod.n_support_
    sv_per_class.append(lin_sv)
    print(lin_sv)
    # plt.show()

    # quadratic case:
    trained_quad_mod = quad_mod.fit(X_train, y_train)
    create_plot(X_train, y_train, trained_quad_mod)
    quad_sv = trained_quad_mod.n_support_
    sv_per_class.append(quad_sv)
    print(quad_sv)
    # plt.show()

    # RBF case:
    trained_rbf_mod = rbf_mod.fit(X_train, y_train)
    create_plot(X_train, y_train, trained_rbf_mod)
    rbf_sv = trained_rbf_mod.n_support_
    sv_per_class.append(rbf_sv)
    print(rbf_sv)
    # plt.show()

    return np.array(sv_per_class)


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    penalties = np.logspace(-5, 6, 12)
    accuracies_train = []
    accuracies_val = []

    for c in penalties:
        lin_mod = svm.SVC(C=c, kernel='linear')
        trained_lin_mod = lin_mod.fit(X_train, y_train)
        train_acc = trained_lin_mod.score(X_train, y_train)
        val_acc = trained_lin_mod.score(X_val, y_val)

        if c == 10 ** 5:
            create_plot(X_val, y_val, trained_lin_mod)
            # plt.show()

        accuracies_train.append(train_acc)
        accuracies_val.append(val_acc)

    plt.plot(penalties, accuracies_train, label='training accuracies')
    plt.plot(penalties, accuracies_val, label='validation accuracies')
    plt.xscale('log')
    plt.xlabel("Penalty")
    plt.ylabel("Accuracy")
    plt.legend()
    # plt.show()

    return accuracies_val


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gammas = np.logspace(-5, 6, 12)
    c = 10
    accuracies_train = []
    accuracies_val = []

    for g in gammas:
        rbf_mod = svm.SVC(C=c, kernel='rbf', gamma=g)
        trained_rbf_mod = rbf_mod.fit(X_train, y_train)
        train_acc = trained_rbf_mod.score(X_train, y_train)
        val_acc = trained_rbf_mod.score(X_val, y_val)
        accuracies_train.append(train_acc)
        accuracies_val.append(val_acc)
        if g == 100:
            create_plot(X_train, y_train, trained_rbf_mod)
            #plt.show()

    plt.plot(gammas, accuracies_train, label='training accuracies')
    plt.plot(gammas, accuracies_val, label='validation accuracies')
    plt.xscale('log')
    plt.legend()
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    # plt.show()
    return accuracies_val


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = get_points()
    # train_three_kernels(X_train, y_train, X_val, y_val)
    # linear_accuracy_per_C(X_train, y_train, X_val, y_val)
    # rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)
