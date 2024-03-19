import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.model_selection as model_selection

np.random.seed(101)


def poly_gridsearch(X_train, X_test, Y_train, Y_test, show=True, save=False):
    c_range = list(np.logspace(start=-4, stop=15, num=20, base=10))
    degree_range = list(np.arange(start=1, stop=21, step=1))
    degree_range.reverse()
    
    accuracy_max = 0
    c_best = 0
    degree_best = 0
    results_grid = []
    for degree in degree_range:
        results_row = []
        for c in c_range:
            cls = svm.SVC(kernel='poly', degree=degree, C=c).fit(X_train, Y_train)
            pred_test = cls.predict(X_test)
            accuracy_test = accuracy_score(Y_test, pred_test)
            results_row.append(accuracy_test)
            if accuracy_max < accuracy_test:
                accuracy_max = accuracy_test
                c_best = c
                degree_best = degree
        results_grid.append(results_row)

    print(f"Best C: {c_best}\nBest degree: {degree_best}\nHighest accuracy: {accuracy_max}")

    fig, ax = plt.subplots(figsize=(8, 8))
    cb = plt.imshow(results_grid, cmap='coolwarm')
    plt.colorbar(cb)
    plt.title('The Accuracy and Hyper-parameters (poly)')
    plt.xticks(ticks=range(len(c_range)), labels=['{:.3g}'.format(c) for c in c_range])
    plt.yticks(ticks=range(len(degree_range)), labels=['{:.3g}'.format(degree) for degree in degree_range])
    plt.xticks(rotation=270)
    plt.xlabel('C')
    plt.ylabel('Degree')
    if show:
        plt.show()
    if save:
        fig.savefig('poly.png', dpi=200)


def rbf_gridsearch(X_train, X_test, Y_train, Y_test, show=True, save=False):
    c_range = list(np.logspace(start=-4, stop=15, num=25, base=10))
    gamma_range = list(np.logspace(start=-15, stop=5, num=25, base=10))
    gamma_range.reverse()

    accuracy_max = 0
    c_best = 0
    gamma_best = 0
    results_grid = []
    for gamma in gamma_range:
        results_row = []
        for c in c_range:
            cls = svm.SVC(kernel='rbf', gamma=gamma, C=c).fit(X_train, Y_train)
            pred_test = cls.predict(X_test)
            accuracy_test = accuracy_score(Y_test, pred_test)
            results_row.append(accuracy_test)
            if accuracy_max < accuracy_test:
                accuracy_max = accuracy_test
                c_best = c
                gamma_best = gamma
        results_grid.append(results_row)

    print(f"Best C: {c_best}\nBest Gamma: {gamma_best}\nHighest accuracy: {accuracy_max}")

    fig, ax = plt.subplots(figsize=(8, 8))
    cb = plt.imshow(results_grid, cmap='coolwarm')
    plt.colorbar(cb)
    plt.title('The Accuracy and Hyper-parameters (poly)')
    plt.xticks(ticks=range(len(c_range)), labels=['{:.3g}'.format(c) for c in c_range])
    plt.yticks(ticks=range(len(gamma_range)), labels=['{:.3g}'.format(gamma) for gamma in gamma_range])
    plt.xticks(rotation=270)
    plt.xlabel('C')
    plt.ylabel('Gamma')
    if show:
        plt.show()
    if save:
        fig.savefig('rbf.png', dpi=200)


if __name__ == '__main__':
    os.chdir("../result_both")
    X = np.load("X_11_34.npy")
    Y = np.load("y.npy")
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.6, test_size=0.4)
    poly_gridsearch(X_train, X_test, Y_train, Y_test, show=True)
    rbf_gridsearch(X_train, X_test, Y_train, Y_test, show=True)