import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.model_selection as model_selection

np.random.seed(101)


def gridsearch(X_train, X_test, y_train, y_test, c_range, second_range, kernel):
    accuracy_best = 0
    c_best = 0
    second_best = 0
    pred_test_best = 0
    results_grid = []

    for i in range(len(second_range)):
        print(f"Iteration: {i}/{len(second_range)}")
        second = second_range[i]
        results_row = []
        for j in range(len(c_range)):
            c = c_range[j]
            print(f"iteration_c: {j}")
            cls = svm.SVC()
            if kernel == 'poly':
                cls = svm.SVC(kernel='poly', degree=second, C=c).fit(X_train, y_train)
            if kernel == 'rbf':
                cls = svm.SVC(kernel='rbf', gamma=second, C=c).fit(X_train, y_train)
            pred_test = cls.predict(X_test)
            accuracy_test = accuracy_score(y_test, pred_test)
            results_row.append(accuracy_test)
            if accuracy_best < accuracy_test:
                accuracy_best = accuracy_test
                c_best = c
                second_best = second
                pred_test_best = pred_test
        results_grid.append(results_row)

    return accuracy_best, c_best, second_best, pred_test_best, results_grid


def poly_gridsearch(X_train, X_test, y_train, y_test, show=True, save=False):
    # poly kernel's c_range is stop at 1e10, to decrease the computation complex.
    c_range = list(np.logspace(start=-5, stop=10, num=20, base=10))
    degree_range = list(np.arange(start=1, stop=21, step=1))
    degree_range.reverse()

    results = gridsearch(X_train, X_test, y_train, y_test, c_range, degree_range, kernel='poly')
    accuracy_best = results[0]
    c_best = results[1]
    degree_best = results[2]
    pred_test_best = results[3]
    results_grid = results[4]

    print(f"Best C: {c_best}\nBest degree: {degree_best}\nHighest accuracy: {accuracy_best}")

    cf = confusion_matrix(y_test, pred_test_best)
    print(f"Confusion Matrix:\n{cf}")

    # -- PLOTTING --
    fig, ax = plt.subplots(figsize=(8, 8))
    cb = plt.imshow(results_grid, cmap='coolwarm')
    plt.colorbar(cb)
    plt.title('The Accuracy and Hyper-parameters (poly)')
    plt.xticks(ticks=range(len(c_range)), labels=[f"{c:.3g}" for c in c_range])
    plt.yticks(ticks=range(len(degree_range)), labels=[f"{degree:.3g}" for degree in degree_range])
    plt.xticks(rotation=270)
    plt.xlabel('C')
    plt.ylabel('Degree')
    if show:
        plt.show()
    if save:
        fig.savefig('../figures/poly_tuning.png', dpi=200)

    return accuracy_best, c_best, degree_best, pred_test_best, results_grid


def rbf_gridsearch(X_train, X_test, Y_train, y_test, show=True, save=False):
    c_range = list(np.logspace(start=-5, stop=14, num=20, base=10))
    gamma_range = list(np.logspace(start=-15, stop=5, num=20, base=10))
    gamma_range.reverse()

    results = gridsearch(X_train, X_test, Y_train, y_test, c_range, gamma_range, kernel='rbf')
    accuracy_best = results[0]
    c_best = results[1]
    gamma_best = results[2]
    pred_test_best = results[3]
    results_grid = results[4]

    print(f"Best C: {c_best}\nBest Gamma: {gamma_best}\nHighest accuracy: {accuracy_best}")

    cf = confusion_matrix(y_test, pred_test_best)
    print(f"Confusion Matrix:\n{cf}")

    # -- PLOTTING --
    fig, ax = plt.subplots(figsize=(8, 8))
    cb = plt.imshow(results_grid, cmap='coolwarm')
    plt.colorbar(cb)
    plt.title('The Accuracy and Hyper-parameters (rbf)')
    plt.xticks(ticks=range(len(c_range)), labels=[f"{c:.3g}" for c in c_range])
    plt.yticks(ticks=range(len(gamma_range)), labels=[f"{gamma:.3g}" for gamma in gamma_range])
    plt.xticks(rotation=270)
    plt.xlabel('C')
    plt.ylabel('Gamma')
    if show:
        plt.show()
    if save:
        fig.savefig('../figures/rbf_tuning.png', dpi=200)

    return accuracy_best, c_best, gamma_best, pred_test_best, results_grid


if __name__ == '__main__':
    os.chdir("../result_both")
    X = np.load("X_4_4.npy")
    y = np.load("Y.npy")

    X_train, X_test, Y_train, y_test = model_selection.train_test_split(X, y, train_size=0.6, test_size=0.4,
                                                                        random_state=101)

    poly_gridsearch(X_train, X_test, Y_train, y_test, show=False, save=True)
    # rbf_gridsearch(X_train, X_test, Y_train, Y_test, show=False, save=True)
