import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection

np.random.seed(101)


def get_learning_curve(X, Y, model, title='Title', check_interval=0.01, show=True, save=False):
    # initial results
    final_apparent_errors = []
    final_true_errors = []
    final_apparent_errors_std = []
    final_true_errors_std = []
    training_set_sizes = []

    value_str = str(check_interval)
    decimal_place = 0
    if '.' in value_str:
        decimal_place = len(value_str.split('.')[1])

    for i in range(int(1 / check_interval - 1)):
        print(f"Iteration: {i}/{int(1 / check_interval - 1) + 1}")
        # split the train_size and test_size
        train_size = round((i + 1) * check_interval, decimal_place)
        test_size = round(1 - train_size, decimal_place)
        print(train_size, test_size)
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
                                                                            train_size=train_size,
                                                                            test_size=test_size)
        model.fit(X_train, Y_train)

        # get training set size of every iteration, used for X labels in plots.
        training_set_sizes.append(int(len(Y) * train_size))

        apparent_errors, true_errors = [], []
        for i in range(5):
            pred_train, pred_test = model.predict(X_train), model.predict(X_test)
            train_score, test_score = accuracy_score(Y_train, pred_train), accuracy_score(Y_test, pred_test)
            apparent_errors.append(1 - train_score), true_errors.append(1 - test_score)

        apparent_error_mean = np.mean(apparent_errors)
        true_error_mean = np.mean(true_errors)
        apparent_error_std = np.std(apparent_errors)
        true_error_std = np.std(true_errors)

        final_apparent_errors.append(apparent_error_mean)
        final_true_errors.append(true_error_mean)
        final_apparent_errors_std.append(apparent_error_std)
        final_true_errors_std.append(true_error_std)

    final_apparent_errors = np.array(final_apparent_errors)
    final_true_errors = np.array(final_true_errors)
    final_apparent_errors_std = np.array(final_apparent_errors_std)
    final_true_errors_std = np.array(final_true_errors_std)
    training_set_sizes = np.array(training_set_sizes)

    fig, ax = plt.subplots()
    ax.plot(training_set_sizes, final_apparent_errors, linewidth=2.0, label="Apparent Error")
    ax.plot(training_set_sizes, final_true_errors, linewidth=2.0, label="True Error")

    plt.fill_between(training_set_sizes, final_true_errors - final_true_errors_std, final_true_errors + final_true_errors_std, alpha=0.1)
    plt.fill_between(training_set_sizes, final_apparent_errors - final_apparent_errors_std, final_apparent_errors + final_apparent_errors_std, alpha=0.1)

    # Restrain the axis and labels
    ax.set(ylim=(0, 1))
    plt.xlabel('Size of training set')
    plt.ylabel('Classification Error')
    plt.legend(loc='best')
    plt.title(title)

    if show:
        plt.show()
    if save:
        plt.savefig(f"../figures/{title}.png", dpi=200)


if __name__ == '__main__':
    os.chdir("../result_both")
    X = np.load("X_11_34.npy")
    Y = np.load("y.npy")

    svm_model = svm.SVC(kernel='rbf', gamma=1.83e-10, C=1e8)
    rf_model = RandomForestClassifier(n_estimators=22, min_samples_leaf=1)

    svm_title = "Learning Curve (Best SVM)"
    rf_title = "Learning Curve (Best RF)"
    # get_learning_curve(X, Y, svm_model, title=svm_title, show=False, save=True)
    get_learning_curve(X, Y, rf_model, title=rf_title, show=True, save=True)