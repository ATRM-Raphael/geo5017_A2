import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

np.random.seed(101)


def rf_gridsearch(X_train, X_test, Y_train, Y_test, show=True, save=False):
    n_estimators_range = list(np.arange(start=2, stop=200, step=10))
    n_estimators_range.reverse()
    min_samples_leaf_range = list(np.arange(start=1, stop=20, step=1))

    best_estimators = 0
    best_min_samples_leaf = 0
    rf_accuracy_best = 0
    rf_pred_best = 0
    rf_accuracy_grid = []

    for n_estimators in n_estimators_range:
        rf_accuracy_row = []
        for min_samples_leaf in min_samples_leaf_range:
            rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf).fit(X_train, Y_train)
            rf_pred = rf.predict(X_test)
            rf_accuracy = accuracy_score(Y_test, rf_pred)
            rf_accuracy_row.append(rf_accuracy)
            if rf_accuracy > rf_accuracy_best:
                rf_accuracy_best = rf_accuracy
                best_estimators = n_estimators
                best_min_samples_leaf = min_samples_leaf
                rf_pred_best = rf_pred
        rf_accuracy_grid.append(rf_accuracy_row)

    print(f"Best Estimators: {best_estimators}\n"
          f"Best Min samples leaf: {best_min_samples_leaf}\n"
          f"Highest accuracy: {rf_accuracy_best}")

    cf = confusion_matrix(Y_test, rf_pred_best)
    print(f"Confusion Matrix:\n{cf}")

    fig, ax = plt.subplots(figsize=(8, 8))
    colorbar = plt.imshow(rf_accuracy_grid, cmap='coolwarm')
    plt.colorbar(colorbar)
    plt.title('The Accuracy and Hyper-parameters (random forest)')
    plt.xticks(ticks=range(len(min_samples_leaf_range)), labels=[f"{m:.3g}" for m in min_samples_leaf_range])
    plt.yticks(ticks=range(len(n_estimators_range)), labels=[f"{n:.3g}" for n in n_estimators_range])
    plt.xlabel('Min sample leafs')
    plt.ylabel('N estimators')
    if show:
        plt.show()
    if save:
        fig.savefig('../figures/rf_tuning.png', dpi=200)


if __name__ == '__main__':
    os.chdir("../result_both")
    X = np.load("X_11_34.npy")
    Y = np.load("y.npy")

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.6, test_size=0.4)
    rf_gridsearch(X_train, X_test, Y_train, Y_test, show=False, save=True)