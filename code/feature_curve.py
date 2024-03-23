import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection

np.random.seed(101)


# This method reads X and Y directly from pre-saved npy files.
def get_result(feature_path, label_path):
    X = np.load(feature_path)
    Y = np.load(label_path)

    train_scores, test_scores = [], []
    for i in range(5):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.6, test_size=0.4)
        clf = svm.SVC(kernel='rbf', gamma=1e-10, C=1e10).fit(X_train, Y_train)
        pred_train, pred_test = clf.predict(X_train), clf.predict(X_test)
        train_score, test_score = accuracy_score(Y_train, pred_train), accuracy_score(Y_test, pred_test)
        train_scores.append(train_score), test_scores.append(test_score)

    # Get the final results of all the classifiers
    train_mean = np.mean(train_scores)
    train_std = np.std(train_scores)
    test_mean = np.mean(test_scores)
    test_std = np.std(test_scores)
    return train_mean, test_mean, train_std, test_std


def get_feature_curve(slices_number, test_error, test_std, title, show=True, save=False):
    # Plot the feature curve
    fig, ax = plt.subplots()
    ax.plot(slices_number, test_error, linewidth=2.0, label="Test Error")

    test_error_min = []

    # add vertical lines of every slices number on the plot
    for x, test in zip(slices_number, test_error):
        if test == min(test_error) and len(test_error_min) == 0:
            test_error_min.append(x)
            test_error_min.append(test)

    plt.axvline(x=test_error_min[0], ymax=test_error_min[1], color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=test_error_min[1], color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.text(test_error_min[0], test_error_min[1],
             f"Slices number: {test_error_min[0]}\nMin Test Error: {test_error_min[1]:.2f}",
             fontsize=9, verticalalignment='top', horizontalalignment='left')

    plt.fill_between(slices_number, test_error - test_std, test_error + test_std, alpha=0.1)

    ax.set(ylim=(0, 1))
    plt.xlabel('Slices Number')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.title(title)

    if show:
        plt.show()
    if save:
        plt.savefig(f"../figures/{title}.png", dpi=200)


if __name__ == "__main__":

    all_file_path = "../pointclouds-500"
    file_names = sorted(os.listdir(all_file_path), key=lambda x: int(x.split('.')[0]))

    all_result_path = "../result_both"

    # generate unequally spread slice number sequences for plotting the feature curve because in this
    # assignment, keep increasing slices number when it is already 20+ or 30+ slices doesn't have
    # significant improvement on accuracy, so just save some time.
    slices_number = list(np.logspace(start=1, stop=int(np.log(100) / np.log(1.25)), num=15, base=1.25))
    slices_number = [int(num) for num in slices_number]
    print(slices_number)

    train_error = []
    test_error = []
    train_std = []
    test_std = []
    for i in range(len(slices_number)):
        slice_number = slices_number[i]
        print(f"Slices number: {slice_number}")
        result = get_result(f"{all_result_path}/X_{i}_{slice_number}.npy",
                              f"{all_result_path}/y.npy")
        train_error.append(1 - result[0])
        test_error.append(1 - result[1])
        train_std.append(result[2])
        test_std.append(result[3])

    title = "Feature Curve (Point number + Point density)"

    train_error = np.array(train_error)
    test_error = np.array(test_error)
    train_std = np.array(train_std)
    test_std = np.array(test_std)
    get_feature_curve(slices_number, test_error, test_std, title, show=False, save=True)
