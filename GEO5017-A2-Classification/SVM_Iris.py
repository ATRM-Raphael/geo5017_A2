"""
This demo is an extension of the codes we show during the lab session. It is provided in: https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
4 classifiers are constructed, svc with linear kernel, linear svc, rbf svc, and poly svc.
Run the code to perform classification on Iris dataset and visualize the decision boundaries.
Note that you need a new version of scikit-learn to run the code (>=1.1)
"""

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay


# import Iris data
iris = datasets.load_iris()
# Take the first two features
X = iris.data[:, :2]
# Take the labels
y = iris.target

# SVM regularization parameter
C = 1.0  
# Initialize a sequence of svc models. You can play with the hyperparameters
models = (
    svm.SVC(kernel="linear", C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
)

# Fit the model with the total data. Note that in practice, you need to perform train test split to evaluate the model performance!
models = (clf.fit(X, y) for clf in models)

# Plotting setup
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]

# Visualize the 4 decision boundaries
for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

# Show
plt.show()
