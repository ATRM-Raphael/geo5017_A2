import numpy as np
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# // LOAD THE DATASET HERE \\
# data = load

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# // Data training vs. test split \\ #
# TODO: Tweak parameters to explore different results
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60, test_size=0.40,
                                                                    random_state=101)

# ---------------------------------
# 3.-- EVALUATION AND ANALYSIS ----
# ---------------------------------

# // MODEL TRAINING \\

# -- Support Vector Machine --
poly = svm.SVC(kernel="poly", degree=3, C=1).fit(X_train, y_train)
rbf = svm.SVC(kernel="rbf", gamma=0.5, C=0.1).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)

# -- Random Forest --
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# // OVERALL ACCURACY \\

# Train SVM with polynomial kernel
poly_accuracy = accuracy_score(y_test, poly_pred)
print(f"Accuracy (Polynomial Kernel): {(poly_accuracy * 100):.2f}")

# Train SVM with RBF kernel
rbf_accuracy = accuracy_score(y_test, rbf_pred)
print(f"Accuracy (RBF Kernel): {(rbf_accuracy * 100):.2f}")

# Train Random Forest
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Accuracy (Random Forest): {(rf_accuracy * 100):.2f}")

# // MEAN PER-CLASS ACCURACY \\
def mean_per_class_accuracy(y_true, y_pred):
    unique_classes = np.unique(y_true)
    accuracy_sum = 0

    for cls in unique_classes:
        # For each class, calculate the accuracy
        cls_accuracy = np.mean(y_pred[y_true == cls] == cls)
        accuracy_sum += cls_accuracy

    # Compute the mean per-class accuracy
    return (accuracy_sum / len(unique_classes)) * 100


# Compute and print mean per-class accuracies
poly_mPCA = mean_per_class_accuracy(y_test, poly_pred)
print(f"Mean Per-Class Accuracy (Polynomial Kernel): {poly_mPCA:.2f}")

rbf_mPCA = mean_per_class_accuracy(y_test, rbf_pred)
print(f"Mean Per-Class Accuracy (RBF Kernel): {rbf_mPCA:.2f}")

rf_mPCA = mean_per_class_accuracy(y_test, rf_pred)
print(f"Mean Per-Class Accuracy (Random Forest): {rf_mPCA:.2f}")

# // CONFUSION MATRIX \\
# TODO: import assignment point cloud with the ground truth labels

# confusion_matrix(y_test, y_pred=)

# ---------------------------------
# ----------- PLOTTING ------------
# ---------------------------------

# Prepare data for plotting
scores_data = {
    'Model': ['SVM (Polynomial Kernel)', 'SVM (RBF Kernel)', 'Random Forest'],
    'Accuracy': [poly_accuracy * 100, rbf_accuracy * 100, rf_accuracy * 100],
    'Mean Per-Class Accuracy': [poly_mPCA, rbf_mPCA, rf_mPCA]
}

df_scores = pd.DataFrame(scores_data)

# Melt the DataFrame to plot with seaborn
df_melted = df_scores.melt(id_vars="Model", var_name="Metric", value_name="Value")

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted)
plt.title('Comparison of SVM Kernels and Random Forest on Iris Dataset')
plt.ylabel('% Score')
plt.xlabel('Model')
plt.ylim(0, 100)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

# # Loading the iris dataset
# iris = sns.load_dataset("iris")
#
# # Creating a pair plot
# sns.pairplot(iris, hue="species", height=2.5)
# plt.show()
