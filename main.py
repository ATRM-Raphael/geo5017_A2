import os
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
from mpl_toolkits.mplot3d import Axes3D


# // LOAD THE DATASET HERE \\
def assign_label(filename):
    # Extract the base name and convert to an integer to assign labels
    base_name = int(filename.split('.')[0])
    if 0 <= base_name <= 99:
        return 'building'
    elif 100 <= base_name <= 199:
        return 'car'
    elif 200 <= base_name <= 299:
        return 'fence'
    elif 300 <= base_name <= 399:
        return 'pole'
    elif 400 <= base_name <= 499:
        return 'tree'
    else:
        return 'unknown'

def load_xyz_file(filepath):
    # Load a single .xyz file and assign a label based on its filename
    df = pd.read_csv(filepath, sep=' ', header=None, names=['x', 'y', 'z'])
    label = assign_label(os.path.basename(filepath))
    df['label'] = label
    return df

def load_point_cloud_data(folder_path):
    # Aggregate labeled .xyz files in the folder into a single DataFrame
    xyz_files = [f for f in os.listdir(folder_path) if f.endswith('.xyz')]
    return pd.concat([load_xyz_file(os.path.join(folder_path, file)) for file in xyz_files], ignore_index=True)

# Specify the path to your folder
folder_path = 'GEO5017-A2-Classification/pointclouds-500/pointclouds-500/'
point_cloud_data = load_point_cloud_data(folder_path)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
categories = point_cloud_data['label'].unique()
for category in categories:
    subset = point_cloud_data[point_cloud_data['label'] == category]
    ax.scatter(subset['x'], subset['y'], subset['z'], label=category)
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Urban Object Point Clouds')
plt.show()


# # iris = datasets.load_iris()
# # X = iris.data[:, :2]
# # y = iris.target
#
# # // Data training vs. test split \\ #
# # TODO: Tweak parameters to explore different results
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60, test_size=0.40,
#                                                                     random_state=101)
#
# # ---------------------------------
# # 3.-- EVALUATION AND ANALYSIS ----
# # ---------------------------------
#
# # // MODEL TRAINING AND TESTING \\
#
# # -- Support Vector Machine --
# poly = svm.SVC(kernel="poly", degree=3, C=1).fit(X_train, y_train)
# rbf = svm.SVC(kernel="rbf", gamma=0.5, C=0.1).fit(X_train, y_train)
#
# poly_pred = poly.predict(X_test)
# rbf_pred = rbf.predict(X_test)
#
# # -- Random Forest --
# rf = RandomForestClassifier().fit(X_train, y_train)
#
# rf_pred = rf.predict(X_test)
#
#
# # // OVERALL ACCURACY \\
#
# poly_accuracy = accuracy_score(y_test, poly_pred)
# print(f"Accuracy (Polynomial Kernel): {(poly_accuracy * 100):.2f}")
#
# rbf_accuracy = accuracy_score(y_test, rbf_pred)
# print(f"Accuracy (RBF Kernel): {(rbf_accuracy * 100):.2f}")
#
# rf_accuracy = accuracy_score(y_test, rf_pred)
# print(f"Accuracy (Random Forest): {(rf_accuracy * 100):.2f}")
#
# # // MEAN PER-CLASS ACCURACY \\
# def mean_per_class_accuracy(y_true, y_pred):
#     unique_classes = np.unique(y_true)
#     accuracy_sum = 0
#
#     for cls in unique_classes:
#         # For each class, calculate the accuracy
#         cls_accuracy = np.mean(y_pred[y_true == cls] == cls)
#         accuracy_sum += cls_accuracy
#
#     # Compute the mean per-class accuracy
#     return (accuracy_sum / len(unique_classes)) * 100
#
#
# # Compute and print mean per-class accuracies
# poly_mPCA = mean_per_class_accuracy(y_test, poly_pred)
# print(f"Mean Per-Class Accuracy (Polynomial Kernel): {poly_mPCA:.2f}")
#
# rbf_mPCA = mean_per_class_accuracy(y_test, rbf_pred)
# print(f"Mean Per-Class Accuracy (RBF Kernel): {rbf_mPCA:.2f}")
#
# rf_mPCA = mean_per_class_accuracy(y_test, rf_pred)
# print(f"Mean Per-Class Accuracy (Random Forest): {rf_mPCA:.2f}")
#
# # // CONFUSION MATRIX \\
# # TODO: import assignment point cloud with the ground truth labels
#
# # # Confusion matrices plotting
# # models = [('Polynomial Kernel SVM', poly_pred), ('RBF Kernel SVM', rbf_pred), ('Random Forest', rf_pred)]
# # for (model_name, model_pred) in models:
# #     plt.figure(figsize=(8, 6))
# #     cm = confusion_matrix(y_test, model_pred)
# #     sns.heatmap(cm, annot=True, fmt='d')
#
# #     plt.title(f'Confusion Matrix for {model_name}')
# #     plt.xlabel('Predicted Label')
# #     plt.ylabel('True Label')
#
# #     plt.show()
#
# # ---------------------------------
# # ----------- PLOTTING ------------
# # ---------------------------------
#
# # Prepare data for plotting
# scores_data = {
#     'Model': ['SVM (Polynomial Kernel)', 'SVM (RBF Kernel)', 'Random Forest'],
#     'Accuracy': [poly_accuracy * 100, rbf_accuracy * 100, rf_accuracy * 100],
#     'Mean Per-Class Accuracy': [poly_mPCA, rbf_mPCA, rf_mPCA]
# }
#
# df_scores = pd.DataFrame(scores_data)
#
# # Melt the DataFrame to plot with seaborn
# df_melted = df_scores.melt(id_vars="Model", var_name="Metric", value_name="Value")
#
# # Create the plot
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted)
# plt.title('Comparison of SVM Kernels and Random Forest on Iris Dataset')
# plt.ylabel('% Score')
# plt.xlabel('Model')
# plt.ylim(0, 100)
# plt.legend(title='Metric')
# plt.tight_layout()
# plt.show()
#
# # # Loading the iris dataset
# # iris = sns.load_dataset("iris")
# #
# # # Creating a pair plot
# # sns.pairplot(iris, hue="species", height=2.5)
# # plt.show()
