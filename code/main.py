import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import sklearn.model_selection as model_selection
from sklearn.preprocessing import LabelEncoder

import learning_curve
import svm_tuning
import rf_tuning
import feature_curve
import feature_engineering


# ---------------------------------
# 1.---- FILE PREPARATION ---------
# ---------------------------------

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


# // LOAD THE DATASET HERE \\

# Specify the path to your folder
folder_path = '../pointclouds-500/pointclouds-500-cropped/'
point_cloud_data = load_point_cloud_data(folder_path)

# # -- Visualise the data --
# # >> WARNING: Computationally heavy
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# categories = point_cloud_data['label'].unique()
# for category in categories:
#     subset = point_cloud_data[point_cloud_data['label'] == category]
#     ax.scatter(subset['x'], subset['y'], subset['z'], label=category)
# ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('Urban Object Point Clouds')
# plt.show()

# Define X - features, AND y - labels:
X = np.load("../result_both/X_4_4.npy")
y = np.load("../result_both/Y.npy")

# X = point_cloud_data[['x', 'y', 'z']]
# y = point_cloud_data['label']

le = LabelEncoder()  # Use numerical indexing, in case it is required
y_encoded = le.fit_transform(y)

# // Data training vs. test split \\ #
# TODO: Tweak parameters to explore different results
train_size = 0.6
test_size = 1 - train_size

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=train_size, test_size=test_size,
                                                                    random_state=101)

# ---------------------------------
# 2.--- PREPARATION AND TUNING ----
# ---------------------------------

np.random.seed(101)

# // SVM TUNING OF HYPER PARAMETERS \\
poly_accuracy_best, poly_c_best, poly_degree_best, poly_pred_test_best, poly_results_grid = svm_tuning.poly_gridsearch(
    X_train, X_test, y_train, y_test, False, False)

rbf_accuracy_best, rbf_c_best, rbf_gamma_best, rbf_pred_test_best, rbf_results_grid = svm_tuning.rbf_gridsearch(X_train,
                                                                                                                X_test,
                                                                                                                y_train,
                                                                                                                y_test,
                                                                                                                False,
                                                                                                                False)

# // RANDOM FOREST TUNING OF HYPER PARAMETERS \\
rf_best_estimators, rf_best_min_samples_leaf, rf_accuracy_best, rf_pred_best = rf_tuning.rf_gridsearch(X_train,
                                                                                                       X_test,
                                                                                                       y_train,
                                                                                                       y_test,
                                                                                                       True,
                                                                                                       False)

# -- SUMMARY --
print(f">> SVM POLY:\n"
      f"POLY: Best Accuracy: {poly_accuracy_best}\n"
      f"Best C Value: {poly_c_best}\n"
      f"Best Degree/Second Parameter: {poly_degree_best}\n"
      f"Confusion Matrix:\n"
      f"{confusion_matrix(y_test, poly_pred_test_best)}"
      "\n"
      )

print(f">> SVM RBF:\n"
      f"Best Accuracy: {rbf_accuracy_best}\n"
      f"Best C Value: {rbf_c_best}\n"
      f"Best Gamma: {rbf_gamma_best}\n"
      f"Confusion Matrix:\n"
      f"{confusion_matrix(y_test, rbf_pred_test_best)}"
      "\n"
      )

print(f">> RANDOM FOREST:\n"
      f"Best Number of Estimators: {rf_best_estimators}\n"
      f"Best Min Samples Leaf: {rf_best_min_samples_leaf}\n"
      f"Best Accuracy: {rf_accuracy_best}\n"
      f"Confusion Matrix:\n{confusion_matrix(y_test, rf_pred_best)}"
      "\n"
      )

# // MODEL TRAINING AND TESTING \\
# -- SVM --
poly = svm.SVC(kernel="poly", degree=poly_degree_best, C=poly_c_best).fit(X_train, y_train)
rbf = svm.SVC(kernel="rbf", gamma=rbf_gamma_best, C=rbf_c_best).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)

# -- RF --


# ---------------------------------
# 3.-- EVALUATION AND ANALYSIS ----
# ---------------------------------

# // OVERALL ACCURACY \\

poly_accuracy = accuracy_score(y_test, poly_pred)
print(f"Accuracy (Polynomial Kernel): {(poly_accuracy * 100):.2f}")

rbf_accuracy = accuracy_score(y_test, rbf_pred)
print(f"Accuracy (RBF Kernel): {(rbf_accuracy * 100):.2f}")

rf_accuracy = accuracy_score(y_test, rf_pred_best)
print(f"Accuracy (Random Forest with tuning): {(rf_accuracy * 100):.2f}")

print("\n")

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

rf_mPCA = mean_per_class_accuracy(y_test, rf_pred_best)
print(f"Mean Per-Class Accuracy (Random Forest with tuning): {rf_mPCA:.2f}")

print("\n")


# // K-FOLD CROSS-VALIDATION\\

# Initialize untrained model instances for cross-validation
poly_model_cv = svm.SVC(kernel="poly", degree=3, C=1)
rbf_model_cv = svm.SVC(kernel="rbf", gamma=0.5, C=0.1)
rf_model_cv = RandomForestClassifier()

# Perform k-fold cross-validation for each model and print the results
score_poly_cv = cross_val_score(poly_model_cv, X, y_encoded, cv=3)
score_rbf_cv = cross_val_score(rbf_model_cv, X, y_encoded, cv=3)
score_rf_cv = cross_val_score(rf_model_cv, X, y_encoded, cv=3)

# Output the mean and standard deviation of the cross-validation scores for each model
print(
    f"Cross Validation -- Polynomial Kernel SVM Mean Accuracy: {(score_poly_cv.mean() * 100):0.2f}, CV Std: {(score_poly_cv.std() * 100):0.2f}")
print(
    f"Cross Validation -- RBF Kernel SVM Mean Accuracy: {(score_rbf_cv.mean() * 100):0.2f}, CV Std: {(score_rbf_cv.std() * 100):0.2f}")
print(
    f"Cross Validation -- Random Forest Mean Accuracy: {(score_rf_cv.mean() * 100):0.2f}, CV Std: {(score_rf_cv.std() * 100):0.2f}")

print("\n")


# // CONFUSION MATRIX \\
# TODO: import assignment point cloud with the ground truth labels

# # Confusion matrices plotting
# models = [('Polynomial Kernel SVM', poly_pred), ('RBF Kernel SVM', rbf_pred), ('Random Forest', rf_pred_best)]
# for (model_name, model_pred) in models:
#     plt.figure(figsize=(8, 6))
#     cm = confusion_matrix(y_test, model_pred)
#     sns.heatmap(cm, annot=True, fmt='d')
#
#     plt.title(f'Confusion Matrix for {model_name}')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#
#     plt.show()

# # // CLASSIFICATION REPORTS \\
#
# print("Detailed classification report for Polynomial Kernel SVM:")
# print(classification_report(y_test, poly_pred))
#
# print("Detailed classification report for RBF Kernel SVM:")
# print(classification_report(y_test, rbf_pred))
#
# print("Detailed classification report for Random Forest:")
# print(classification_report(y_test, rf_pred_best))

# # ---------------------------------
# # ----------- PLOTTING ------------
# # ---------------------------------

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
