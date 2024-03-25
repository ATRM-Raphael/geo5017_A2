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
    xyz_files_sorted = sorted(xyz_files, key=lambda x: int(x.split('.')[0]))

    # Return both the DataFrame and the sorted list of file names
    return pd.concat([load_xyz_file(os.path.join(folder_path, file)) for file in xyz_files_sorted],
                     ignore_index=True), xyz_files_sorted


# // LOAD THE DATASET HERE \\

# Specify the path to your folder
folder_path = '../pointclouds-500'
point_cloud_data, xyz_files_sorted = load_point_cloud_data(folder_path)
all_results_path = "../result_both"

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


# ---------------------------------
# 2.----- FEATURE DEFINITION ------
# ---------------------------------

# // FEATURE ENGINEERING \\
slices_number = list(np.logspace(start=1, stop=int(np.log(100) / np.log(1.25)), num=15, base=1.25))
slices_number = [int(num) for num in slices_number]

for i in range(len(slices_number)):
    n = slices_number[i]
    print(f"Slice: {n}")
    X = []  # 500 feature vectors
    for j, file_name in enumerate(xyz_files_sorted):
        file_path = os.path.join(folder_path, file_name)
        features = feature_engineering.get_all_features(file_path, n, "both")
        X.append(features)
        # Just to show the progress
        if j % 50 == 0 or j == len(xyz_files_sorted) - 1:
            print(f"Processed file {j + 1}: {file_name}")
    X = np.array(X)
    np.save(f"{all_results_path}/X_{i}_{n}.npy", X, allow_pickle=True, fix_imports=True)

# # -- fix this to make the code more dynamic and robust
# y = point_cloud_data['label']
# le = LabelEncoder()  # Use numerical indexing, in case it is required
# y_encoded = le.fit_transform(y)
# np.save(f"{all_results_path}/y.npy", y_encoded, allow_pickle=True, fix_imports=True)

# -- hardcoded labels
Y = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100  # 500 labels
Y = np.array(Y)
np.save(f"{all_results_path}/y.npy", Y, allow_pickle=True, fix_imports=True)

# // FEATURE CURVE \\
print(slices_number)

train_error = []
test_error = []
train_std = []
test_std = []
for i in range(len(slices_number)):
    slice_number = slices_number[i]
    print(f"Slices number: {slice_number}")
    result = feature_curve.get_result(f"{all_results_path}/X_{i}_{slice_number}.npy",
                                      f"{all_results_path}/y.npy")
    train_error.append(1 - result[0])
    test_error.append(1 - result[1])
    train_std.append(result[2])
    test_std.append(result[3])

title_fc = "Feature Curve (Point number + Point density)"

train_error = np.array(train_error)
test_error = np.array(test_error)
train_std = np.array(train_std)
test_std = np.array(test_std)
feature_curve.get_feature_curve(slices_number, test_error, test_std, title_fc, show=True, save=True)

# // SAVE THE RESULT WITH THE LOWEST TEST ERROR \\
min_test_error_idx = np.argmin(test_error)
optimal_slice_number = slices_number[min_test_error_idx]

optimal_features_filename = f"{all_results_path}/X_{min_test_error_idx}_{optimal_slice_number}.npy"

# Define X - features, AND y - labels:
# X = np.load("../result_both/X_4_4.npy")
X = np.load(optimal_features_filename)
y = np.load("../result_both/y.npy")
# y = point_cloud_data['label']

# // Data training vs. test split \\ #
# TODO: Tweak parameters to explore different results
train_size = 0.6
test_size = 1 - train_size

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=train_size, test_size=test_size,
                                                                    random_state=101)

# ---------------------------------
# 3.--- PREPARATION AND TUNING ----
# ---------------------------------

np.random.seed(101)

# // SVM TUNING OF HYPER PARAMETERS \\
print("SVM Tuning: Poly")
poly_accuracy_best, poly_c_best, poly_degree_best, poly_pred_test_best, poly_results_grid = svm_tuning.poly_gridsearch(
    X_train, X_test, y_train, y_test, False, False)
print("\n")

print("SVM Tuning: RBF")
rbf_accuracy_best, rbf_c_best, rbf_gamma_best, rbf_pred_test_best, rbf_results_grid = svm_tuning.rbf_gridsearch(X_train,
                                                                                                                X_test,
                                                                                                                y_train,
                                                                                                                y_test,
                                                                                                                False,
                                                                                                                False)
print("\n")

# // RANDOM FOREST TUNING OF HYPER PARAMETERS \\
print("RF Tuning")
rf_best_estimators, rf_best_min_samples_leaf, rf_accuracy_best, rf_pred_best = rf_tuning.rf_gridsearch(X_train,
                                                                                                       X_test,
                                                                                                       y_train,
                                                                                                       y_test,
                                                                                                       True,
                                                                                                       False)
print("\n")

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
rf_base = RandomForestClassifier(n_estimators=rf_best_estimators, min_samples_leaf=rf_best_min_samples_leaf,
                                 random_state=101)

rf_base.fit(X_train, y_train)
rf_pred = rf_base.predict(X_test)

# ---------------------------------
# 4.--- EVALUATION AND ANALYSIS ---
# ---------------------------------

# // OVERALL ACCURACY \\

poly_accuracy = accuracy_score(y_test, poly_pred)
print(f"Accuracy (Polynomial Kernel): {(poly_accuracy * 100):.2f}")

rbf_accuracy = accuracy_score(y_test, rbf_pred)
print(f"Accuracy (RBF Kernel): {(rbf_accuracy * 100):.2f}")

rf_accuracy = accuracy_score(y_test, rf_pred)
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

rf_mPCA = mean_per_class_accuracy(y_test, rf_pred)
print(f"Mean Per-Class Accuracy (Random Forest with tuning): {rf_mPCA:.2f}")

print("\n")

# # ---------------------------------
# # 5.-------- VALIDATION -----------
# # ---------------------------------

# // K-FOLD CROSS-VALIDATION\\

# Initialize untrained model instances for cross-validation
poly_model_cv = svm.SVC(kernel="poly", degree=poly_degree_best, C=poly_c_best)
rbf_model_cv = svm.SVC(kernel="rbf", gamma=rbf_gamma_best, C=rbf_c_best)
rf_model_cv = RandomForestClassifier()

# Perform k-fold cross-validation for each model and print the results
score_poly_cv = cross_val_score(poly_model_cv, X, y, cv=5)
score_rbf_cv = cross_val_score(rbf_model_cv, X, y, cv=5)
score_rf_cv = cross_val_score(rf_model_cv, X, y, cv=5)

# Output the mean and standard deviation of the cross-validation scores for each model
print(
    f"Cross Validation -- Polynomial Kernel SVM Mean Accuracy: {(score_poly_cv.mean() * 100):0.2f}, CV Std: {(score_poly_cv.std() * 100):0.2f}")
print(
    f"Cross Validation -- RBF Kernel SVM Mean Accuracy: {(score_rbf_cv.mean() * 100):0.2f}, CV Std: {(score_rbf_cv.std() * 100):0.2f}")
print(
    f"Cross Validation -- Random Forest Mean Accuracy: {(score_rf_cv.mean() * 100):0.2f}, CV Std: {(score_rf_cv.std() * 100):0.2f}")

print("\n")

# // CONFUSION MATRIX \\

# Confusion matrices plotting
models = [('Polynomial Kernel SVM', poly_pred), ('RBF Kernel SVM', rbf_pred), ('Random Forest', rf_pred)]
for (model_name, model_pred) in models:
    cm_fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, model_pred)
    sns.heatmap(cm, annot=True, fmt='d')

    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # plt.show()
    cm_fig.savefig(f'../figures/ConfusionMatrix_{model_name}.png', dpi=200)

# # // CLASSIFICATION REPORTS \\
#
# print("Detailed classification report for Polynomial Kernel SVM:")
# print(classification_report(y_test, poly_pred))
#
# print("Detailed classification report for RBF Kernel SVM:")
# print(classification_report(y_test, rbf_pred))
#
# print("Detailed classification report for Random Forest:")
# print(classification_report(y_test, rf_pred))

# # ---------------------------------
# # ------ PLOT LEARNING CURVE ------
# # ---------------------------------

svm_title = "Learning Curve (SVM: POLY)"
rbf_title = "Learning Curve (SVM: RBF)"
rf_title = "Learning Curve (RF)"

learning_curve.get_learning_curve(X, Y, poly, title=svm_title, show=True, save=True)

learning_curve.get_learning_curve(X, Y, rbf, title=rbf_title, show=True, save=True)

learning_curve.get_learning_curve(X, y, rf_base, title=rf_title, show=True, save=True)