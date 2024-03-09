from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60, test_size=0.40,
                                                                    random_state=101)

poly = svm.SVC(kernel="poly", degree=3, C=1).fit(X_train, y_train)
rbf = svm.SVC(kernel="rbf", gamma=0.5, C=0.1).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)

poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average="weighted")
print(f"Accuracy (Polynomial Kernel): {(poly_accuracy * 100):.2f}")
print(f"F1 (Polynomial Kernel): {(poly_f1 * 100):.2f}")

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average="weighted")
print(f"Accuracy (RBF Kernel): {(rbf_accuracy * 100):.2f}")
print(f"F1 (RBF Kernel): {(rbf_f1 * 100):.2f}")


# // PLOTTING \\

# Create a DataFrame to hold the scores
scores_data = {
    'Kernel': ['Polynomial', 'RBF'],
    'Accuracy': [poly_accuracy * 100, rbf_accuracy * 100],
    'F1 Score': [poly_f1 * 100, rbf_f1 * 100]
}

df_scores = pd.DataFrame(scores_data)

# Melt the DataFrame to have proper format for seaborn barplot
df_melted = df_scores.melt(id_vars="Kernel", var_name="Metric", value_name="Value")

# # Create the barplot
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Metric', y='Value', hue='Kernel', data=df_melted)
# plt.title('Comparison of SVM Kernels on Iris Dataset')
# plt.ylabel('% Score')
# plt.xlabel('Metric')
# plt.ylim(0, 100)
# plt.legend(title='Kernel')
# plt.tight_layout()
# plt.show()

# Loading the iris dataset
iris = sns.load_dataset("iris")

# Creating a pair plot
sns.pairplot(iris, hue="species", height=2.5)
plt.show()

