from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]

# Create a range of x values
x_range = np.linspace(0, 3, 300).reshape(-1, 1)

# -- OVO --
# Initialize the classifier with 'ovo' and train
clf_ovo = svm.SVC(decision_function_shape='ovo')
clf_ovo.fit(X, Y)
# Get decision function values for the range of x values
dec_ovo = clf_ovo.decision_function(x_range)

# -- OVR --
# Change to 'ovr' and get decision function values
clf_ovr = svm.SVC(decision_function_shape='ovr')
clf_ovr.fit(X, Y)
dec_ovr = clf_ovr.decision_function(x_range)

# -- Plotting --
sns.set()
plt.figure(figsize=(10, 6))

# Plot for 'ovo'
for i in range(dec_ovo.shape[1]):
    plt.plot(x_range.flatten(), dec_ovo[:, i], label=f'ovo decision {i+1}')

# Plot for 'ovr'
for i in range(dec_ovr.shape[1]):
    plt.plot(x_range.flatten(), dec_ovr[:, i], '--', label=f'ovr decision {i+1}')

plt.title('Comparison of Decision Functions in SVM')
plt.xlabel('X value')
plt.ylabel('Decision function value')
plt.legend()
plt.show()
