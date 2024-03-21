from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

# Define training data points
X = [[i, i] for i in range(10)]

# Define labels: first 5 points as class 0 and next 5 points as class 1
y = [0] * 5 + [1] * 5

clf = svm.SVC()
clf.fit(X, y)

# Define a point to predict
prediction = clf.predict([[2., 4.]])
print(f"Prediction for [2., 4.]: {prediction}")

# Plotting
sns.set()

# Plot the training points (differentiating by label)
for i, point in enumerate(X):
    plt.scatter(point[0], point[1], color='red' if y[i] == 0 else 'blue', label=f'Class {y[i]}' if i == 0 or i == 5 else "")

# Plot the prediction point
plt.scatter(2, 4, color='green', label='Prediction')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Classification and Prediction")

# Ensure the legend does not repeat labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.grid(True)
plt.show()


