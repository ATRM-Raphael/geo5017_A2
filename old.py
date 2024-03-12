# Train SVM with polynomial kernel
poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average="weighted")
print(f"Accuracy (Polynomial Kernel): {(poly_accuracy * 100):.2f}")
print(f"F1 (Polynomial Kernel): {(poly_f1 * 100):.2f}")

# Train SVM with RBF kernel
rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average="weighted")
print(f"Accuracy (RBF Kernel): {(rbf_accuracy * 100):.2f}")
print(f"F1 (RBF Kernel): {(rbf_f1 * 100):.2f}")

# Train Random Forest
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="weighted")

print(f"Accuracy (Random Forest): {(rf_accuracy *100):.2f}")
print(f"F1 (Random Forest): {(rf_f1 * 100):.2f}")