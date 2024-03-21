# -- Random Forest with Hyperparameter Tuning --
# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the random forest
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'max_depth': [None, 10, 20],  # Maximum number of levels in tree
}

# Initialize a base Random Forest model
rf_base = RandomForestClassifier()

# Set up the grid search with cross-validation
grid_search_rf = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search_rf.fit(X_train, y_train)

# Retrieve the best Random Forest model from grid search
rf_best = grid_search_rf.best_estimator_

# Make predictions with the best model
rf_pred = rf_best.predict(X_test)

# Print the best parameters found by GridSearchCV
print("Best parameters found by grid search:", grid_search_rf.best_params_)

# -- Evaluation Metrics for Random Forest --
# Calculate and print overall accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Accuracy (Random Forest with tuning): {(rf_accuracy * 100):.2f}")

# Calculate and print mean per-class accuracy
rf_mPCA = mean_per_class_accuracy(y_test, rf_pred)
print(f"Mean Per-Class Accuracy (Random Forest with tuning): {rf_mPCA:.2f}")

# Print classification report for detailed analysis
print("Detailed classification report for Random Forest:")
print(classification_report(y_test, rf_pred))