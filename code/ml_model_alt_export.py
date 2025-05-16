"""
This Python file was used to export the best RandomForest model to model.py.
"""

import time
import pandas as pd
import numpy as np
import m2cgen as m2c
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys

################################################################################
# 1. Load the preprocessed data
################################################################################

df = pd.read_csv("preprocessed_data.csv")  # Created by your full_script.py
X = df.drop(columns=["Label"])
y = df["Label"]

################################################################################
# 2. Split into training and test sets
################################################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

################################################################################
# 3. Use the best hyperparameters you found
################################################################################

best_params = {
    'bootstrap': True,
    'criterion': 'gini',
    'max_depth': 10,
    'max_features': 'log2',
    'min_samples_leaf': 1,
    'min_samples_split': 15,
    'n_estimators': 300,
    'random_state': 42
}

best_rf = RandomForestClassifier(**best_params)

################################################################################
# 4. Train the RandomForest with the best parameters
################################################################################

print("Training RandomForest with best hyperparameters...")
start_time = time.time()
best_rf.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

################################################################################
# 5. Evaluate on training and test sets
################################################################################

# Training accuracy
y_train_pred = best_rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"RandomForest Train Accuracy: {train_accuracy:.3f}")

# Test accuracy
y_test_pred = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"RandomForest Test Accuracy: {test_accuracy:.3f}")

################################################################################
# 6. Export the model to model.py using m2cgen
################################################################################

model_code = m2c.export_to_python(best_rf)
with open("model.py", "w") as f:
    f.write(model_code)

print("Exported best RandomForest model to model.py")
