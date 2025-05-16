"""
This Python file was used to train multiple models and to compare their best performances relative to each other.
The models trained were Decision Tree, Random Forest, KNN, Logistic Regression, and MLP.
The models were trained using GridSearchCV to find the best hyperparameters for each model.
The models were evaluated using various metrics such as accuracy, precision, recall, F1 score, and confusion matrix.
The results were saved to a folder along with the code used to generate the results.

Our final RF model parameters were determined this way.
"""

import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import contextlib
import shutil
import os
import m2cgen as m2c

# https://medium.com/@sahin.samia/scikit-learn-pipelines-explained-streamline-and-optimize-your-machine-learning-processes-f17b1beb86a4
# https://www.freecodecamp.org/news/machine-learning-pipeline/

# Globals -----------------------------------------------------------------------------------------------
random_state=14 # For reproducibility

# Loding data -------------------------------------------------------------------------------------------
np.random.seed(random_state)

df = pd.read_csv("preprocessed_data.csv")

X = df.drop(columns=["Label"])
y = df["Label"]

# Split data into training and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state)

# Global print capture buffer
summary_output_buffer = io.StringIO()

# Print dataset sizes
print("Dataset Sizes:")
print(f"Training Set:     X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Validation Set:   X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"Test Set:         X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# Storing metrics ---------------------------------------------------------------------------------------
# modified by collect_metrics
model_metrics = {
    "Model": [],
    "Inference Time": [],
    "Training Time": [],
    "Train Accuracy": [],
    "Validation Accuracy": [],
    "Test Accuracy": [],
    "CV Mean": [],
    "CV Std": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": [],
    "Confusion Matrix": []
}

# Helper functions --------------------------------------------------------------------------------------
def collect_metrics(name, model, best_params, training_time, printout = True):
    global summary_output_buffer

    model_metrics["Model"].append(name)
    
    # Calculate inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time)

    # Calculate accuracies
    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    test_accuracy = model.score(X_test, y_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Capture and print out the information if required
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        if printout:
            print(f"[{name}] Best Parameters:", best_params)
            print(f"[{name}] Inference Time: {inference_time*1000:.4f} ms")
            print(f"[{name}] Training Time: {training_time*1000:.4f} ms") 
            print(f"[{name}] Train Accuracy: {train_accuracy*100:.2f}%")
            print(f"[{name}] Validation Accuracy: {val_accuracy*100:.2f}% ({(val_accuracy-train_accuracy)*100:.2f} From Train)")
            print(f"[{name}] Test Accuracy: {test_accuracy*100:.2f}% ({(test_accuracy-train_accuracy)*100:.2f} From Train)")
            print(f"[{name}] 5-Fold Cross-Validation Accuracy: Mean[{cv_scores.mean():.5f}] +- Std[{cv_scores.std():.5f}]")
            print(f"[{name}] Accuracy: {accuracy:.5f}")
            print(f"[{name}] Precision: {precision:.5f}")
            print(f"[{name}] Recall: {recall:.5f}")
            print(f"[{name}] F1 Score: {f1:.5f}")
        else:
            print(f"Collected metrics for {name}...")
    
    # Print captured output
    captured_output = output.getvalue()
    print(captured_output, end='')

    # Store output in summary buffer
    summary_output_buffer.write(captured_output)

    # Store metrics
    model_metrics["Inference Time"].append(inference_time)
    model_metrics["Training Time"].append(training_time)
    model_metrics["Train Accuracy"].append(train_accuracy)
    model_metrics["Validation Accuracy"].append(val_accuracy)
    model_metrics["Test Accuracy"].append(test_accuracy)
    model_metrics["CV Mean"].append(cv_scores.mean())
    model_metrics["CV Std"].append(cv_scores.std())
    model_metrics["Accuracy"].append(accuracy)
    model_metrics["Precision"].append(precision)
    model_metrics["Recall"].append(recall)
    model_metrics["F1 Score"].append(f1)
    model_metrics["Confusion Matrix"].append(cm)

# Lab 4
def plot_confusion_matrices(df):
    """
    Plots the confusion matrices for each model in the DataFrame via the "Confusion Matrix" column.
    """

    # Extract data from the dataframe conversion of metrics dictionary
    model_names = df.index.tolist()
    confusion_matrices = df["Confusion Matrix"].tolist()

    class_labels = np.unique(y_test)

    # 3 columns, as many rows as needed
    n_models = len(model_names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, (name, cm) in enumerate(zip(model_names, confusion_matrices)):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=axes[i], values_format="d", colorbar=False)
        axes[i].set_title(f"Confusion Matrix for {name}")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig

from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
def plot_training_curves(models, model_names, X, y, cv=5, scoring='accuracy'):
    """
    Plots the learning curves for each model specified
    """
    # 3 columns, as many rows as needed
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    # For each model:
    for i, (model, name) in enumerate(zip(models, model_names)):
        # Deal with int/none https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            train_sizes=np.linspace(0.1, 1.0, 20),
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score=np.nan
        )

        # Training and validation mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        ax = axes[i]
        ax.set_title(f"Learning Curve: {name}")
        ax.set_xlabel("Training Examples")
        ax.set_ylabel(scoring.capitalize())
        ax.grid(True)
        ax.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
        ax.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1)
        ax.plot(train_sizes, train_mean, 'o-', label="Training Score")
        ax.plot(train_sizes, val_mean, 'o-', label="Validation Score")
        ax.legend(loc="best")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig

import matplotlib.ticker as mticker
def plot_compare_metrics(df, columns=["CV Mean", "CV Std", "Accuracy", "Precision", "Recall", "F1 Score"]):
    """
    Plots a bar chart comparing the specified metrics across all models in the DataFrame.
    """
    if columns:
        # Filter to specific olumns
        df_to_plot = df[columns]
    else:
        # Otherwise use numerical columns
        df_to_plot = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(12, 6))
    df_to_plot.plot(kind="bar", edgecolor="black", ax=ax)

    ax.set_title("Model Comparison Across Evaluation Metrics")
    ax.set_ylabel("Score")
    ax.set_xticklabels(df_to_plot.index, rotation=45)
    ax.set_ylim(0, 1)

    # Compare every 5%
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, 0.5))
    fig.tight_layout()

    return fig

def export_results(figures, metrics_df, printout_text, base_folder="saves"):
    import os

    # Create the base folder if it doesn't exist
    os.makedirs(base_folder, exist_ok=True)

    # Find next available save folder name (save1, save2, ...)
    i = 1
    while os.path.exists(os.path.join(base_folder, f"save{i}")):
        i += 1
    save_path = os.path.join(base_folder, f"save{i}")
    os.makedirs(save_path)

    # Save figures
    for j, fig in enumerate(figures, 1):
        fig.savefig(os.path.join(save_path, f"figure{j}.png"), bbox_inches="tight")

    # Save metrics to a text file
    with open(os.path.join(save_path, "metrics_summary.txt"), "w") as f:
        f.write(printout_text)
        f.write("\n\nFull Metrics Table:\n")
        f.write(metrics_df.to_string())

    for filename in ["ml_model_alt.py", "full_script.py"]:
        src_path = os.path.join(os.getcwd(), filename)
        dst_path = os.path.join(save_path, filename)
        shutil.copy(src_path, dst_path)

    print(f"\nResults saved to: {save_path}")
    
# Decision Tree ----------------------------------------------------------------------------------------

# Grid search Hyperparameter options
dt_param_grid = {
    'max_depth': [3, 5, 7, 9, 11, 13],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

dt = DecisionTreeClassifier(random_state=random_state)

dt_grid_search = GridSearchCV(dt, dt_param_grid, cv=5, scoring='accuracy',n_jobs=-1)
start_time = time.time()
dt_grid_search.fit(X_train, y_train)
training_time = (time.time() - start_time)

best_dt = dt_grid_search.best_estimator_
best_dt_params = dt_grid_search.best_params_

collect_metrics("Decision Tree", best_dt, best_dt_params, training_time)

# Random Forest ----------------------------------------------------------------------------------------
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [1, 2, 5, 10, 15],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'criterion': ['entropy', 'gini']
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=random_state)

rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
start_time = time.time()
rf_grid_search.fit(X_train, y_train)
training_time = (time.time() - start_time)

best_rf = rf_grid_search.best_estimator_
best_rf_params = rf_grid_search.best_params_

collect_metrics("Random Forest", best_rf, best_rf_params, training_time)

# KNN -------------------------------------------------------------------------------------------------
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Grid search Hyperparameter options
knn_param_grid = {
    'knn__n_neighbors': [1, 3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}   

# Grid search for KNN using 5-fold cross validation
knn_grid_search = GridSearchCV(knn_pipeline,knn_param_grid,scoring='accuracy',cv=5,n_jobs=-1)
start_time = time.time()
knn_grid_search.fit(X_train, y_train)
training_time = (time.time() - start_time)

best_knn = knn_grid_search.best_estimator_
best_knn_params = knn_grid_search.best_params_

collect_metrics("KNN", best_knn, best_knn_params, training_time)

# Logistic Regression ----------------------------------------------------------------------------------
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(solver='liblinear', random_state=random_state))
])

# Grid search Hyperparameter options
logreg_param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__penalty': ['l1', 'l2']
}

# Grid search for Logistic Regression using 5-fold cross validation
logreg_grid_search = GridSearchCV(logreg_pipeline,logreg_param_grid,scoring='accuracy',cv=5,n_jobs=-1)
start_time = time.time()
logreg_grid_search.fit(X_train, y_train)
training_time = (time.time() - start_time)

best_logreg = logreg_grid_search.best_estimator_
best_logreg_params = logreg_grid_search.best_params_

collect_metrics("Log Reg", best_logreg, best_logreg_params, training_time)

# MLP -------------------------------------------------------------------------------------------------
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(random_state=random_state))
])

# Grid search Hyperparameter options
mlp_param_grid = {
    'mlp__hidden_layer_sizes': [
        (32,), (64,), (128,),
        (32, 32), (64, 64), (128, 128)
    ],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__learning_rate_init': [0.001, 0.01, 0.1],
    'mlp__early_stopping': [True],
    'mlp__max_iter': [200, 500, 1000],
}

# Grid search for MLP using 5-fold cross validation
grid_search = GridSearchCV(mlp_pipeline,mlp_param_grid,scoring='accuracy',cv=5,n_jobs=-1)
start_time = time.time()
grid_search.fit(X_train, y_train)
training_time = (time.time() - start_time)

best_mlp = grid_search.best_estimator_
best_params = grid_search.best_params_

collect_metrics("MLP", best_mlp, best_params, training_time)

# ------------------------------------------------------------------------------------------------------

# Create DataFrame from collected metrics
metrics_df = pd.DataFrame(model_metrics)
metrics_df.set_index("Model", inplace=True)

models = [best_dt, best_rf, best_knn, best_logreg, best_mlp]
model_names = ["Decision Tree", "Random Forest", "KNN", "Logistic Regression", "MLP"]
metric_names = ["Train Accuracy", "Validation Accuracy", "Test Accuracy", "CV Mean", "CV Std", "Accuracy", "Precision", "Recall", "F1 Score"]

fig3 = plot_training_curves(models, model_names, X, y)
fig2 = plot_confusion_matrices(metrics_df)
fig1 = plot_compare_metrics(metrics_df, metric_names)

plt.show()

best_val_score = rf_grid_search.best_score_
print("RF Highest Validation Score Achieved During Grid Search", best_val_score)

summary_output = summary_output_buffer.getvalue()

# If the user enters S, save the results
choice = input("\nEnter 'S' to save results: ").strip().upper()
if choice == "S":
    export_results(
        figures=[fig1, fig2, fig3],
        metrics_df=metrics_df,
        printout_text=summary_output
    )