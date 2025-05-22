# CSC311-ML CLasisfier

## Overview
**ML CLasisfier** is a machine learning project completed for the Winter 2025 *CSC311 Machine Learning Challenge* at the University of Toronto. The aim of this project was to build a classifier that predicts which of three food items a student is referring to (Pizza, Shawarma, or Sushi) based on their responses to a survey. This repository contains the code and documentation for our solution, which ultimately achieved high accuracy in classifying the food item from survey data.

## Dataset
The model is trained on a dataset of student survey responses, with each sample characterized by eight features. These features include a mix of **numerical ratings, categorical (multiple-choice) answers, and free-text responses**. We performed extensive preprocessing to convert all features into a machine-readable numeric form:
- **Numerical features** (e.g. ratings on a 1–5 scale) were used in their integer form (with minimal processing since they were already numeric).
- **Categorical features** (multiple-choice questions) were encoded into numeric vectors. Depending on the nature of the feature, we applied either **one-hot encoding** for nominal categories or **ordinal encoding** if an inherent order existed in the responses.
- **Textual features** (free-response questions) were transformed using a **Bag-of-Words** representation. We adopted a *filtered* Bag-of-Words approach that keeps only the most predictive words, striking a balance between interpretability and dimensionality. This filtering reduced the vocabulary from over 1000 possible words to about 50 key terms, removing infrequent or redundant words while preserving those with strong predictive power. For example, movie titles strongly associated with each food (like *Teenage Mutant Ninja Turtles* for pizza or *Jiro Dreams of Sushi* for sushi) were retained as features, whereas rarer titles were dropped as they were less generalizable.  
- We also handled missing or ambiguous survey answers. For instance, if a student answered "None" to a free-text question, we treated it as an *unknown* category and mapped it to a special one-hot vector so that the model would not misinterpret it as a missing value.

## Modeling Approach
We explored a range of machine learning models to find the best classifier for this task. In particular, we experimented with four families of models: **k-Nearest Neighbors (kNN)**, **Logistic Regression**, **Multilayer Perceptron (MLP)** neural networks, and **Decision Tree ensembles (Random Forests)**. 

Our modeling pipeline involved rigorous cross-validation and model comparison:
- We first split the data into training and testing sets (80% of the data was used for training, with the remaining 20% held out for final evaluation). The training portion was further used in a **5-fold cross-validation** framework for model selection and hyperparameter tuning.
- Each candidate model was trained and evaluated via cross-validation to assess its performance. We used **accuracy** as the primary metric for comparison, given the multi-class nature of the prediction and roughly balanced classes.
- We tuned key hyperparameters for each model type (e.g. the number of neighbors *k* for kNN, regularization strength for logistic regression, architecture and learning rate for MLPs, and tree parameters for Random Forests) using grid search. This process was exhaustive – for the Random Forest alone, we tested around 1800 hyperparameter combinations across the 5 folds (training a total of ~9000 models in cross-validation) – to ensure we identified the optimal configuration.
- Throughout this process, we noted the trade-offs of each model. Simpler models like kNN and logistic regression served as useful baselines but struggled to capture non-linear patterns in the data. MLPs offered more complexity but risked overfitting given the dataset size and required careful tuning. Ultimately, the **Random Forest** consistently outperformed the others in cross-validation, thanks to its ability to handle diverse feature types and capture complex relationships while controlling overfitting.

## Final Model
After comparing models, we selected a **Random Forest classifier** as our final model, as it delivered the highest validation accuracy and robust generalization performance. We optimized the Random Forest’s hyperparameters via grid search and 5-fold cross-validation. The best model’s configuration was as follows:

- **n_estimators** (number of trees): 300  
- **max_depth** (maximum tree depth): 10  
- **min_samples_split** (min samples to split a node): 15  
- **min_samples_leaf** (min samples per leaf node): 1  
- **max_features** (features considered per split): "log2"  

These hyperparameters were chosen based on the combination that yielded the highest cross-validation accuracy on the training data. The model was then re-trained on the entire training set with this optimized configuration, and subsequently evaluated on the held-out test set.

## Results
The finalized Random Forest model demonstrated **strong performance** on the classification task. It achieved an accuracy of approximately **89.7% on the test set**, which was in line with its cross-validation performance (about 90% on average). This high accuracy suggests that the model generalizes well to unseen data, likely due to the nature of the Random Forest ensemble. By averaging the predictions of many decision trees, the Random Forest reduces overfitting and variance, leading to better generalization on new examples. The model effectively leveraged the varied feature types in the survey (numerical, categorical, and textual) without requiring the data to be linearly separable, giving it an edge over simpler classifiers. In practice, the Random Forest outperformed the other approaches we tried, providing a good balance of accuracy and robustness. We are confident that our model captures meaningful patterns in the survey responses, as evidenced by its stable performance from cross-validation through to the final test evaluation.

## Usage
This repository includes a script `pred.py` which contains a convenient `predict_all` function for making predictions on new survey data. To generate food item predictions for a given input dataset:
1. Prepare the input data as a CSV file with the same format and columns as the original training data (eight feature columns, with identical question wording or headers).  
2. Call the `predict_all(csv_path)` function provided in `pred.py`, passing the path to your CSV file. The function will read and parse the data (using the same preprocessing steps as during training, such as converting categorical answers to one-hot vectors and applying the Bag-of-Words mapping for text). It ensures all features are aligned to the expected format and order of the model.  
3. The script then loads our pre-trained Random Forest classifier from the embedded `model.py` (a serialized model file) and produces a prediction for each survey response. The output is the predicted food category (Pizza, Shawarma, or Sushi) for every entry in the input file.

Under the hood, `model.py` contains the exported decision logic of the Random Forest (generated via model serialization) and is used by `pred.py` to compute predictions. **No external machine learning libraries are required** to run predictions – the model’s logic is self-contained in plain Python code (consisting of a series of nested if-else statements representing the trees). This design, mandated by the course challenge rules, allows the prediction script to run in a standard Python environment without needing scikit-learn or other ML frameworks.

## Technologies Used
- **Python 3.10** – All code was written in Python 3.10, taking advantage of its modern features and stability for scientific computing.  
- **NumPy & Pandas** – Used for data manipulation, analysis, and I/O. **Pandas** was especially useful for reading CSV data and preprocessing features (e.g., applying one-hot encodings, handling missing values), while **NumPy** underpinned numerical computations and array operations throughout the project.  
- **scikit-learn** – Employed for model training and evaluation. We utilized scikit-learn implementations of kNN, logistic regression, MLP, and Random Forest during experimentation. The final Random Forest model was trained using scikit-learn’s `RandomForestClassifier`, which provided robust out-of-the-box functionality for our needs.  
- **m2cgen (Model-to-Code Generator)** – A library we used for model **serialization**. After training the final Random Forest with scikit-learn, we used m2cgen (referred to as the "m2c" library) to convert the learned model into native Python code. This produced the `model.py` file containing the hard-coded logic of the classifier, enabling predictions without relying on external machine learning libraries at runtime.

## Acknowledgements
This project was developed as a team effort for the CSC311 course at University of Toronto. We would like to thank the **CSC311 instructors and teaching staff** for organizing the machine learning challenge and for their guidance throughout the term. The experience gained from this project was invaluable. *This repository showcases our final work for the course, reflecting the contributions of our team members (kept anonymous here) and the knowledge we obtained from CSC311.*  
