<h1 align="center">
  CSC311 – ML Classifier
  <br/>
  <sub>Predicting 🍕 Pizza • 🌯 Shawarma • 🍣 Sushi from student survey data</sub>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> Final project for **CSC311 — Machine Learning (Winter 2025, University of Toronto)**  
> Highest-scoring Random-Forest solution (≈ 89.7 % accuracy) in the course challenge.

<div align="center">
  <!-- Swap the GIF/PNG below after recording a short demo -->
  <img src="docs/demo.gif" width="650" alt="demo gif"/>
</div>

---

## 📑 Table of Contents
1. [Dataset](#dataset)
2. [Pre-processing](#pre-processing)
3. [Modelling & Validation](#modelling--validation)
4. [Final Model & Results](#final-model--results)
5. [Quick Start](#quick-start)
6. [Tech & Libraries](#tech--libraries)
7. [Acknowledgements](#acknowledgements)

---

## Dataset<a name="dataset"></a>

* 1 600 student survey rows  
* **8 questions** per row → ratings, categorical choices, free-text answers  
* Goal: classify the target food (`Pizza`, `Shawarma`, `Sushi`)

## Pre-processing<a name="pre-processing"></a>

| Feature type | Technique |
|--------------|-----------|
| Numerical ratings | Passed through as integers |
| Categorical (MCQ) | One-hot or ordinal encoding |
| Free-text answers | Filtered **Bag-of-Words** (≈ 50 predictive tokens) |
| Missing / “None” | Routed to explicit **unknown** category |

---

## Modelling & Validation<a name="modelling--validation"></a>

* Evaluated **kNN**, **LogReg**, **MLP**, **Random Forest**  
* **5-fold CV** on 80 / 20 train-test split  
* Exhaustive grid search → ~9 000 model/param combos  
* **Random Forest** best overall: captures nonlinear signals & mixed datatypes

---

## Final Model & Results<a name="final-model--results"></a>

| Hyperparam         | Value |
|--------------------|-------|
| `n_estimators`     | 300   |
| `max_depth`        | 10    |
| `min_samples_split`| 15    |
| `min_samples_leaf` | 1     |
| `max_features`     | "log2"|

* **Test accuracy:** **89.7 %** (held-out)  
* CV accuracy: ~90 % ± 1 % → robust generalisation

The model was exported to pure Python via **m2cgen**, yielding `model.py` (no sklearn dependency at inference time).

---

## ⚡ Quick Start<a name="quick-start"></a>

```bash
# clone & install minimal deps
git clone https://github.com/Myk0laUA/CSC311-MLProj.git
cd CSC311-MLProj
pip install -r requirements.txt   # only pandas & numpy
# predict on your own survey CSV
python pred.py --input my_survey.csv --output predictions.csv
```
pred.py loads model.py, applies the exact training-time preprocessing pipeline, and writes the predicted food for each row.
## Tech & Libraries<a name="tech--libraries"></a>

* Python 3.10 • NumPy • pandas
* scikit-learn 1.4 – training / CV
* m2cgen – model-to-code serialization
