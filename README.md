

# Customer Churn Prediction

---

## Project Overview

This project predicts which customers are likely to churn (stop using a service) using **machine learning**.
The repository includes **data preprocessing, feature engineering, class imbalance handling, model training (ML + deep learning), hyperparameter tuning, evaluation, and deployment via a lightweight app**.

Why this project is strong for interviews:

* Shows an **end-to-end ML workflow** (cleaning → encoding → scaling → resampling → model training → evaluation → deployment).
* Covers **both ML models and Neural Networks**.
* Handles **class imbalance** correctly (SMOTE).
* Demonstrates **hyperparameter tuning** and explains tradeoffs (accuracy vs recall vs F1).
* Shows awareness of **real-world pitfalls** (data leakage, imbalanced classes).

---

## Table of Contents

* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Setup](#setup)
* [Dataset](#dataset)
* [Data Preprocessing](#data-preprocessing)
* [Model Training & Hyperparameter Tuning](#model-training--hyperparameter-tuning)
* [Neural Network Training (TensorFlow)](#neural-network-training-tensorflow)
* [Evaluation](#evaluation)
* [Model Export & Inference](#model-export--inference)
* [How to Run](#how-to-run)
* [Improvements & Interview Notes](#improvements--interview-notes)
* [Contact](#contact)

---

## Repository Structure

```
customer-churn-prediction/
├─ dataset/                 # contains the customer churn dataset (CSV)
├─ notebooks/               # Jupyter notebooks for EDA & training
├─ scripts/                 # Python scripts for preprocessing & training
├─ app.py                   # Flask app for inference
├─ requirements.txt          # dependencies
└─ README.md                 # project documentation
```

---

## Requirements

Recommended Python version: **3.9+**

Minimal `requirements.txt`:

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
tensorflow
flask
jupyter
```

Install:

```bash
python -m venv venv
source venv/bin/activate   # linux / macOS
venv\Scripts\activate     # windows
pip install -r requirements.txt
```

---

## Dataset

The dataset is provided separately inside the `dataset/` folder.

**Target variable:**

* `Exited` → (1 = churned, 0 = stayed)

**Features include:**

* **Demographics**: Gender, Age, Geography
* **Account info**: Tenure, Balance, Salary, Products
* **Behavioral/contract**: Credit card ownership, Active membership

**Sample (first 5 rows):**

| CustomerId | Surname  | Geography | Gender | Age | Tenure | Balance    | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
| ---------- | -------- | --------- | ------ | --- | ------ | ---------- | ------------- | --------- | -------------- | --------------- | ------ |
| 15634602   | Hargrave | France    | Female | 42  | 2      | 0.00       | 1             | 1         | 1              | 58,101.00       | 1      |
| 15647311   | Hill     | Spain     | Female | 41  | 1      | 83,810.00  | 1             | 0         | 1              | 83,221.00       | 0      |
| 15619304   | Onio     | France    | Female | 42  | 8      | 0.00       | 3             | 1         | 0              | 113,932.00      | 1      |
| 15701354   | Boni     | France    | Female | 39  | 1      | 0.00       | 2             | 0         | 0              | 112,452.00      | 0      |
| 15737888   | Mitchell | Spain     | Male   | 43  | 2      | 125,510.00 | 1             | 1         | 1              | 79,034.00       | 0      |

**Notes:**

* Dataset is **imbalanced** (fewer churned customers).
* Encoders (`LabelEncoder`, `OneHotEncoder`) and scaler (`StandardScaler`) are **saved with pickle** for reuse.

---

## Data Preprocessing

Steps applied:

* **Missing values**: dropped or imputed.
* **Encoding**: categorical → numerical (Gender = LabelEncoder, Geography = OneHotEncoder).
* **Scaling**: numerical features scaled with `StandardScaler`.
* **SMOTE**: oversampling applied **only on training data** to fix imbalance.

Encoders and scalers are saved for inference:

```python
with open('scaler.pkl','wb') as f:
    pickle.dump(scaler, f)
```

---

## Model Training & Hyperparameter Tuning

Models trained with **GridSearchCV**:

* Logistic Regression
* Random Forest
* SVM
* Decision Tree

**GridSearchCV setup**:

* `cv=5` → 5-fold cross-validation
* `scoring='f1'` → focus on balance of precision & recall
* `n_jobs=-1` → parallel execution across all CPU cores

Example:

```python
clf = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
clf.fit(X_train_res, y_train_res)
print(clf.best_params_)
```

---

## Neural Network Training (TensorFlow)

* Built with `Sequential` API (Dense layers).
* **Callbacks**:

  * `EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)`
  * `TensorBoard` logging

Example:

```python
history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[tensorboard_cb, early_stopping_cb]
)
```

---

## Evaluation

Metrics:

* **Classification Report** (precision, recall, F1)
* **F1 Score** → balances precision & recall
* **Recall** → important when missing churn is costly
* **ROC-AUC** → threshold-independent performance

Example:

```python
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs >= 0.5).astype(int)

print(classification_report(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('AUC-ROC:', roc_auc_score(y_test, y_pred_probs))
```

---

## Model Export & Inference

Models and preprocessing objects are saved with `pickle`/`joblib`.

A **Flask app (`app.py`)** is included for inference:

```bash
python app.py
# or
flask run
```

---

## How to Run

1. Clone repo:

```bash
git clone https://github.com/Sahilsharma500/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Setup environment & install dependencies:

```bash
pip install -r requirements.txt
```

3. Check dataset in `dataset/` folder.

4. Train models via notebooks or scripts in `scripts/` or `notebooks/`.

5. Run app:

```bash
python app.py
```

---

## Contact

Author: **Sahil Sharma**
GitHub: [Sahilsharma500](https://github.com/Sahilsharma500)

---

