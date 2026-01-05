# 🚲 Bike Demand Prediction using Ridge Regression (Mini-Batch Gradient Descent)

This project implements **Ridge Regression from scratch** to predict hourly bike rental demand using weather and temporal features.  
The model is optimized using **Mini-Batch Gradient Descent (MBGD)** and evaluated with **5-fold cross-validation**.

---

## 📌 Project Overview

- **Dataset**: Bike Sharing Dataset (Hourly + Daily)
- **Problem Type**: Regression
- **Target Variable**: Hourly bike count (`cnt`)
- **Model**: Ridge Regression (L2 Regularization)
- **Optimization**: Mini-Batch Gradient Descent

---

## 🧠 Concepts Implemented

- Dataset merging and feature selection  
- Feature standardization (z-score normalization)  
- Manual bias (intercept) handling  
- Ridge (L2) regularization  
- Mini-Batch Gradient Descent  
- 5-Fold Cross-Validation  
- Mean Squared Error (MSE) evaluation  

---

## 🛠️ Tech Stack

- **Python**
- **NumPy** – numerical computation
- **Pandas** – data loading and preprocessing
- **Scikit-learn** – `StandardScaler`, `KFold` (no ML models used)

---

## 📂 Project Structure
  ├── ridge_regression_mbgd.py
  
  ├── README.md

---

## ⚙️ Data Preparation

- Merged hourly and daily bike datasets on date (`dteday`)
- Selected weather and temporal features:
  - Temperature, Humidity, Windspeed
  - Hour of day, Weekday, Working day
- Standardized all input features
- Added bias (intercept) term manually

---

## 📐 Model Description

### Ridge Regression Objective

\[
\min_{\theta} \; \frac{1}{m}\|X\theta - y\|^2 + \lambda \|\theta\|^2
\]

- L2 regularization helps reduce overfitting  
- Bias term is excluded from regularization  

---

## 🔁 Training & Evaluation

- Model trained using **Mini-Batch Gradient Descent**
- **5-fold cross-validation** used for evaluation
- Model reinitialized and retrained for each fold
- Performance measured using **Mean Squared Error (MSE)**

---

## 📊 Results & Observations

- Ridge regularization improves generalization
- Feature scaling significantly improves convergence
- Mini-batch updates balance training stability and speed
- Cross-validation confirms consistent performance across folds

---

## 🚀 How to Run

```bash
pip install numpy pandas scikit-learn
python ridge_regression_mbgd.py



