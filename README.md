# AI-ML-ELEVATE-LABS-04


# Breast Cancer Classification using Logistic Regression

This project demonstrates a binary classification task to predict whether a tumor is **malignant** or **benign** using the Breast Cancer dataset. The model uses **Logistic Regression**, explores threshold optimization, and evaluates performance using metrics like **accuracy, confusion matrix, and ROC-AUC curve**.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Modeling](#modeling)  
5. [Evaluation Metrics](#evaluation-metrics)  
6. [Threshold Tuning](#threshold-tuning)  
7. [ROC Curve & Sigmoid Function](#roc-curve--sigmoid-function)  
8. [Results](#results)  
9. [Visualizations](#visualizations)  
10. [Conclusion](#conclusion)  

---

## Project Overview

The goal of this project is to build a logistic regression model to classify tumors into:

- **0 → Benign**  
- **1 → Malignant**

Logistic regression is chosen for its simplicity and interpretability in binary classification tasks. The project also explores **threshold tuning** to improve classification performance.

---

## Dataset

- The dataset used is `Breast Cancer.csv`.  
- It contains features extracted from tumor images, including:
  - Radius, Texture, Perimeter, Area
  - Smoothness, Compactness, Concavity, Symmetry, Fractal Dimension
- Columns `id` and `Unnamed: 32` are dropped as they are not useful for modeling.  

### Dataset Preview

```python
df.head()
Null Values Check
df.isnull().sum()
Data Preprocessing
Label Encoding:
The diagnosis column is categorical (M or B) and is encoded to 1 (malignant) and 0 (benign).

Feature Selection:
Features id and diagnosis are separated into:

X → Input features

y → Target label

Train-Test Split:

80% data for training

20% data for testing

random_state=42 ensures reproducibility

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Modeling
Algorithm: Logistic Regression

Library: sklearn.linear_model.LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
Evaluation Metrics
Accuracy: Measures overall correctness of predictions.

Confusion Matrix: Shows true positives, true negatives, false positives, and false negatives.

Classification Report: Provides precision, recall, and F1-score.

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
Threshold Tuning
By default, logistic regression uses a 0.5 threshold to classify probabilities into classes.

Optimizing threshold can improve sensitivity and specificity.

Youden’s J statistic is used to find the best threshold:

j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]
y_pred_best = (y_probs >= best_threshold).astype(int)
ROC Curve & Sigmoid Function
Sigmoid Function: Maps any value to a probability between 0 and 1.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
ROC Curve: Plots True Positive Rate (TPR) vs False Positive Rate (FPR).

AUC (Area Under Curve): Measures overall ability of the model to discriminate between classes.

roc_auc = roc_auc_score(y_test, y_probs)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
Results
Accuracy (default threshold 0.5): 0.9474

Best Threshold Accuracy: 0.9532

ROC-AUC Score: 0.98

Confusion matrix shows high true positive and true negative rates, indicating reliable classification.

Visualizations
Confusion Matrix: Visualize classification errors

ROC Curve: Evaluate model discrimination

Sigmoid Function: Understand probability mapping

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

plt.plot(fpr, tpr)
plt.show()

plt.plot(z, sigmoid(z))
plt.show()
Conclusion
Logistic Regression effectively classifies breast cancer tumors as malignant or benign.

Threshold tuning can improve model performance beyond the default 0.5 cutoff.

ROC-AUC score confirms the model's high discriminative power.

This project can be extended using other classifiers like Random Forests or XGBoost, or by performing feature selection to improve performance further.

References
Scikit-learn Logistic Regression Documentation

Breast Cancer Wisconsin Dataset
