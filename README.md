# Credit_Card_Fraud_Detection_ml
<div align="center">

<img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="100">

# 💳 Credit Card Fraud Detection Using Machine Learning

### Detecting fraudulent transactions from highly imbalanced financial data using Logistic Regression and Python

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Model-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)](https://github.com/Vanshika26Ramjas/credit_card_fraud_detection_ml)

</div>
Dataset_pdf :- [Credit Card Fraud Detection_ ML _ Python.pdf](https://github.com/user-attachments/files/29143214/Credit.Card.Fraud.Detection_.ML._.Python.pdf)

---

# 📌 Table of Contents

- Project Overview
- Business Problem
- Dataset Overview
- Dashboard & Report
- Machine Learning Pipeline
- Handling Class Imbalance
- Model Performance
- Confusion Matrix Analysis
- Key Insights
- Tech Stack
- Project Structure
- Getting Started
- Dataset Link
- Repository Link
- Author

---

# 🎯 Project Overview

Financial institutions process millions of transactions every day, making fraud detection one of the most critical applications of Machine Learning.

This project builds a fraud detection system using **Logistic Regression** to classify transactions as:

- ✅ Legitimate
- 🚨 Fraudulent

The dataset contains over **284,000 real-world credit card transactions** with severe class imbalance, where fraudulent transactions represent only **0.17%** of all records.

The objective is to identify fraudulent activity while minimizing false alarms and maximizing detection accuracy.

---

# 💼 Business Problem

Credit card fraud causes billions of dollars in losses annually.

Traditional rule-based systems often struggle because:

- Fraud patterns continuously evolve
- Fraudulent transactions are extremely rare
- High false positives negatively impact customer experience

This project demonstrates how Machine Learning can improve fraud detection by learning hidden patterns within transaction behavior.

---

# 📊 Dataset Overview

<div align="center">

| Metric | Value |
|----------|----------|
| Total Transactions | 284,807 |
| Fraud Cases | 492 |
| Legitimate Cases | 284,315 |
| Features | 30 |
| Target Variable | Class |
| Fraud Percentage | 0.17% |
| Missing Values | 0 |

</div>

---

## 📋 Dataset Features

| Category | Features |
|-----------|-----------|
| Time Information | Time |
| Transaction Amount | Amount |
| PCA Features | V1 - V28 |
| Target | Class |

### Target Variable

| Class | Meaning |
|---------|---------|
| 0 | Legitimate Transaction |
| 1 | Fraudulent Transaction |

---

# 📈 Class Distribution

Before balancing:

```text
Legitimate Transactions : 284,315 (99.83%)

Fraudulent Transactions : 492 (0.17%)
```

This extreme imbalance makes traditional accuracy metrics misleading.

A model predicting every transaction as legitimate would still achieve:

```text
99.83% Accuracy
```

while detecting zero fraud.

---

# ⚖️ Handling Class Imbalance

To create a fair learning environment, Random Under-Sampling was applied.

### Before Sampling

```text
Legitimate : 284,315

Fraud : 492
```

### After Sampling

```text
Legitimate : 492

Fraud : 492
```

### Final Balanced Dataset

```text
984 Transactions

50% Legitimate

50% Fraudulent
```

This allows the model to learn fraud patterns effectively instead of being dominated by legitimate transactions.

---

# 🔄 Machine Learning Pipeline

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Under-Sampling
     │
     ▼
Train-Test Split
     │
     ▼
Logistic Regression
     │
     ▼
Model Evaluation
```

---

# 🤖 Model Used

## Logistic Regression

Logistic Regression was selected because:

- Fast and interpretable
- Effective for binary classification
- Strong baseline model
- Handles high-dimensional data efficiently

Training Configuration:

```python
train_test_split(test_size=0.2)

LogisticRegression()
```

---

# 📊 Model Performance

## Training Accuracy

```text
94.41%
```

## Testing Accuracy

```text
95.43%
```

### Observation

```text
Test Accuracy > Training Accuracy
```

This indicates:

✅ Good Generalization

✅ No Significant Overfitting

✅ Stable Performance on Unseen Data

---

# 🎯 Confusion Matrix

| | Predicted Legit | Predicted Fraud |
|------------|------------|------------|
| Actual Legit | 189 | 10 |
| Actual Fraud | 9 | 188 |

---

# 📈 Evaluation Metrics

<div align="center">

| Metric | Score |
|----------|----------|
| Accuracy | 95.43% |
| Precision | 94.9% |
| Recall | 95.4% |
| F1 Score | 95.1% |

</div>

---

# 🔍 Feature Importance Insights

Top features contributing to fraud detection:

| Rank | Feature |
|--------|--------|
| 1 | V3 |
| 2 | V7 |
| 3 | V1 |
| 4 | V4 |
| 5 | V2 |
| 6 | V5 |
| 7 | V9 |
| 8 | V11 |
| 9 | V14 |
| 10 | V16 |

### Key Observation

```text
V3 and V7 show the strongest deviation
between fraudulent and legitimate transactions.
```

These variables provide the strongest fraud signals within the dataset.

---

# 💡 Key Insights

### 1. Severe Imbalance Exists

Only 0.17% transactions are fraudulent.

---

### 2. Accuracy Alone Is Misleading

A naive model achieves 99.83% accuracy while detecting no fraud.

---

### 3. Under-Sampling Improves Learning

Balancing the dataset enables meaningful fraud classification.

---

### 4. Strong Generalization

Test accuracy exceeds training accuracy, indicating stable performance.

---

### 5. Fraud Transactions Show Distinct Patterns

PCA-transformed features reveal significant behavioral differences.

---

# 🛠️ Tech Stack

| Layer | Technology |
|----------|----------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |
| Version Control | Git, GitHub |

---

# 📁 Project Structure

```text
credit_card_fraud_detection_ml/
│
├── creditcard.csv
├── Credit_Card_Fraud_Detection.ipynb
├── Credit Card Fraud Detection Report.pdf
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/Vanshika26Ramjas/credit_card_fraud_detection_ml.git
```

### Navigate to Project

```bash
cd credit_card_fraud_detection_ml
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Run Notebook

```bash
jupyter notebook
```

---

# 📂 Dataset Link

Dataset Source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

# 📑 Project Report

Upload your PDF inside:

```text
/assets/Credit_Card_Fraud_Detection_Report.pdf
```

Then add:

```markdown
📄 [View Full Project Report](assets/Credit_Card_Fraud_Detection_Report.pdf)
```

---

# 🔗 Repository Link

GitHub Repository:

https://github.com/Vanshika26Ramjas/credit_card_fraud_detection_ml

---

# 👩‍💻 Author

### Vanshika Kumar

B.Sc. Statistics (Hons) | Ramjas College, University of Delhi

Data Analyst | SQL | Python | Power BI | Machine Learning

🔗 GitHub:
https://github.com/Vanshika26Ramjas

🔗 LinkedIn:
(Add Your LinkedIn URL)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star.

Built with Python, Statistics, and Machine Learning.

</div>
