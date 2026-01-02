# **Loan Approval Prediction – Machine Learning Project**

## **1. Business Problem Summary**

Financial institutions must decide whether to approve or reject loan applications while minimizing default risk.
Manual evaluation is time-consuming and subjective.

**Objective:**
Build a machine learning model that predicts whether a loan application will be **approved (1)** or **rejected (0)** based on applicant financial and demographic information.

The goal is to:

* Accurately identify creditworthy applicants
* Reduce financial risk
* Maintain fairness and interpretability

## **2. Dataset Overview**

The dataset contains historical loan application records with both **numeric** and **categorical** features.

### **Target Variable**

* `LoanApproved`

  * `0` → Loan Rejected
  * `1` → Loan Approved

### **Class Distribution**

* Class 0: **15,220 samples**
* Class 1: **4,780 samples**

➡ The dataset is **imbalanced**, making metrics like ROC-AUC and PR-AUC essential.

### **Key Features**

* Numeric:

  * DebtToIncomeRatio
  * TotalDebtToIncomeRatio
  * AnnualIncome
  * MonthlyIncome
  * LoanAmount
* Categorical:

  * EmploymentStatus
  * EducationLevel
  * MaritalStatus
  * HomeOwnershipStatus
  * LoanPurpose

## **3. Approach & Methodology**

### **Data Preprocessing**

* Removed irrelevant columns (e.g., `ApplicationDate`)
* Label-encoded ordered categorical features
* Applied **StandardScaler** to numeric features (required for KNN)

### **Train/Test Split**

* 70% Training
* 30% Testing
* Stratified to preserve class distribution

### **Models Trained**

* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest (Ensemble model)

### **Validation Strategy**

* 5-Fold **Stratified Cross-Validation**
* Evaluation on unseen test data

### **Evaluation Metrics**

Mandatory metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix


## **4. Key Insights**

### **Model Behavior**

* Income-related features **increase approval probability**
* High total debt-to-income ratio **strongly reduces approval chances**
* The model captures **non-linear relationships** between income and debt

### **Interpretability Findings**

* Partial Dependence Plots show:

  * Higher income → higher approval likelihood
  * Higher total debt burden → sharp decline in approval
* Feature importance confirms:

  * Debt and income metrics are dominant decision factors

### **Business Alignment**

The model’s decisions align with real-world lending logic:

* Strong repayment capacity → approval
* Excessive financial burden → rejection

## **5. Final Model Performance**

| Model             | CV ROC-AUC | Test Accuracy | Precision | Recall    | F1        | ROC-AUC   | PR-AUC    |
| ----------------- | ---------- | ------------- | --------- | --------- | --------- | --------- | --------- |
| KNN               | 0.966      | 0.933         | 0.936     | 0.783     | 0.853     | 0.971     | 0.945     |
| Decision Tree     | 0.980      | 0.983         | 0.965     | 0.963     | 0.964     | 0.976     | 0.969     |
| **Random Forest** | **0.999**  | **0.990**     | **0.983** | **0.977** | **0.980** | **0.999** | **0.998** |

### ✅ **Final Model Selected: Random Forest**

**Why?**

* Best performance across all metrics
* Excellent minority-class detection
* Robust to overfitting
* High interpretability via feature importance and PDPs

## **6. How to Run the Project**

### **Requirements**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **Steps**

1. Load the dataset
2. Run preprocessing script
3. Train models and evaluate performance
4. Generate metrics, confusion matrices, and plots
5. Interpret results using feature importance and PDPs


## **Conclusion**

This project successfully demonstrates how machine learning can automate loan approval decisions with high accuracy and interpretability.
The Random Forest model balances predictive power with business transparency, making it suitable for real-world deployment.


