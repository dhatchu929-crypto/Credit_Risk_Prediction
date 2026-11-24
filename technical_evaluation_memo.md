# Technical Evaluation Memo
## Credit Risk Prediction Using Machine Learning

### 1. Project Overview
This project predicts loan default risk using Machine Learning algorithms. The objective is to identify applicants who may have difficulty repaying their loans.

### 2. Dataset
- Dataset: sample_credit_risk_india.csv
- Total records: 600
- Target column: loan_status
- Classes: Default and Non-Default

**Note:** This is a synthetic practice dataset, not real bank customer data.

### 3. Data Preprocessing
- Handled missing values.
- Checked duplicate records.
- Applied outlier handling.
- Encoded categorical features.
- Split data into training and testing sets.

### 4. Models Evaluated
Two Machine Learning models were compared:

1. Logistic Regression
2. Random Forest

### 5. Model Performance

| Model | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.6488 | 0.4359 | 0.4722 |
| Random Forest | 0.6675 | 0.4815 | 0.3611 |

Random Forest achieved the higher ROC-AUC and Precision scores. However, its Recall was lower than Logistic Regression.

### 6. Model Selection
Random Forest was selected as the current best model based on ROC-AUC and Precision. The model was saved using Joblib.

### 7. Feature Importance
Feature importance results are available in `outputs/feature_importance.csv`. The most influential features should be reviewed before making business decisions.

### 8. Risk Threshold Recommendation
A prediction probability threshold of 0.5 can be used as an initial experimental threshold. Applicants identified as high risk should be sent for manual review rather than automatically rejected.

### 9. Limitations
- The dataset is synthetic.
- Model performance may differ on real-world banking data.
- Further validation is required before practical use.

### 10. Conclusion
The project successfully compares Logistic Regression and Random Forest for credit risk prediction. Random Forest achieved a ROC-AUC of approximately 0.6675. Further improvements and validation are required before real-world deployment.