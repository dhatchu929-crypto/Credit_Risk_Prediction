# Credit Risk Prediction Using Machine Learning

## Project Overview
This project predicts loan default risk using Machine Learning techniques. It compares Logistic Regression and Random Forest models to identify applicants who may have difficulty repaying loans.

## Objectives
- Perform Exploratory Data Analysis (EDA).
- Handle missing values and outliers.
- Encode categorical features.
- Train Logistic Regression and Random Forest models.
- Compare model performance.
- Save the best model using Joblib.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Dataset
- File: `sample_credit_risk_india.csv`
- Records: 600
- Target: `loan_status`
- Classes: Default and Non-Default

**Note:** This is a synthetic practice dataset, not real banking customer data.

## Machine Learning Models
1. Logistic Regression
2. Random Forest

## Model Performance

| Model | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.6488 | 0.4359 | 0.4722 |
| Random Forest | 0.6675 | 0.4815 | 0.3611 |

Random Forest achieved the higher ROC-AUC and Precision. However, its Recall is lower, so further improvement is needed.

## Project Outputs
- `credit_risk_model.pkl` – Saved machine learning model
- `model_comparison.csv` – Model performance comparison
- `feature_importance.csv` – Feature importance results
- `feature_importance.png` – Feature importance chart
- `loan_status.png` – Loan status distribution
- `roc_curve.png` – ROC curve
- `technical_evaluation_memo.md` – Technical evaluation report

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Conclusion
The project successfully compares two machine learning models for credit risk prediction. Random Forest achieved a ROC-AUC score of approximately 0.6675 and was selected as the current best model.

Further validation using real-world data is required before practical deployment. Predictions should support human review and should not be used for automatic loan rejection.
