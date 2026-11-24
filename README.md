# Credit Risk Prediction Using Machine Learning

## Project Overview
This project predicts loan default risk using Machine Learning algorithms.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Dataset
- File: sample_credit_risk_india.csv
- Records: 600
- Dataset used: Synthetic practice dataset

## Machine Learning Models
1. Logistic Regression
2. Random Forest

## Model Performance

| Model | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.649 | 0.436 | 0.477 |
| Random Forest | 0.667 | 0.481 | 0.361 |

## Project Outputs
- Loan status distribution graph
- Feature importance graph
- ROC curve
- Model comparison CSV
- Saved ML model

## How to Run

1. Install dependencies:
   `pip install -r requirements.txt`

2. Run the project:
   `python main.py`

## Conclusion
Random Forest achieved the higher ROC-AUC score and was selected as the final model.

Note: This is an educational project using synthetic data. It is not intended for real loan approval or rejection decisions.