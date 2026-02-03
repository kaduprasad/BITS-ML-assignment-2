# Heart Disease Prediction - ML Assignment 2

## Problem Statement

 This project builds and compares multiple machine learning classification models to predict heart disease based on clinical and demographic features.

## Dataset Description

**Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

**Source:** Kaggle

### Features (11 input features):

| Feature | Description | Type |
|---------|-------------|------|
| Age | Age of the patient | Numerical |
| Sex | Sex of the patient (M/F) | Categorical |
| ChestPainType | Type of chest pain (TA, ATA, NAP, ASY) | Categorical |
| RestingBP | Resting blood pressure (mm Hg) | Numerical |
| Cholesterol | Serum cholesterol (mm/dl) | Numerical |
| FastingBS | Fasting blood sugar > 120 mg/dl (1=true, 0=false) | Binary |
| RestingECG | Resting ECG results (Normal, ST, LVH) | Categorical |
| MaxHR | Maximum heart rate achieved | Numerical |
| ExerciseAngina | Exercise-induced angina (Y/N) | Categorical |
| Oldpeak | ST depression induced by exercise | Numerical |
| ST_Slope | Slope of peak exercise ST segment (Up, Flat, Down) | Categorical |

**Target Variable:** HeartDisease (1 = heart disease, 0 = normal)

**Dataset Statistics:**
- Total Instances: 918
- Total Features: 11
- Target Classes: 2 (Binary Classification)

## Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8587 | 0.9238 | 0.8594 | 0.8587 | 0.8586 | 0.7165 |
| Decision Tree | 0.7826 | 0.7821 | 0.7827 | 0.7826 | 0.7826 | 0.5647 |
| kNN | 0.8478 | 0.9105 | 0.8479 | 0.8478 | 0.8478 | 0.6951 |
| Naive Bayes | 0.8424 | 0.9055 | 0.8432 | 0.8424 | 0.8423 | 0.6841 |
| Random Forest (Ensemble) | 0.8804 | 0.9379 | 0.8808 | 0.8804 | 0.8804 | 0.7604 |
| XGBoost (Ensemble) | 0.8696 | 0.9288 | 0.8699 | 0.8696 | 0.8695 | 0.7388 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved 85.87% accuracy with AUC of 0.9238. The linear decision boundary works well for this dataset and provides interpretable results. |
| Decision Tree | Accuracy of 78.26% with AUC of 0.7821. Shows signs of overfitting as it has the lowest performance among all models. |
| kNN | Achieved 84.78% accuracy with AUC of 0.9105. Performance is sensitive to the choice of k value and feature scaling. |
| Naive Bayes | Accuracy of 84.24% with AUC of 0.9055. Despite the independence assumption, it performs reasonably well on this dataset. |
| Random Forest (Ensemble) | Best performer with 88.04% accuracy and highest AUC of 0.9379. The ensemble approach reduces overfitting effectively. |
| XGBoost (Ensemble) | Second best with 86.96% accuracy and AUC of 0.9288. Gradient boosting captures feature interactions well. |
