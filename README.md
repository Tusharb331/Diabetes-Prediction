# Diabetes Prediction Project

## Overview

This project uses machine learning to predict diabetes based on health metrics. It employs the Support Vector Machine (SVM) with a linear kernel for classification.

## Dataset

The dataset used is `diabetes.csv`, containing the health information of individuals. The target variable is `Outcome`, where:
- 0: Non-diabetic
- 1: Diabetic

## Features

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration in a 2-hour oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years

## Workflow

1. **Data Loading and Preprocessing**:
   - Load the dataset (`diabetes.csv`).
   - Separate features and target variable (`Outcome`).

2. **Feature Standardization**:
   - Standardize the features using `StandardScaler` to ensure each feature contributes equally to the model.

3. **Model Training**:
   - Split the dataset into training and testing sets (80%-20%).
   - Train an SVM model with a linear kernel on the training data.

4. **Model Evaluation**:
   - Evaluate the model on both training and testing data using the accuracy score.

5. **Prediction**:
   - Perform predictions on new input data to determine if the person is diabetic or not.

## Results

- **Accuracy on Training Data**: XX.XX%
- **Accuracy on Test Data**: XX.XX%

## Example Prediction

```python
# Example prediction for new input data
input_sample = np.array([5, 166, 72, 19, 175, 22.7, 0.6, 51]).reshape(1, -1)
input_df = pd.DataFrame(input_sample, columns=X.columns)  # Assuming X has feature names
input_standardized = scaler.transform(input_df)
prediction = clf.predict(input_standardized)

if prediction[0] == 0:
    print("Prediction: Person is not diabetic")
else:
    print("Prediction: Person is diabetic")
