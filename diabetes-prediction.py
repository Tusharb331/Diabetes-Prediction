import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('diabetes.csv')

# Separate features and target variable
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

# Standardize the features with feature names
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# Train the SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model on training data
train_predictions = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Accuracy on training data: {:.2f}%".format(train_accuracy * 100))

# Evaluate the model on test data
test_predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy on test data: {:.2f}%".format(test_accuracy * 100))

# Example prediction for new input data with feature names
input_sample = np.array([5, 166, 72, 19, 175, 22.7, 0.6, 51]).reshape(1, -1)
input_df = pd.DataFrame(input_sample, columns=X.columns)  # Assuming X has feature names
input_standardized = scaler.transform(input_df)
prediction = clf.predict(input_standardized)

if prediction[0] == 0:
    print("Prediction: Person is not diabetic")
else:
    print("Prediction: Person is diabetic")
