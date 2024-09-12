# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:19:08 2024

@author: Sai.Vigneshwar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


# Define the file path
file_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Microsoft -Classifying Cybersecurity/train data/train data.xlsx'

# Load the dataset
try:
    train_data_df = pd.read_excel(file_path)
except UnicodeDecodeError:
    print("UnicodeDecodeError encountered. Trying with a different encoding.")
    train_data_df = pd.read_excel(file_path, encoding='ISO-8859-1')  # This is just for illustration; encoding is not an argument for `pd.read_excel`


''' Step 1: Data Exploration and Understanding '''

# Initial inspection
print(train_data_df.info()) # Overview of the dataset
print(train_data_df.describe()) #Summary statistics for numerical columns
print(train_data_df.head()) # First few rows of the dataset


# Check for missing values
print("\nMissing Values in Each Column:")
print(train_data_df.isnull().sum())


# Exploratory Data Analysis (EDA)
# Assuming 'IncidentGrade' is the name of your target column
target_column = 'IncidentGrade'  # Replace with the actual column name


# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='IncidentGrade', data=train_data_df)
plt.title('Distribution of Target Variable')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Histograms for numerical features
train_data_df.hist(bins=30, figsize=(20, 15), edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Boxplots to detect outliers in numerical features
plt.figure(figsize=(20, 10))
sns.boxplot(data=train_data_df.select_dtypes(include=[np.number]), orient='h', palette='Set2')
plt.title('Boxplots of Numerical Features')
plt.show()

# Count plot for each categorical variable (if applicable)
categorical_columns = train_data_df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, data=train_data_df)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
    


''' Step 2: Data Preprocessing '''

# Checking for missing values
missing_values = train_data_df.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values[missing_values > 0])


# Impute numerical columns with median
numerical_cols = train_data_df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    train_data_df[col].fillna(train_data_df[col].median(), inplace=True)

# Impute categorical columns with the most frequent value (mode)
categorical_cols = train_data_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train_data_df[col].fillna(train_data_df[col].mode()[0], inplace=True)

# Check if all missing values are handled
print("Missing Values After Imputation:")
print(train_data_df.isnull().sum().sum())  # Should return 0


# Feature Engineering
# Assuming the 'Timestamp' column is in ISO 8601 format as shown in the image
if 'Timestamp' in train_data_df.columns:
    # Convert the 'Timestamp' column to datetime format
    train_data_df['Timestamp'] = pd.to_datetime(train_data_df['Timestamp'])

    # Extract useful time-based features
    train_data_df['hour'] = train_data_df['Timestamp'].dt.hour
    train_data_df['day_of_week'] = train_data_df['Timestamp'].dt.dayofweek
    train_data_df['month'] = train_data_df['Timestamp'].dt.month

    # Print the first few rows to check the new features
    print(train_data_df[['Timestamp', 'hour', 'day_of_week', 'month']].head())
    

scaler = MinMaxScaler()
train_data_df[numerical_cols] = scaler.fit_transform(train_data_df[numerical_cols])

# Display the modified dataframe
print(train_data_df.head())


# Encoding Categorical Variables

# Label Encoding for binary categorical variables
label_encoder = LabelEncoder()

# Identify binary categorical columns
binary_categorical_cols = [col for col in categorical_cols if train_data_df[col].nunique() <= 3]

for col in binary_categorical_cols:
    train_data_df[col] = label_encoder.fit_transform(train_data_df[col])

# One-Hot Encoding for categorical variables with more than two categories
train_data_df = pd.get_dummies(train_data_df, columns=[col for col in categorical_cols if col not in binary_categorical_cols])

# Display the processed dataframe
print("Processed DataFrame:")
print(train_data_df.head())


''' Step 3 : Data Splitting '''

# Train-Validation Split

# 'IncidentGrade' is the name of your target column (replace it with your actual target column name)
target_column = 'IncidentGrade'  # Replace with the actual column name

# Define features (X) and target (y)
X = train_data_df.drop(columns=['IncidentGrade'])
y = train_data_df['IncidentGrade']

# Perform the train-validation split (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)


# Stratification (Handling Imbalanced Data)

# Perform the train-validation split with stratification
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Verify the stratification
print("Original class distribution:")
print(y.value_counts(normalize=True))

print("\nTraining set class distribution:")
print(y_train.value_counts(normalize=True))

print("\nValidation set class distribution:")
print(y_val.value_counts(normalize=True))


''' Step 4: Model Selection and Training '''

# Baseline Model: Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(random_state=42)

# Train the model on the training data
logistic_regression_model.fit(X_train, y_train)

y_val_pred = logistic_regression_model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_val_pred)


# Advanced Models: Initialize RandomForestClassifier model
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_rf_model = grid_search.best_estimator_

# Predict on validation set
y_pred_rf = best_rf_model.predict(X_val)

# Evaluate the model
print("Random Forest Model Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))

# Cross-validate the best model from Grid Search
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", np.mean(cv_scores))


''' Step 5 & 6 : Model Evaluation and Tuning  &  Model Interpretation'''

# Instantiate the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)

# Fit the model on training data (assuming X_train and y_train are defined)
random_forest_model.fit(X_train, y_train)


y_val_pred = random_forest_model.predict(X_val)

# Generate a classification report
report = classification_report(y_val, y_val_pred, target_names=['TP', 'BP', 'FP'])

# Calculate the macro-F1 score
macro_f1 = f1_score(y_val, y_val_pred, average='macro')

# Calculate precision and recall for each class
precision = precision_score(y_val, y_val_pred, average='macro')
recall = recall_score(y_val, y_val_pred, average='macro')

print(report)
print(f"Macro-F1 Score: {macro_f1:.2f}")
print(f"Macro Precision: {precision:.2f}")
print(f"Macro Recall: {recall:.2f}")


# Calculate precision, recall, and F1-score for each class
precision = precision_score(y_val, y_val_pred, average=None, labels=['TP', 'BP', 'FP'])
recall = recall_score(y_val, y_val_pred, average=None, labels=['TP', 'BP', 'FP'])
f1 = f1_score(y_val, y_val_pred, average=None, labels=['TP', 'BP', 'FP'])

# Calculate macro F1-score (average of F1 scores for all classes)
macro_f1 = f1_score(y_val, y_val_pred, average='macro')

print("Precision for each class:", precision)
print("Recall for each class:", recall)
print("F1-Score for each class:", f1)
print("Macro F1-Score:", macro_f1)

# Generate a detailed classification report
print(classification_report(y_val, y_val_pred, target_names=['TP', 'BP', 'FP']))
