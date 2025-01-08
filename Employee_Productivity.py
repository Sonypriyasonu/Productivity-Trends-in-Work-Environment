## Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Loading data
df = pd.read_csv('Employee_Productivity_Data.csv')
print(df.head())

df.info()

df.describe()

# Shape of data
df.shape

# Check missing value
df.isnull().sum()

# categorical variabes
df_cat = df.loc[:, df.dtypes == 'object']
df_cat = df_cat.drop('Hire_Date', axis='columns')
for col in df_cat:
    print(df[col].unique(), df[col].unique().__len__())

## Handling missing columns
# Handle missing values in numeric columns by filling with mean
num_col = ['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 
           'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 
           'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score']  # Adjust based on your dataset
for col in num_col:
    df[col] = df[col].fillna(df[col].mean())

# Handle missing values in categorical columns by filling with mode (most frequent value)
cat_col = ['Department', 'Gender','Education_Level']  # Adjust based on your dataset (no Education_Level here for one-hot encoding)
for col in cat_col:
    df[col] = df[col].fillna(df[col].mode()[0])

## Exploratory Data Analysis (EDA)
### Checking Categorical variables effect performance score
for col in df_cat:
    plt.figure()
    (df.groupby(col)['Performance_Score'].mean().sort_values(ascending=False).plot(kind='bar'))
    plt.plot()

df_num = df.select_dtypes(include=['number'])
# Compute the correlation matrix
corr_matrix = df_num.corr()
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
plt.title('Correlation Matrix')
plt.show()

for col in df_num:
    print(col,len(df_num[col].unique()), sep=':')

df_num.Performance_Score.value_counts()

df_num.loc[df_num.Performance_Score == 5].sample(5)

df_num.loc[df_num.Performance_Score == 1].sample(5)

df.groupby('Performance_Score').Age.mean()

df.groupby('Performance_Score').Employee_Satisfaction_Score.mean()

df.head()

# Numerical vs Categorical: Performance Score vs. Department
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Performance_Score',hue='Department', data=df, palette='Set2')
plt.title('Performance Score vs Department')
plt.xticks(rotation=45)
plt.show()

# Performance Score vs Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Performance_Score',hue='Gender', data=df, palette='Set2')
plt.title('Performance Score vs Gender')
plt.show()

# Performance Score vs Education Level
plt.figure(figsize=(10, 6))
sns.boxplot(x='Education_Level', y='Performance_Score',hue='Education_Level', data=df, palette='Set2')
plt.title('Performance Score vs Education Level')
plt.show()

# Visualize value counts of categorical variables
for col in df_cat:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=col,hue=col, palette='Set2')
    plt.title(f'Number of Employees by {col}')
    plt.xticks(rotation=45)
    plt.show()

# Salary Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Monthly_Salary'], kde=True, color='blue')
plt.title("Salary Distribution")
plt.xlabel("Monthly Salary")
plt.ylabel("Frequency")
plt.show()

# Work Hours vs Performance Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Work_Hours_Per_Week', y='Performance_Score', data=df, color='green')
plt.title("Work Hours vs Performance Score")
plt.xlabel("Work Hours Per Week")
plt.ylabel("Performance Score")
plt.show()

# Work Hours Per Week Distribution (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Work_Hours_Per_Week', data=df, color='orange')
plt.title("Work Hours Per Week Distribution")
plt.xlabel("Work Hours Per Week")
plt.show()

# Employee Satisfaction vs. Performance Score (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Employee_Satisfaction_Score', y='Performance_Score', data=df, hue='Department', palette='Set2')
plt.title("Employee Satisfaction vs Performance Score")
plt.xlabel("Employee Satisfaction Score")
plt.ylabel("Performance Score")
plt.legend(title='Department')
plt.show()

# Years at Company vs Monthly Salary (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Years_At_Company', y='Monthly_Salary', data=df, color='red')
plt.title("Years at Company vs Monthly Salary")
plt.xlabel("Years at Company")
plt.ylabel("Monthly Salary")
plt.show()

## Feature Engineering
from sklearn.preprocessing import StandardScaler

# One-Hot Encoding for Nominal Variables (non-ordinal categorical variables)
nominal_columns = ['Department', 'Gender','Job_Title']  # Removed Education_Level here
df_encoded = pd.get_dummies(df, columns=nominal_columns, drop_first=True)  # Drop the first column to avoid multicollinearity

# Manual Mapping for Ordinal Variables
# Suppose 'Education_Level' is ordinal with an inherent ranking.
education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}  # Adjust based on actual data
df_encoded['Education_Level'] = df_encoded['Education_Level'].map(education_mapping)

# Assuming `df_encoded` is your DataFrame with encoded columns

# List of numerical columns to scale (replace with the actual numerical columns in your dataset)
numerical_columns = ['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 'Overtime_Hours',
                     'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 'Training_Hours', 'Promotions',
                     'Employee_Satisfaction_Score','Years_At_Company']  # Adjust column names as per your dataset

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling to the numerical columns
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

# Display the first few rows of the scaled DataFrame
print(df_encoded.head())


## Model Development
from sklearn.model_selection import train_test_split
import pickle

X = df_encoded.drop(columns=['Employee_ID','Performance_Score', 'Hire_Date']) # Drop the target and any unwanted columns
y = df_encoded['Performance_Score']  # Target variable: Performance_Score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_save = pd.DataFrame(X_test)
df_save['Performance_Score'] = y_test

df_save.to_csv('inference_data.csv', index=False)
print("20% of the data saved to 'inference_data.csv'.")

## Load the saved Test Data
import pandas as pd

# Load the saved 20% data from the CSV file
df_loaded = pd.read_csv('inference_data.csv')

# Separate features (x) and target (y)
x_test1 = df_loaded.drop(columns=['Performance_Score'])  # Features (excluding the target)
y_test1 = df_loaded['Performance_Score']  # Target variable

### KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Initialize and train the K-Nearest Neighbors model
knn= KNeighborsClassifier(n_neighbors=15, metric='manhattan')
knn.fit(X_train, y_train)

# Save the model using pickle
with open('knn_classifier_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
print("K-Nearest Neighbors model saved as 'knn_classifier_model.pkl'.")

from sklearn.metrics import accuracy_score, confusion_matrix

# Load the saved KNN Classifier model
with open('knn_classifier_model.pkl', 'rb') as f:
    loaded_knn_clf = pickle.load(f)

# Make predictions on the test data
y_pred_knn= loaded_knn_clf.predict(x_test1)

# Evaluate the model's performance
accuracy_knn = accuracy_score(y_test1, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test1, y_pred_knn)

print("\nK-Nearest Neighbors Model Evaluation:")
print(f"Accuracy: {accuracy_knn:.4f}")
print("Confusion Matrix:")
print(conf_matrix_knn)

### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, random_state=42)
rf.fit(X_train, y_train)
with open('rf_classifier_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("Random Forest model saved as 'rf_classifier_model.pkl'.")

# Load the saved Random Forest Classifier model
with open('rf_classifier_model.pkl', 'rb') as f:
    loaded_rf_clf = pickle.load(f)

# Make predictions on the test data for Random Forest
y_pred_rf = loaded_rf_clf.predict(x_test1)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test1, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test1, y_pred_rf)
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_rf:.4f}")
print("Confusion Matrix:")
print(conf_matrix_rf)

### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

with open('dt_classifier_model.pkl', 'wb') as f:
    pickle.dump(dt, f)
print("Decision Tree model saved as 'dt_classifier_model.pkl'.")

# Load the saved Decision Tree Classifier model
with open('dt_classifier_model.pkl', 'rb') as f:
    loaded_dt_clf = pickle.load(f)

# Make predictions on the test data for Decision Tree
y_pred_dt = loaded_dt_clf.predict(x_test1)

# Evaluate Decision Tree model
accuracy_dt = accuracy_score(y_test1, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test1, y_pred_dt)
print("\nDecision Tree Model Evaluation:")
print(f"Accuracy: {accuracy_dt:.4f}")
print("Confusion Matrix:")
print(conf_matrix_dt)

### Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
with open('lr_classifier_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("Logistic Regression model saved as 'lr_classifier_model.pkl'.")

# Load the saved Logistic Regression Classifier model
with open('lr_classifier_model.pkl', 'rb') as f:
    loaded_lr_clf = pickle.load(f)

# Make predictions on the test data for Logistic Regression
y_pred_lr = loaded_lr_clf.predict(x_test1)

# Evaluate Logistic Regression model
accuracy_lr = accuracy_score(y_test1, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test1, y_pred_lr)
print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_lr:.4f}")
print("Confusion Matrix:")
print(conf_matrix_lr)

## Build Stacked model
from sklearn.ensemble import StackingClassifier

meta_model = LogisticRegression(random_state=42)

stacked_model = StackingClassifier(
    estimators=[('knn', knn), ('dt', dt), ('rf', rf)],
    final_estimator=meta_model
)

# Train the stacked model
stacked_model.fit(X_train, y_train)

with open('stacked_classifier_model.pkl', 'wb') as f:
    pickle.dump(stacked_model, f)

print("Stacked model saved as 'stacked_classifier_model.pkl'.")

with open('stacked_classifier_model.pkl', 'rb') as f:
    loaded_stacked_model = pickle.load(f)

# Make predictions on the test data
y_pred_stacked = loaded_stacked_model.predict(x_test1)

# Evaluate the stacked model
accuracy_stacked = accuracy_score(y_test1, y_pred_stacked)
conf_matrix_stacked = confusion_matrix(y_test1, y_pred_stacked)

print("\nStacked Model Evaluation:")
print(f"Accuracy: {accuracy_stacked:.4f}")
print("Confusion Matrix:")
print(conf_matrix_stacked)

