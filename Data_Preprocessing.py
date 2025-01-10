import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle

def handle_missing_values(df, method='mean'):
    # Handle missing values in numeric columns by filling with mean
    if method=='mean':
        num_col = ['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 
                'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 
                'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score']  # Adjust based on your dataset
        for col in num_col:
            df[col] = df[col].fillna(df[col].mean())
        return df
    if method=='Mode':
        cat_col = ['Department', 'Gender', 'Education_Level','Job_Title']  # Adjust based on your dataset
        for col in cat_col:
            df[col] = df[col].fillna(df[col].mode()[0])
    
        return df

def encode_categorical_columns(df, encoding_method='One'):
    if encoding_method == 'One':
        # One-Hot Encoding for Nominal Columns (non-ordinal categorical columns)
        #nominal_columns = ['Department', 'Gender', 'Job_Title']  # Non-ordinal categorical columns
        df_encoded = pd.get_dummies(df, columns=['Department', 'Gender', 'Job_Title'] , drop_first=True)
        
        # Map Education Level (Ordinal Variable)
        education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}  # Manual mapping for ordinal variable
        df_encoded['Education_Level'] = df_encoded['Education_Level'].map(education_mapping)
        
        return df_encoded
    
    elif encoding_method == 'Label':
        # Label Encoding for Categorical Variables
        label_encoder = LabelEncoder()
        cat_columns = ['Department', 'Gender', 'Job_Title', 'Education_Level']
        
        for col in cat_columns:
            df[col] = label_encoder.fit_transform(df[col])
        
        return df

def scale_numerical_columns(df, scale_method='StandardScaler'):

    numerical_columns = ['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 
                         'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 
                         'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score', 'Years_At_Company']
    
    # Instantiate the scaler if it is not passed
    if scale_method == 'StandardScaler':
        
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        
        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
    
    elif scale_method == 'MinMaxScaler':
        min_max_scaler = MinMaxScaler()
        df[numerical_columns] = min_max_scaler.fit_transform(df[numerical_columns])
    
    return df

def change_data_type(df, column_name, new_type):
    if new_type == "int64":
        df[column_name] = df[column_name].astype("int64")
    elif new_type == "float64":
        df[column_name] = df[column_name].astype("float64")
    elif new_type == "str":
        df[column_name] = df[column_name].astype("str")
    elif new_type == "category":
        df[column_name] = df[column_name].astype("category")
    else:
        raise ValueError(f"Unsupported data type: {new_type}")
    
    return df
def remove_unnecessary_columns(df, columns_to_remove):
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Preprocessing: One-Hot Encoding for nominal columns and mapping for ordinal columns
def preprocess_data(df, fit=False):
    nominal_columns = ['Department', 'Gender', 'Job_Title']  # Non-ordinal categorical columns
    education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}  # Manual mapping for ordinal variable
    df = df.drop(columns=['Employee_ID', 'Hire_Date', 'Performance_Score'], errors='ignore')

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

    # Map Education Level (Ordinal Variable)
    df_encoded['Education_Level'] = df_encoded['Education_Level'].map(education_mapping)
    
    # List of numerical columns to scale
    numerical_columns = ['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 
                         'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 
                         'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score', 'Years_At_Company']
        
    # Apply scaling to the numerical columns
    df_encoded[numerical_columns] = scaler.transform(df_encoded[numerical_columns])
    
    if fit:
        # Save the feature names (for reordering the prediction data)
        return df_encoded, scaler, df_encoded.columns
    else:
        return df_encoded
