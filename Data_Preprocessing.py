import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

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
    scaler1 = pickle.load(scaler_file)

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
    df_encoded[numerical_columns] = scaler1.transform(df_encoded[numerical_columns])
    
    if fit:
        # Save the feature names (for reordering the prediction data)
        return df_encoded, scaler1, df_encoded.columns
    else:
        return df_encoded

def data_preprocessing():
    st.title("Data Preprocessing")
    
    st.markdown("""
    ## Data Preprocessing
    This page allows you to preprocess your dataset by:
    - Changing data types of columns  
    - Removing unnecessary columns  
    - Handling missing values
    - Encoding categorical variables
    - Scaling numerical columns
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file is not None:
        # Load the dataset into a DataFrame and store it in session state
        if 'df' not in st.session_state:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df.copy()  # Store a copy of the data to preserve the original

        else:
            df = st.session_state.df  # Use the copy stored in session state

        # Show the dataset preview
        st.subheader("Dataset Preview")
        st.write(df.head())
        
        # Step 1: Remove Unnecessary Columns
        st.subheader("Remove Unnecessary Columns")
        columns_to_remove = st.multiselect(
            "Select columns to remove:",
            options=df.columns.tolist(),
            default=[]
        )
        
        if columns_to_remove:
            df = remove_unnecessary_columns(df, columns_to_remove)
            st.session_state.df = df.copy()  # Save the updated DataFrame back to session state
            st.write(f"The following columns have been removed: {', '.join(columns_to_remove)}.")
        
        st.subheader("Preprocessed Data (After Removing Columns)")
        st.write(df.head())
        
        # Step 2: Change Data Type
        st.subheader("Change Data Types of Columns")
        st.write("Here are the columns and their current data types:")
        st.write(df.dtypes)
        
        column_to_change = st.selectbox("Select Column to Change Data Type:", df.columns)
        data_type_option = st.selectbox(
            "Select New Data Type:",
            ["int64", "float64", "str", "category"]
        )
        
        if st.button("Change Data Type"):
            df = change_data_type(df, column_to_change, data_type_option)
            st.session_state.df = df.copy()  # Save the updated DataFrame back to session state
            st.write(f"Column `{column_to_change}` has been changed to `{data_type_option}`.")
        
        st.subheader("DataFrame After Data Type Change")
        st.write(df.dtypes)
        
        # Step 3: Handle Missing Values
        st.subheader("Check for Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values)
        
        st.subheader("Handle Missing Values")
        missing_option = st.selectbox('Choose Missing Value Handling Method:', ['Skip','Fill with Mean (Numeric)', 'Fill with Mode (Categorical)'])
        
        if missing_option == 'Fill with Mean (Numeric)':
            df = handle_missing_values(df, method='mean')
            st.session_state.df = df.copy()  # Save the updated DataFrame back to session state
            st.write("Missing numeric values have been filled with the mean.")
        elif missing_option == 'Fill with Mode (Categorical)':
            df = handle_missing_values(df, method='mode')
            st.session_state.df = df.copy()  # Save the updated DataFrame back to session state
            st.write("Missing categorical values have been filled with the mode.")
            
        st.subheader("Preprocessed Data")
        st.write(df.head())
        
        # Step 4: Scaling Numerical Columns
        st.subheader("Scale Numerical Columns")
        scale_option = st.selectbox('Choose Scaling Method:', ['Skip','StandardScaler', 'MinMaxScaler'])
        
        if scale_option == 'StandardScaler':
            df = scale_numerical_columns(df, scale_method='StandardScaler')
            st.session_state.df = df.copy()  # Save the updated DataFrame back to session state
            st.write("Numerical columns have been scaled using StandardScaler.")
            
        elif scale_option == 'MinMaxScaler':
            df = scale_numerical_columns(df, scale_method='MinMaxScaler')
            st.session_state.df = df.copy()  # Save the updated DataFrame back to session state
            st.write("Numerical columns have been scaled using MinMaxScaler.")
            
        st.subheader("Preprocessed Data (After Scaling)")
        st.write(df.head())
        
        # Step 5: Encoding Categorical Columns
        st.subheader("Encoding Categorical Columns")
        encoding_option = st.selectbox('Choose Encoding Method:', ['Skip','One-Hot Encoding', 'Label Encoding'])
        
        if encoding_option == 'One-Hot Encoding':
            df = encode_categorical_columns(df, encoding_method='One')
            st.session_state.df = df.copy()  # Save the updated DataFrame back to session state
            st.write("Categorical columns have been encoded using One-Hot Encoding.")
            st.session_state.encoding_option = 'Skip'
            
        
        elif encoding_option == 'Label Encoding':
            df = encode_categorical_columns(df, encoding_method='Label')
            st.session_state.df = df.copy()  # Save the updated DataFrame back to session state
            st.write("Categorical columns have been encoded using Label Encoding.")
        
        st.subheader("Preprocessed Data (After Encoding)")
        st.write(df.head())
        
        # Download the preprocessed dataset
        st.subheader("Download Preprocessed Dataset")
        csv = df.to_csv(index=False)
        st.download_button(label="Download Preprocessed Data", data=csv, file_name="preprocessed_data.csv", mime="text/csv")
