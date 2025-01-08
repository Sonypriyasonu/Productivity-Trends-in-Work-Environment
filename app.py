import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.markdown(
    """
    <style>   
        .st-emotion-cache-ocsh0s  {
        width: 223px;
        height: 50px;
        font-size: 1em;
        border-radius: 12px;
        border: none;
        background-color: skyblue; 
        color: white;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        cursor: pointer;
    }
    .st-emotion-cache-ocsh0s:hover {
        color: black;  
    }

    .st-emotion-cache-ocsh0s:active {
        background-color: #4682B4;  
        color: black;  
        
    </style>
    """,
    unsafe_allow_html=True
)
 

# Load the dataset
def load_data():
    return pd.read_csv('Employee_Productivity_Data.csv')

df = load_data()
#df = df.drop(columns=['Employee_ID', 'Hire_Date'], errors='ignore')


model = pickle.load(open('stacked_classifier_model.pkl', 'rb'))

# One-Hot Encoding for Nominal Variables (non-ordinal categorical variables)
nominal_columns = ['Department', 'Gender', 'Job_Title']  # Non-ordinal categorical columns
education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}  # Manual mapping for ordinal variable

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Preprocessing: One-Hot Encoding for nominal columns and mapping for ordinal columns
def preprocess_data(df, fit=False):
    
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


# Preprocess the training dataset
df_encoded, scaler, train_columns = preprocess_data(df, fit=True)

#inference_df = pd.read_csv('inference_data.csv')


# Sidebar for navigation
st.sidebar.title("Employee Productivity")
page = st.sidebar.radio("Select a Page", ["Objective", "Dataset Description", "Data Visualization", "Predict Performance"])

        
if page == "Objective":
    st.title("Employee Productivity Analysis Project")
    st.markdown("""
    ## Objective:
    The objective of this project is to analyze employee performance trends and predict their productivity scores based on various factors, such as:
    - **Work hours**, **salary**, **performance scores**, **employee satisfaction**, and more.
    - Identify trends in factors like **remote work**, **overtime hours**, **education**, **years at the company**, etc.
    
    By leveraging this data, the goal is to identify key trends that can help improve HR strategies, enhance employee engagement, and predict performance outcomes.
    """)

# Dataset Description Page
if page == "Dataset Description":
    st.title("Employee Productivity Dataset")
    st.markdown("""
    ## Dataset Description
    This dataset contains 100,000 rows of data capturing key aspects of employee performance, productivity, and demographics in a corporate environment. 
    It includes details related to the employee's job, work habits, education, performance, and satisfaction. The dataset is designed for various purposes such as HR analytics, employee churn prediction, productivity analysis, and performance evaluation.

    **Columns in the dataset:**
    - **Employee_ID**: Unique identifier for each employee.
    - **Department**: The department in which the employee works (e.g., Sales, HR, IT).
    - **Gender**: Gender of the employee (Male, Female, Other).
    - **Age**: Employee's age (between 22 and 60).
    - **Job_Title**: The role held by the employee (e.g., Manager, Analyst, Developer).
    - **Hire_Date**: The date the employee was hired.
    - **Years_At_Company**: The number of years the employee has been working for the company.
    - **Education_Level**: Highest educational qualification (High School, Bachelor, Master, PhD).
    - **Performance_Score**: Employee's performance rating (1 to 5 scale).
    - **Monthly_Salary**: The employee's monthly salary in USD, correlated with job title and performance score.
    - **Work_Hours_Per_Week**: Number of hours worked per week.
    - **Projects_Handled**: Total number of projects handled by the employee.
    - **Overtime_Hours**: Total overtime hours worked in the last year.
    - **Sick_Days**: Number of sick days taken by the employee.
    - **Remote_Work_Frequency**: Percentage of time worked remotely (0, 25, 50, 75, 100).
    - **Team_Size**: Number of people in the employee's team.
    - **Training_Hours**: Number of hours spent in training.
    - **Promotions**: Number of promotions received during their tenure.
    - **Employee_Satisfaction_Score**: Employee satisfaction rating (1.0 to 5.0 scale).
    - **Resigned**: Boolean value indicating if the employee has resigned (1 for Yes, 0 for No).
    """)

# Data Visualization Page
if page == "Data Visualization":
    st.title("Data Visualization")
    
    # Select visualization type
    chart_type = st.selectbox('Choose a Chart:', 
                              ['Performance Score vs Department', 
                               'Performance Score vs Gender', 
                               'Performance Score vs Education Level', 
                               'Categorical Data', 
                               'Employee Satisfaction vs Performance Score',
                               'Years at Company vs Monthly Salary', 
                               'Correlation Matrix', 'All Visualizations'])
    
    # Performance Score vs Department
    if chart_type == 'Performance Score vs Department':
        st.subheader("Performance Score vs Department")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Department', y='Performance_Score', hue='Department', data=df, palette='Set2')
        plt.title('Performance Score vs Department')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Performance Score vs Gender
    elif chart_type == 'Performance Score vs Gender':
        st.subheader("Performance Score vs Gender")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Gender', y='Performance_Score', hue='Gender', data=df, palette='Set2')
        plt.title('Performance Score vs Gender')
        st.pyplot(plt)
    
    # Performance Score vs Education Level
    elif chart_type == 'Performance Score vs Education Level':
        st.subheader("Performance Score vs Education Level")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Education_Level', y='Performance_Score', hue='Education_Level', data=df, palette='Set2')
        plt.title('Performance Score vs Education Level')
        st.pyplot(plt)
    
    # Categorical Data Analysis (Value Counts)
    elif chart_type == 'Categorical Data':
        st.subheader("Categorical Variables Analysis")
        
        df_cat = df.loc[:, df.dtypes == 'object']
        df_cat = df_cat.drop('Hire_Date', axis='columns')
        
        for col in df_cat:
            plt.figure(figsize=(8, 6))
            sns.countplot(data=df, x=col, hue=col, palette='Set2')
            plt.title(f'Number of Employees by {col}')
            plt.xticks(rotation=45)
            st.pyplot(plt)
    
    # Employee Satisfaction vs Performance Score (Scatter Plot)
    elif chart_type == 'Employee Satisfaction vs Performance Score':
        st.subheader("Employee Satisfaction vs Performance Score")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Employee_Satisfaction_Score', y='Performance_Score', data=df, hue='Department', palette='Set2')
        plt.title("Employee Satisfaction vs Performance Score")
        plt.xlabel("Employee Satisfaction Score")
        plt.ylabel("Performance Score")
        plt.legend(title='Department')
        st.pyplot(plt)
    
    # Years at Company vs Monthly Salary (Scatter Plot)
    elif chart_type == 'Years at Company vs Monthly Salary':
        st.subheader("Years at Company vs Monthly Salary")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Years_At_Company', y='Monthly_Salary', data=df, color='red')
        plt.title("Years at Company vs Monthly Salary")
        plt.xlabel("Years at Company")
        plt.ylabel("Monthly Salary")
        st.pyplot(plt)
    
    # Correlation Matrix
    elif chart_type == 'Correlation Matrix':
        st.subheader("Correlation Matrix of Numeric Features")
        
        # Handle missing values in numeric columns
        num_col = ['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 
                   'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 
                   'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score']
        
        for col in num_col:
            df[col] = df[col].fillna(df[col].mean())  # Fill missing values with mean
        
        # Compute the correlation matrix for numeric columns
        df_num = df.select_dtypes(include=['number'])
        corr_matrix = df_num.corr()

        # Plot the correlation matrix as a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
        plt.title('Correlation Matrix of Numeric Features')
        st.pyplot(plt)
    elif chart_type == "All Visualizations":
        st.title("All Visualizations")
        
        # Scrollable container for visualizations
        st.markdown(
            """
            <style>
            .scrollable-container {
                max-height: 1000px;
                overflow-y: scroll;
            }
            </style>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
            
            # Performance Score vs Department
            st.subheader("Performance Score vs Department")
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Department', y='Performance_Score', hue='Department', data=df, palette='Set2')
            plt.title('Performance Score vs Department')
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Performance Score vs Gender
            st.subheader("Performance Score vs Gender")
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Gender', y='Performance_Score', hue='Gender', data=df, palette='Set2')
            plt.title('Performance Score vs Gender')
            st.pyplot(plt)
        
            # Performance Score vs Education Level
            st.subheader("Performance Score vs Education Level")
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Education_Level', y='Performance_Score', hue='Education_Level', data=df, palette='Set2')
            plt.title('Performance Score vs Education Level')
            st.pyplot(plt)
        
            # Categorical Data Analysis (Value Counts)
            st.subheader("Categorical Variables Analysis")
            
            df_cat = df.loc[:, df.dtypes == 'object']
            df_cat = df_cat.drop('Hire_Date', axis='columns')
            
            for col in df_cat:
                plt.figure(figsize=(8, 6))
                sns.countplot(data=df, x=col, hue=col, palette='Set2')
                plt.title(f'Number of Employees by {col}')
                plt.xticks(rotation=45)
                st.pyplot(plt)
        
            # Employee Satisfaction vs Performance Score (Scatter Plot)
            st.subheader("Employee Satisfaction vs Performance Score")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Employee_Satisfaction_Score', y='Performance_Score', data=df, hue='Department', palette='Set2')
            plt.title("Employee Satisfaction vs Performance Score")
            plt.xlabel("Employee Satisfaction Score")
            plt.ylabel("Performance Score")
            plt.legend(title='Department')
            st.pyplot(plt)
        
            # Years at Company vs Monthly Salary (Scatter Plot)
            st.subheader("Years at Company vs Monthly Salary")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Years_At_Company', y='Monthly_Salary', data=df, color='red')
            plt.title("Years at Company vs Monthly Salary")
            plt.xlabel("Years at Company")
            plt.ylabel("Monthly Salary")
            st.pyplot(plt)
        
            # Correlation Matrix
            st.subheader("Correlation Matrix of Numeric Features")
            
            # Handle missing values in numeric columns
            num_col = ['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 
                    'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 
                    'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score']
            
            for col in num_col:
                df[col] = df[col].fillna(df[col].mean())  # Fill missing values with mean
            
            # Compute the correlation matrix for numeric columns
            df_num = df.select_dtypes(include=['number'])
            corr_matrix = df_num.corr()

            # Plot the correlation matrix as a heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
            plt.title('Correlation Matrix of Numeric Features')
            st.pyplot(plt)
        
            st.markdown('</div>', unsafe_allow_html=True)

if page == "Predict Performance":
    st.title("Predict Employee Performance")

    # Select mode (single employee or multiple employees)
    mode = st.radio("Select Mode", ["Single Employee", "Multiple Employees"])

    if mode == "Single Employee":
        st.subheader("Predict Performance for a Single Employee")

        # Add custom CSS for scrollable container
        st.markdown("""
        <style>
        .scrollable-container {
            max-height: 600px;
            overflow-y: scroll;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create a scrollable container
        with st.container():
            st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)

            # Input fields for the new employee's data
            department = st.selectbox("Department", ['IT', 'Finance', 'Customer Support', 'Engineering', 'Marketing', 'HR', 'Operations', 'Sales', 'Legal'])
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            age = st.number_input("Age", min_value=18, max_value=100, value=18)
            job_title = st.selectbox("Job Title", ['Specialist', 'Developer', 'Analyst', 'Manager', 'Technician', 'Engineer', 'Consultant'])
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=0)
            education_level = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'])
            monthly_salary = st.number_input("Monthly Salary (in USD)", min_value=1000, max_value=200000, value=1000)
            work_hours_per_week = st.number_input("Work Hours per Week", min_value=0, max_value=80, value=0)
            projects_handled = st.number_input("Projects Handled", min_value=0, max_value=50, value=0)
            overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=100, value=0)
            sick_days = st.number_input("Sick Days", min_value=0, max_value=30, value=0)
            remote_work_frequency = st.selectbox("Remote Work Frequency", ['0', '25', '50', '75', '100'])
            team_size = st.number_input("Team Size", min_value=1, max_value=20, value=1)
            training_hours = st.number_input("Training Hours", min_value=0, max_value=200, value=0)
            promotions = st.number_input("Promotions", min_value=0, max_value=10, value=0)
            employee_satisfaction_score = st.number_input("Employee Satisfaction Score (1-5)", min_value=1.0, max_value=5.0, value=1.0)
            # Prepare input data for prediction
            input_data = {
                'Department': department,
                'Gender': gender,
                'Age': age,
                'Job_Title': job_title,
                'Years_At_Company': years_at_company,
                'Education_Level': education_level,
                'Monthly_Salary': monthly_salary,
                'Work_Hours_Per_Week': work_hours_per_week,
                'Projects_Handled': projects_handled,
                'Overtime_Hours': overtime_hours,
                'Sick_Days': sick_days,
                'Remote_Work_Frequency': remote_work_frequency,
                'Team_Size': team_size,
                'Training_Hours': training_hours,
                'Promotions': promotions,
                'Employee_Satisfaction_Score': employee_satisfaction_score
            }

            input_df = pd.DataFrame([input_data])

            # Preprocess the input data (encoding and scaling)
            input_df_encoded = preprocess_data(input_df)

            # Align the columns of input_df_encoded with the training columns
            input_df_encoded = input_df_encoded.reindex(columns=train_columns, fill_value=0)

            # Predict performance score using the loaded model
            if st.button("Predict Performance Score"):
                prediction = model.predict(input_df_encoded)
                st.write(f"Predicted Performance Score: {prediction[0]}")

            st.markdown('</div>', unsafe_allow_html=True)
    elif mode == "Multiple Employees":
        st.subheader("Predict Performance for Multiple Employees")
        
        uploaded_file = st.file_uploader("Upload Inference Data (CSV)", type="csv")
        
        if uploaded_file is not None:
        # Load the uploaded CSV file
            inference_df = pd.read_csv(uploaded_file)

            # Ensure the necessary columns are present in the uploaded file
            if 'Performance_Score' not in inference_df.columns:
                st.error("Uploaded file must contain 'Performance_Score' column.")
            else:
                # Drop 'Performance_Score' from features (X) and separate it (y)
                X_inference = inference_df.drop(columns=['Performance_Score'])
                y_true = inference_df['Performance_Score']

                # ---- KNN Classifier Evaluation ----
                st.subheader("KNN Classifier Model Performance:")
                knn_model = pickle.load(open('knn_classifier_model.pkl', 'rb'))
                knn_predictions = knn_model.predict(X_inference)

                # KNN Model Accuracy and Confusion Matrix
                knn_accuracy = accuracy_score(y_true, knn_predictions)
                knn_conf_matrix = confusion_matrix(y_true, knn_predictions)

                # Display KNN model performance
                st.write(f"**Accuracy:** {knn_accuracy:.4f}")
                st.write("**Confusion Matrix:**")
                st.write(knn_conf_matrix)

                # Display KNN predictions
                inference_df['KNN_Predicted_Performance_Score'] = knn_predictions

                # Display the first 10 rows for KNN model
                st.write("**First 10 KNN Predictions:**")
                st.write(inference_df[['Performance_Score', 'KNN_Predicted_Performance_Score']].head(10))

                # KNN Model Download (green button)
                knn_csv = inference_df[['Performance_Score', 'KNN_Predicted_Performance_Score']].to_csv(index=False)
                
                st.download_button(
                    label="Download KNN Predicted Performance Scores",
                    data=knn_csv,
                    file_name="knn_predicted_performance_scores.csv",
                    mime="text/csv"
                )

                # ---- Decision Tree Classifier Evaluation ----
                st.subheader("Decision Tree Classifier Model Performance:")
                dt_model = pickle.load(open('dt_classifier_model.pkl', 'rb'))
                dt_predictions = dt_model.predict(X_inference)

                # Decision Tree Model Accuracy and Confusion Matrix
                dt_accuracy = accuracy_score(y_true, dt_predictions)
                dt_conf_matrix = confusion_matrix(y_true, dt_predictions)

                # Display Decision Tree model performance
                st.write(f"**Accuracy:** {dt_accuracy:.4f}")
                st.write("**Confusion Matrix:**")
                st.write(dt_conf_matrix)

                # Display Decision Tree predictions
                inference_df['DT_Predicted_Performance_Score'] = dt_predictions

                # Display the first 10 rows for Decision Tree model
                st.write("**First 10 Decision Tree Predictions:**")
                st.write(inference_df[['Performance_Score', 'DT_Predicted_Performance_Score']].head(10))

                # Decision Tree Model Download (green button)
                dt_csv = inference_df[['Performance_Score', 'DT_Predicted_Performance_Score']].to_csv(index=False)
               
                st.download_button(
                    label="Download DT Predicted Performance Scores",
                    data=dt_csv,
                    file_name="dt_predicted_performance_scores.csv",
                    mime="text/csv"
                )

                # ---- Logistic Regression Evaluation ----
                st.subheader("Logistic Regression Model Performance:")
                lr_model = pickle.load(open('lr_classifier_model.pkl', 'rb'))
                lr_predictions = lr_model.predict(X_inference)

                # Logistic Regression Model Accuracy and Confusion Matrix
                lr_accuracy = accuracy_score(y_true, lr_predictions)
                lr_conf_matrix = confusion_matrix(y_true, lr_predictions)

                # Display Logistic Regression model performance
                st.write(f"**Accuracy:** {lr_accuracy:.4f}")
                st.write("**Confusion Matrix:**")
                st.write(lr_conf_matrix)

                # Display Logistic Regression predictions
                inference_df['LR_Predicted_Performance_Score'] = lr_predictions

                # Display the first 10 rows for Logistic Regression model
                st.write("**First 10 Logistic Regression Predictions:**")
                st.write(inference_df[['Performance_Score', 'LR_Predicted_Performance_Score']].head(10))

                # Logistic Regression Model Download (green button)
                lr_csv = inference_df[['Performance_Score', 'LR_Predicted_Performance_Score']].to_csv(index=False)
                
                st.download_button(
                    label="Download LR Predicted Performance Scores",
                    data=lr_csv,
                    file_name="lr_predicted_performance_scores.csv",
                    mime="text/csv"
                )

                # ---- Random Forest Classifier Evaluation ----
                st.subheader("Random Forest Classifier Model Performance:")
                rf_model = pickle.load(open('rf_classifier_model.pkl', 'rb'))
                rf_predictions = rf_model.predict(X_inference)

                # Random Forest Model Accuracy and Confusion Matrix
                rf_accuracy = accuracy_score(y_true, rf_predictions)
                rf_conf_matrix = confusion_matrix(y_true, rf_predictions)

                # Display Random Forest model performance
                st.write(f"**Accuracy:** {rf_accuracy:.4f}")
                st.write("**Confusion Matrix:**")
                st.write(rf_conf_matrix)

                # Display Random Forest predictions
                inference_df['RF_Predicted_Performance_Score'] = rf_predictions

                # Display the first 10 rows for Random Forest model
                st.write("**First 10 Random Forest Predictions:**")
                st.write(inference_df[['Performance_Score', 'RF_Predicted_Performance_Score']].head(10))

                # Random Forest Model Download (green button)
                rf_csv = inference_df[['Performance_Score', 'RF_Predicted_Performance_Score']].to_csv(index=False)
                
                st.download_button(
                    label="Download RF Predicted Performance Scores",
                    data=rf_csv,
                    file_name="rf_predicted_performance_scores.csv",
                    mime="text/csv"
                )

                # ---- Stacked Classifier Evaluation ----
                st.subheader("Stacked Classifier Model Performance:")
                stacked_model = pickle.load(open('stacked_classifier_model.pkl', 'rb'))
                stacked_predictions = stacked_model.predict(X_inference)

                # Stacked Classifier Model Accuracy and Confusion Matrix
                stacked_accuracy = accuracy_score(y_true, stacked_predictions)
                stacked_conf_matrix = confusion_matrix(y_true, stacked_predictions)

                # Display Stacked Classifier model performance
                st.write(f"**Accuracy:** {stacked_accuracy:.4f}")
                st.write("**Confusion Matrix:**")
                st.write(stacked_conf_matrix)

                # Display Stacked Classifier predictions
                inference_df['Stacked_Predicted_Performance_Score'] = stacked_predictions

                # Display the first 10 rows for Stacked Classifier model
                st.write("**First 10 Stacked Classifier Predictions:**")
                st.write(inference_df[['Performance_Score', 'Stacked_Predicted_Performance_Score']].head(10))

                # Stacked Classifier Model Download (green button)
                stacked_csv = inference_df[['Performance_Score', 'Stacked_Predicted_Performance_Score']].to_csv(index=False)
    
                st.download_button(
                    label="Download SC Predicted Performance Scores",
                    data=stacked_csv,
                    file_name="stacked_predicted_performance_scores.csv",
                    mime="text/csv"
                )

        else:
            st.warning("Please upload a CSV file to proceed.")
            
