import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from Data_Visualization import (performance_score_vs_department,performance_score_vs_gender,performance_score_vs_education_level,
    employee_satisfaction_vs_performance,years_at_company_vs_salary,correlation_matrix,number_of_employees_by_education_level)

from Data_Preprocessing import handle_missing_values, encode_categorical_columns, scale_numerical_columns, change_data_type,remove_unnecessary_columns,preprocess_data
from Model_Development import download_pickle,download_csv,train_knn, train_rf, train_dt, train_lr, train_stacked_model,evaluate_model

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


# Sidebar for navigation
st.sidebar.title("Employee Productivity")
page = st.sidebar.radio("Select a Page", ["Objective", "Dataset Description","Data Preprocessing", "Data Visualization", "Train Test Split Data","Model Development", "Predict Performance"])


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

# Data Preprocessing Page
if page == "Data Preprocessing":
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
        
# Data Visualization Page
if page == "Data Visualization":
    st.title("Data Visualization")
    st.markdown("""
    **Note:** 
     Please ensure that the dataset contains the following columns: 
    
    Employee_ID, Department, Gender, Age, Job_Title, Years_At_Company, Education_Level, Performance_Score, Monthly_Salary, Work_Hours_Per_Week, Projects_Handled, Overtime_Hours, Sick_Days, Remote_Work_Frequency, Team_Size, Training_Hours, Promotions, Employee_Satisfaction_Score.") 
    
    Missing any of these columns may result in incomplete or inaccurate analysis.
    """)   
    uploaded_file = st.file_uploader("Upload Inference Data (CSV)", type="csv")
       
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        chart_type = st.selectbox('Choose a Chart:', 
                                ['Performance Score vs Department', 
                                 'Performance Score vs Gender', 
                                 'Performance Score vs Education Level', 
                                 'Number of Employees by Education Level', 
                                 'Employee Satisfaction vs Performance Score',
                                 'Years at Company vs Monthly Salary', 
                                 'Correlation Matrix', 'All Visualizations'])
        
        if chart_type == 'Performance Score vs Department':
            st.subheader("Performance Score vs Department")
            fig = performance_score_vs_department(df)
            st.pyplot(fig)

        elif chart_type == 'Performance Score vs Gender':
            st.subheader("Performance Score vs Gender")
            fig = performance_score_vs_gender(df)
            st.pyplot(fig)
        
        elif chart_type == 'Performance Score vs Education Level':
            st.subheader("Performance Score vs Education Level")
            fig = performance_score_vs_education_level(df)
            st.pyplot(fig)
        
        elif chart_type == 'Number of Employees by Education Level':
            st.subheader("Number of Employees by Education Level")
            fig = number_of_employees_by_education_level(df)
            st.pyplot(fig)
        
        elif chart_type == 'Employee Satisfaction vs Performance Score':
            st.subheader("Employee Satisfaction vs Performance Score")
            fig = employee_satisfaction_vs_performance(df)
            st.pyplot(fig)
        
        elif chart_type == 'Years at Company vs Monthly Salary':
            st.subheader("Years at Company vs Monthly Salary")
            fig = years_at_company_vs_salary(df)
            st.pyplot(fig)
        
        elif chart_type == 'Correlation Matrix':
            st.subheader("Correlation Matrix of Numeric Features")
            fig = correlation_matrix(df)
            st.pyplot(fig)
        
        elif chart_type == "All Visualizations":
            st.subheader("All Visualizations")
            fig = performance_score_vs_department(df)
            st.pyplot(fig)
            
            fig = performance_score_vs_gender(df)
            st.pyplot(fig)
            
            fig = performance_score_vs_education_level(df)
            st.pyplot(fig)
            
            fig = number_of_employees_by_education_level(df)
            st.pyplot(fig)
            
            fig = employee_satisfaction_vs_performance(df)
            st.pyplot(fig)
            
            fig = years_at_company_vs_salary(df)
            st.pyplot(fig)
            
            fig = correlation_matrix(df)
            st.pyplot(fig)
    else:
        st.warning("Please upload a CSV file to proceed.")
        
if page == "Train Test Split Data":
    st.title("Train Test Split Data")
    
    st.subheader("Step 1: Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Check if file is uploaded
    if uploaded_file is not None:
        # Read the uploaded CSV into a DataFrame
        if 'df' not in st.session_state:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.write("Data Preview:", df.head())  # Display a preview of the data
        else:
            df = st.session_state.df  # Use the DataFrame from session state

        # Step 2: Select Y (target) column
        st.subheader("Step 2: Select Target Column (Y)")
        
        if 'y_column' not in st.session_state:
            columns = df.columns.tolist()
            y_column = 'Performance_Score' if 'Performance_Score' in columns else st.selectbox("Select column for Y (target)", columns)           
            st.session_state.y_column = y_column  # Store selected target column
        else:
            y_column = st.session_state.y_column  # Use the saved target column

        X_columns = [col for col in df.columns if col != y_column]

        if X_columns and y_column:
            st.write(f"Features (X): {X_columns}")
            st.write(f"Target (Y): {y_column}")

            # Step 3: Train-Test Split (automatically performed)
            st.subheader("Step 3: Train-Test Split")
            test_size = st.slider("Select test size (as a percentage of total data)", 10, 90, 20) / 100
            random_state = st.number_input("Enter random state for reproducibility", value=42)

            # Split the data automatically after user selections
            X = df[X_columns]
            y = df[y_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Combine X_test and y_test for inference (test data)
            test_data = X_test.copy()
            test_data[y_column] = y_test

            # Combine X_train and y_train for training (train data)
            train_data = X_train.copy()
            train_data[y_column] = y_train

            st.session_state.train_data = train_data
            st.session_state.test_data = test_data

            # Show the split dataframes
            st.write("Train Data (Features):", X_train.head())
            st.write("Test Data (Features):", X_test.head())

            # Step 4: Download Data as CSV (automatically displayed)
            st.subheader("Step 4: Download Data as CSV")

            # Download train data
            train_data_csv = download_csv(st.session_state.train_data, "train_data.csv")
            st.download_button(
                label="Download Train Data",
                data=train_data_csv,
                file_name="train_data.csv",
                mime="text/csv"
            )

            # Download test data
            test_data_csv = download_csv(st.session_state.test_data, "test_data.csv")
            st.download_button(
                label="Download Test Data",
                data=test_data_csv,
                file_name="test_data.csv",
                mime="text/csv"
            )

        else:
            st.warning("Please select the target column (Y).")

if page=="Model Development":
    st.title("Model Development")
    
    st.markdown("""
    ## Model Development
    This page allows you to:
    - Upload preprocessed data for training
    - Train multiple models (KNN, Random Forest, Decision Tree, Logistic Regression)
    - Evaluate model accuracy and confusion matrix
    - Download pickle files of trained models
    """)
    
    uploaded_file = st.file_uploader("Upload Train Data CSV File", type="csv")
    
    if uploaded_file is not None:
        # Load the preprocessed data into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        # Show a preview of the dataset
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Automatically select the target column 'Performance_Score'
        target_column = 'Performance_Score' if 'Performance_Score' in df.columns else st.selectbox("Select Target Column", options=df.columns.tolist())

        # Only proceed if a target column is selected
        if target_column:
            # Split the data into features (X) and target (y)
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            # Train-Test Split (for reference if you need it)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Selection Dropdown
            model_option = st.selectbox(
            "Select Model to Train",
            options=["Select a Model", "K-Nearest Neighbors", "Random Forest", "Decision Tree", "Logistic Regression", "Stacking Classifier"]
            )

            # Train and evaluate the selected model automatically
            if model_option == "K-Nearest Neighbors":
                knn_model = train_knn(X_train, y_train)
                accuracy, conf_matrix_figure = evaluate_model(knn_model, X_train, y_train, model_name="K-Nearest Neighbors")
                
                # Display accuracy
                st.subheader("Training Performance for K-Nearest Neighbors")
                st.write(f"Accuracy: {accuracy:.4f}")
                
                # Display confusion matrix
                st.write("Confusion Matrix for K-Nearest Neighbors:")
                st.pyplot(conf_matrix_figure)  # Display the confusion matrix heatmap

                st.download_button(
                    label="Download KNN Model (Pickle)",
                    data=download_pickle(knn_model, "knn_model.pkl"),
                    file_name="knn_model.pkl",
                    mime="application/octet-stream"
                )

            elif model_option == "Random Forest":
                rf_model = train_rf(X_train, y_train)
                accuracy, conf_matrix_figure = evaluate_model(rf_model, X_train, y_train, model_name="Random Forest")
                
                # Display accuracy
                st.subheader("Training Performance for Random Forest")
                st.write(f"Accuracy: {accuracy:.4f}")
                
                # Display confusion matrix
                st.write("Confusion Matrix for Random Forest:")
                st.pyplot(conf_matrix_figure)  # Display the confusion matrix heatmap

                st.download_button(
                    label="Download Random Forest Model (Pickle)",
                    data=download_pickle(rf_model, "random_forest_model.pkl"),
                    file_name="random_forest_model.pkl",
                    mime="application/octet-stream"
                )

            elif model_option == "Decision Tree":
                dt_model = train_dt(X_train, y_train)
                accuracy, conf_matrix_figure = evaluate_model(dt_model, X_train, y_train, model_name="Decision Tree")
                
                # Display accuracy
                st.subheader("Training Performance for Decision Tree")
                st.write(f"Accuracy: {accuracy:.4f}")
                
                # Display confusion matrix
                st.write("Confusion Matrix for Decision Tree:")
                st.pyplot(conf_matrix_figure)  # Display the confusion matrix heatmap

                st.download_button(
                    label="Download Decision Tree Model (Pickle)",
                    data=download_pickle(dt_model, "decision_tree_model.pkl"),
                    file_name="decision_tree_model.pkl",
                    mime="application/octet-stream"
                )

            elif model_option == "Logistic Regression":
                lr_model = train_lr(X_train, y_train)
                accuracy, conf_matrix_figure = evaluate_model(lr_model, X_train, y_train, model_name="Logistic Regression")
                
                # Display accuracy
                st.subheader("Training Performance for Logistic Regression")
                st.write(f"Accuracy: {accuracy:.4f}")
                
                # Display confusion matrix
                st.write("Confusion Matrix for Logistic Regression:")
                st.pyplot(conf_matrix_figure)  # Display the confusion matrix heatmap

                st.download_button(
                    label="Download Logistic Regression Model (Pickle)",
                    data=download_pickle(lr_model, "logistic_regression_model.pkl"),
                    file_name="logistic_regression_model.pkl",
                    mime="application/octet-stream"
                )

            elif model_option == "Stacking Classifier":
                st.subheader("Training Stacking Classifier...")
                models = {
                    'knn': train_knn(X_train, y_train), 
                    'dt': train_dt(X_train, y_train), 
                    'rf': train_rf(X_train, y_train)
                }
                stacking_model = train_stacked_model(X_train, y_train, models)
                accuracy, conf_matrix_figure = evaluate_model(stacking_model, X_train, y_train, model_name="Stacking Classifier")
                
                # Display accuracy
                st.write(f"Accuracy: {accuracy:.4f}")
                
                # Display confusion matrix
                st.write("Confusion Matrix for Stacking Classifier:")
                st.pyplot(conf_matrix_figure)  # Display the confusion matrix heatmap

                st.download_button(
                    label="Download Stacking Classifier Model (Pickle)",
                    data=download_pickle(stacking_model, "stacking_classifier_model.pkl"),
                    file_name="stacking_classifier_model.pkl",
                    mime="application/octet-stream"
                )
                       
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
            train_df = pd.read_csv('train_data.csv')
            train_columns = train_df.drop(columns='Performance_Score').columns.tolist()
            # Preprocess the input data (encoding and scaling)
            input_df_encoded = preprocess_data(input_df)
            input_df_encoded = input_df_encoded.reindex(columns=train_columns, fill_value=0)
            
            # Align the columns of input_df_encoded with the training columns
            model = pickle.load(open('stacked_model_model.pkl', 'rb'))

            # Predict performance score using the loaded model
            if st.button("Predict Performance Score"):
                prediction = model.predict(input_df_encoded)
                st.write(f"Predicted Performance Score: {prediction[0]}")

            st.markdown('</div>', unsafe_allow_html=True)
    elif mode == "Multiple Employees":
        st.subheader("Predict Performance for Multiple Employees")
        
        uploaded_file = st.file_uploader("Upload Inference Data (CSV)", type="csv")
        
        train_df = pd.read_csv('train_data.csv')
        train_columns = train_df.columns.tolist()
        
        if uploaded_file is not None:
        # Load the uploaded CSV file
            inference_df = pd.read_csv(uploaded_file)
            inference_df = inference_df[train_columns]

            # Ensure the necessary columns are present in the uploaded file
            if 'Performance_Score' not in inference_df.columns:
                st.error("Uploaded file must contain 'Performance_Score' column.")
            else:
                # Drop 'Performance_Score' from features (X) and separate it (y)
                X_inference = inference_df.drop(columns=['Performance_Score'])
                y_true = inference_df['Performance_Score']

                # ---- KNN Classifier Evaluation ----
                st.subheader("KNN Classifier Model Performance:")
                #knn_model = pickle.load(open('knn_classifier_model.pkl', 'rb'))
                knn_model = pickle.load(open('k-nearest_neighbors_model.pkl', 'rb'))
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
                #dt_model = pickle.load(open('dt_classifier_model.pkl', 'rb'))
                dt_model = pickle.load(open('decision_tree_model.pkl', 'rb'))                
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
                #lr_model = pickle.load(open('lr_classifier_model.pkl', 'rb'))
                lr_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
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
                #rf_model = pickle.load(open('rf_classifier_model.pkl', 'rb'))
                rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
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
                #stacked_model = pickle.load(open('stacked_classifier_model.pkl', 'rb'))
                stacked_model = pickle.load(open('stacked_model_model.pkl', 'rb'))
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
            
