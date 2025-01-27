import streamlit as st
import pandas as pd
from Predict_Performance import predict_single_employee,predict_multiple_employees
from Data_Visualization import data_visualization
from Model_Development import model_development
from Train_Test_Split import train_test_split_
from Data_Preprocessing import data_preprocessing

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
    data_preprocessing()        
# Data Visualization Page
if page == "Data Visualization":
    data_visualization()
if page == "Train Test Split Data":
    train_test_split_()
if page=="Model Development":
    model_development()
                       
if page == "Predict Performance":
    st.title("Predict Employee Performance")
    mode = st.radio("Select Mode", ["Single Employee", "Multiple Employees"])

    if mode == "Single Employee":
        predict_single_employee()
    elif mode == "Multiple Employees":
        predict_multiple_employees()
            
