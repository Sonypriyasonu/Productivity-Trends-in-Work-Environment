import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to visualize "Performance Score vs Department"
def performance_score_vs_department(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Department', y='Performance_Score', hue='Department', data=df, palette='Set2')
    plt.title('Performance Score vs Department')
    plt.xticks(rotation=45)
    return plt

# Function to visualize "Performance Score vs Gender"
def performance_score_vs_gender(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Gender', y='Performance_Score', hue='Gender', data=df, palette='Set2')
    plt.title('Performance Score vs Gender')
    return plt

# Function to visualize "Performance Score vs Education Level"
def performance_score_vs_education_level(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Education_Level', y='Performance_Score', hue='Education_Level', data=df, palette='Set2')
    plt.title('Performance Score vs Education Level')
    return plt

# Function to visualize "Employee Satisfaction vs Performance Score"
def employee_satisfaction_vs_performance(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Employee_Satisfaction_Score', y='Performance_Score', data=df, hue='Department', palette='Set2')
    plt.title("Employee Satisfaction vs Performance Score")
    plt.xlabel("Employee Satisfaction Score")
    plt.ylabel("Performance Score")
    plt.legend(title='Department')
    return plt

# Function to visualize "Years at Company vs Monthly Salary"
def years_at_company_vs_salary(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Years_At_Company', y='Monthly_Salary', data=df, color='red')
    plt.title("Years at Company vs Monthly Salary")
    plt.xlabel("Years at Company")
    plt.ylabel("Monthly Salary")
    return plt

# Function to visualize "Correlation Matrix"
def correlation_matrix(df):
    num_col = ['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 
                    'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 
                    'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score']
            
    for col in num_col:
        df[col] = df[col].fillna(df[col].mean())  # Fill missing values with mean
            
    # Compute the correlation matrix for numeric columns
    df_num = df.select_dtypes(include=['number'])
    corr_matrix = df_num.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    plt.title('Correlation Matrix of Numeric Features')
    return plt

def number_of_employees_by_education_level(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Education_Level', hue='Education_Level', palette='Set2')
    plt.title('Number of Employees by Education Level')
    plt.xticks(rotation=45)
    plt.xlabel("Education Level")
    plt.ylabel("Number of Employees")
    return plt

def data_visualization():
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
            
