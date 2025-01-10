import matplotlib.pyplot as plt
import seaborn as sns

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
