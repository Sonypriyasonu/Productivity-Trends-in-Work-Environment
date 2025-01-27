import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from Data_Preprocessing import preprocess_data

def predict_single_employee():
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

        # Load the pre-trained model
        model = pickle.load(open('stacked_model_model.pkl', 'rb'))

        # Predict performance score using the loaded model
        if st.button("Predict Performance Score"):
            prediction = model.predict(input_df_encoded)
            st.write(f"Predicted Performance Score: {prediction[0]}")

        st.markdown('</div>', unsafe_allow_html=True)

def predict_multiple_employees():
    st.subheader("Predict Performance for Multiple Employees")
    
    uploaded_file = st.file_uploader("Upload Inference Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        inference_df = pd.read_csv(uploaded_file)
        
        # Ensure the necessary columns are present
        train_df = pd.read_csv('train_data.csv')
        train_columns = train_df.columns.tolist()

        inference_df = inference_df[train_columns]

        if 'Performance_Score' not in inference_df.columns:
            st.error("Uploaded file must contain 'Performance_Score' column.")
        else:
            X_inference = inference_df.drop(columns=['Performance_Score'])
            y_true = inference_df['Performance_Score']

            # Load models and evaluate them
            models = {
                "KNN": "k-nearest_neighbors_model.pkl",
                "Decision Tree": "decision_tree_model.pkl",
                "Logistic Regression": "logistic_regression_model.pkl",
                "Random Forest": "random_forest_model.pkl",
                "Stacked Model": "stacked_model_model.pkl"
            }

            for model_name, model_file in models.items():
                model = pickle.load(open(model_file, 'rb'))
                predictions = model.predict(X_inference)
                accuracy = accuracy_score(y_true, predictions)
                conf_matrix = confusion_matrix(y_true, predictions)

                st.subheader(f"{model_name} Model Performance:")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                st.write("**Confusion Matrix:**")
                st.write(conf_matrix)

                # Add predictions to dataframe
                inference_df[f'{model_name}_Predicted_Performance_Score'] = predictions
                st.write(f"**First 10 {model_name} Predictions:**")
                st.write(inference_df[['Performance_Score', f'{model_name}_Predicted_Performance_Score']].head(10))

                # Download button for predicted performance scores
                model_csv = inference_df[[f'{model_name}_Predicted_Performance_Score']].to_csv(index=False)
                st.download_button(
                    label=f"Download {model_name} Predicted Scores",
                    data=model_csv,
                    file_name=f"{model_name}_predicted_performance_scores.csv",
                    mime="text/csv"
                )
    else:
        st.warning("Please upload a CSV file to proceed.")
