import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from Model_Development import download_csv

def train_test_split_():
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
