from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to download CSV
def download_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    return csv

# Function to save pickle files
def download_pickle(model, filename):
    pickle_bytes = pickle.dumps(model)
    return pickle_bytes

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=15, metric='manhattan')
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_dt(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lr(X_train, y_train):
    model = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_stacked_model(X_train, y_train, models):
    meta_model = LogisticRegression(random_state=42)
    stacked_model = StackingClassifier(
        estimators=[('knn', models['knn']),
                    ('dt', models['dt']),
                    ('rf', models['rf'])],
        final_estimator=meta_model
    )
    stacked_model.fit(X_train, y_train)
    return stacked_model

def evaluate_model(model, X_train, y_train, model_name):
    # Make predictions using the model
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    conf_matrix = confusion_matrix(y_train, y_pred)

    # Create a confusion matrix heatmap
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")

    return accuracy, fig

# Streamlit interface for model development
def model_development():
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
