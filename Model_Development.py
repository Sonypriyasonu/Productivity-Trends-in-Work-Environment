from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

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
