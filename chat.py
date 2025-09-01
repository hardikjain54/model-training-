import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import io
import chardet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ----------------- Preprocessing -----------------
def preprocess_data(X, y, problem):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)

    # encode categorical target for classification
    if problem == "Classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    return X_processed, y


# ----------------- Training -----------------
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


# ----------------- Streamlit App -----------------
def main():
    st.title("⚡ Machine Learning Application")
    st.write("Upload a dataset or use example data, select features and target, then train a model.")
    
    # Data source
    data_source = st.sidebar.selectbox("Do you want to upload data or use example data?", ["Upload", "Example"])
    data = None

    if data_source == "Upload":
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx', 'tsv'])
        if uploaded_file is not None:
            file_name = uploaded_file.name.lower()

            try:
                if file_name.endswith(".csv") or file_name.endswith(".tsv"):
                    raw_data = uploaded_file.getvalue()
                    detected = chardet.detect(raw_data)
                    encoding = detected["encoding"]
                    sep = "," if file_name.endswith(".csv") else "\t"
                    data = pd.read_csv(io.BytesIO(raw_data), encoding=encoding, sep=sep)

                elif file_name.endswith(".xlsx"):
                    data = pd.read_excel(uploaded_file, engine="openpyxl")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    else:
        dataset_name = st.sidebar.selectbox("Select a dataset", ["titanic", "tips", "iris"])
        data = sns.load_dataset(dataset_name)
    
    # Main app logic
    if data is not None:
        st.write("### 📊 Data Preview")
        st.write(data.head())
        st.write("Shape:", data.shape)

        # Feature/target selection
        features = st.multiselect("Select feature columns", data.columns.tolist())
        target = st.selectbox("Select target column", data.columns.tolist())
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])
        
        if features and target:
            X = data[features]
            y = data[target]

            st.write(f"You have selected a **{problem_type}** problem.")
            run_analysis = st.checkbox("🚀 Run Analysis")
            
            if run_analysis:
                # Preprocess
                X_processed, y_processed = preprocess_data(X, y, problem_type)

                # Split
                test_size = st.slider("Test split size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42
                )

                # Model selection
                model_options = (
                    ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM']
                    if problem_type == 'Regression'
                    else ['Decision Tree', 'Random Forest', 'SVM']
                )
                selected_model = st.sidebar.selectbox("Select model", model_options)

                if selected_model == 'Linear Regression':
                    model = LinearRegression()
                elif selected_model == 'Decision Tree':
                    model = DecisionTreeRegressor() if problem_type == 'Regression' else DecisionTreeClassifier()
                elif selected_model == 'Random Forest':
                    model = RandomForestRegressor() if problem_type == 'Regression' else RandomForestClassifier()
                elif selected_model == 'SVM':
                    model = SVR() if problem_type == 'Regression' else SVC()

                # Train + predict
                predictions = train_and_evaluate(X_train, X_test, y_train, y_test, model)
                st.success("✅ Model training complete.")

                # Metrics
                if problem_type == "Regression":
                    st.subheader("📈 Regression Metrics")
                    st.write("MAE:", mean_absolute_error(y_test, predictions))
                    st.write("MSE:", mean_squared_error(y_test, predictions))
                    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
                    st.write("R² Score:", r2_score(y_test, predictions))
                else:
                    st.subheader("📊 Classification Metrics")
                    st.write("Accuracy:", accuracy_score(y_test, predictions))
                    st.write("Precision:", precision_score(y_test, predictions, average='weighted', zero_division=0))
                    st.write("Recall:", recall_score(y_test, predictions, average='weighted', zero_division=0))
                    st.write("F1 Score:", f1_score(y_test, predictions, average='weighted', zero_division=0))
                    st.write("Confusion Matrix:")
                    st.write(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
