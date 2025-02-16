import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Streamlit app title
st.title("Logistic Regression Prediction with CSV/Excel Upload")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Process the file
if uploaded_file is not None:
    try:
        # Check if the uploaded file is Excel or CSV
        if uploaded_file.name.endswith('.xlsx'):
            # Read Excel file
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            # Read CSV file with specific encoding
            df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try ISO-8859-1 encoding
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # Display the dataset preview
    st.write("Dataset preview:", df.head(10))

    # Check for missing values and invalid data types
    st.write("Missing values in the dataset:")
    st.write(df.isnull().sum())
    
    st.write("Data types in the dataset:")
    st.write(df.dtypes)

    # Select target column
    target_column = st.selectbox("Select the target column (y)", df.columns)

    # Select features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Ensure no missing values in features
    if X.isnull().sum().any():
        st.write("Missing values found in feature columns. Handling missing values...")
        X = X.fillna(0)  # Example: fill missing values with 0

    # Handle categorical features by encoding them
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # Ensure target column is numeric
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # If the user wants to make a prediction
    if st.button("Make Prediction"):
        # User input for prediction
        input_data = []
        for col in X.columns:
            value = st.number_input(f"Enter value for {col}")
            input_data.append(value)
        
        # Make prediction
        prediction = model.predict([input_data])
        st.write(f"Prediction: {prediction[0]}")
