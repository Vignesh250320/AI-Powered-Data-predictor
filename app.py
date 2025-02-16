import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Streamlit app title
st.title("Logistic Regression Prediction with CSV/Excel Upload")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Process the file
if uploaded_file is not None:
    # Check if the uploaded file is Excel or CSV
    if uploaded_file.name.endswith('.xlsx'):
        # Read Excel file
        df = pd.read_excel(uploaded_file)
    else:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
    
    # Display the dataset preview
    st.write("Dataset preview:", df.head())

    # Basic data exploration
    if st.checkbox('Show column names'):
        st.write(df.columns)
    
    if st.checkbox('Show dataset shape'):
        st.write(df.shape)
    
    # Select target column
    target_column = st.selectbox("Select the target column (y)", df.columns)

    # Select features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
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
