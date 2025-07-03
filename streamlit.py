# NOTE: This code requires the 'streamlit' package. Install it with:
# pip install streamlit

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'streamlit' package is not installed. Please install it using 'pip install streamlit' and run this script in a local environment.")

import pandas as pd
import pickle

# Load the trained model and label encoder
with open('student_placement_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# App Title
st.title("Student Placement Prediction App")
st.write("Upload your Excel file below to get predictions.")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        st.subheader("Uploaded Data Preview:")
        st.dataframe(df)

        # Label encoding if needed (assuming one categorical column needs encoding)
        if 'gender' in df.columns:
            df['gender'] = le.transform(df['gender'])

        # Make predictions
        predictions = model.predict(df)

        # Display predictions
        st.subheader("Predictions:")
        df['Prediction'] = predictions
        st.dataframe(df)

        # Downloadable CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
