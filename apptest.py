# Import libraries
import joblib
import pandas as pd

# Load the trained model and label encoder
model = joblib.load('student_placement_model.pkl')
le = joblib.load('label_encoder.pkl')

# Load the Excel file
df = pd.read_excel('datayesno_modified.xlsx')   # <-- Put your correct Excel filename here

# Drop rows with missing values (optional but recommended)
df = df.dropna()

# Define the feature columns (must match training)
feature_columns = [
    'cgpa', 'internship', 'research papers',
    'enterpreunership experience', 'skills for stratup',
    'programming languages', 'coding'
]

# Select the features (X)
X = df[feature_columns]

# Predict for all students
predictions = model.predict(X)

# Decode the predictions (convert numbers back to original labels)
decoded_predictions = le.inverse_transform(predictions)

# Add predictions to the DataFrame
df['Predicted Result'] = decoded_predictions

# Show the DataFrame with predictions
print(df)

new_student_data = {
    'usn': [201],
    'name': ['Alice'],
    'cgpa': [8.5],
    'post graduation plans': ['Startup'],
    'Unnamed: 4': ['arts'],
    # Add other feature columns here as per your data
}


# (Optional) Save the predictions to a new Excel file
df.to_excel('student_predictions2.xlsx', index=False)
print("Predictions saved to 'student_predictions2.xlsx' ✅")

# Create a DataFrame with the new student data and column names
new_student = pd.DataFrame(new_student_data, columns=feature_columns)

# Predict
prediction = model.predict(new_student)
decoded_prediction = le.inverse_transform(prediction)
print("Predicted Result:", decoded_prediction[0])