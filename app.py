import os
import joblib

current_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(current_dir, 'student_placement_model.pkl'))
le = joblib.load(os.path.join(current_dir, 'label_encoder.pkl'))


current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'student_placement_model.pkl')

model = joblib.load(model_path)
# Load the model and encoder
import joblib
model = joblib.load(r'C:\Users\sambh\OneDrive\Desktop\pred_model\student_placement_model.pkl')
le = joblib.load('label_encoder.pkl')

# Example input data (adjust values based on your features)
new_student_data = [[
    8.5,   # cgpa
    1,      # internship
    2,      # research papers
    3,      # entrepreneurship experience
    4,      # skills for startup
    5,      # programming languages
    3       # coding
]]

# Column names used during training (from your previous code)
feature_columns = [
    'cgpa', 'internship', 'research papers',
    'enterpreunership experience', 'skills for stratup',
    'programming languages', 'coding'
]

# Create a DataFrame with the new student data and column names
new_student = pd.DataFrame(new_student_data, columns=feature_columns)

# Predict
prediction = model.predict(new_student)
decoded_prediction = le.inverse_transform(prediction)
print("Predicted Result:", decoded_prediction[0])