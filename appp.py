import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Career Path Analysis System",
    page_icon="🎓",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('career_model.joblib')
    scaler = joblib.load('career_scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return model, scaler, feature_names

def main():
    st.title("🎓 Career Path Analysis System")
    st.write("""
    This system helps predict the most suitable career path based on your profile.
    Please fill in your details below to get a career recommendation.
    """)
    
    try:
        model, scaler, feature_names = load_model()
    except:
        st.error("Please run generate_dataset.py and train_model.py first!")
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Academic & Technical Profile")
        cgpa = st.slider("CGPA (0-100)", 0, 100, 75)
        technical_skills = st.slider("Technical Skills (0-100)", 0, 100, 70)
        communication_skills = st.slider("Communication Skills (0-100)", 0, 100, 65)
        internships = st.slider("Number of Internships", 0, 5, 1)
        projects = st.slider("Number of Projects", 0, 10, 2)
        extracurricular_score = st.slider("Extracurricular Score (0-100)", 0, 100, 60)
    
    with col2:
        st.subheader("Personality & Interests")
        leadership = st.slider("Leadership Skills (0-100)", 0, 100, 60)
        creativity = st.slider("Creativity (0-100)", 0, 100, 65)
        analytical_thinking = st.slider("Analytical Thinking (0-100)", 0, 100, 70)
        research_interest = st.slider("Research Interest (0-100)", 0, 100, 60)
        business_interest = st.slider("Business Interest (0-100)", 0, 100, 55)
        technical_interest = st.slider("Technical Interest (0-100)", 0, 100, 70)
    
    # Create input array
    input_data = np.array([[
        cgpa, technical_skills, communication_skills,
        internships, projects, extracurricular_score,
        leadership, creativity, analytical_thinking,
        research_interest, business_interest, technical_interest
    ]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    if st.button("Analyze Career Path"):
        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_scaled)[0]
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Career Path Analysis Results")
        
        # Create columns for results
        col1, col2, col3 = st.columns(3)
        
        # Define colors for each career path
        colors = {
            'Higher Studies': '#FF9999',
            'Entrepreneurship': '#66B2FF',
            'Job Placement': '#99FF99'
        }
        
        # Display probability bars
        for i, (career, prob) in enumerate(zip(model.classes_, prediction_proba)):
            with [col1, col2, col3][i]:
                st.metric(
                    career,
                    f"{prob:.1%}",
                    delta=None,
                    delta_color="normal"
                )
                st.progress(prob)
        
        # Display recommendation
        st.markdown("---")
        st.success(f"Recommended Career Path: {prediction}")
        
        # Display reasoning
        st.subheader("Analysis Details")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(5), x='importance', y='feature')
        plt.title('Top 5 Most Important Features')
        st.pyplot(fig)
        
        # Display detailed analysis
        st.write("""
        ### Career Path Characteristics:
        
        **Higher Studies:**
        - Strong research interest
        - High academic performance
        - Strong analytical thinking
        
        **Entrepreneurship:**
        - High business interest
        - Strong leadership skills
        - Creative thinking
        
        **Job Placement:**
        - Strong technical skills
        - Good internship experience
        - Strong communication skills
        """)

if __name__ == "__main__":
    main() 