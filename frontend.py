import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page config
st.set_page_config(page_title="🎓 Career Path Forecasting & Analysis App", layout="wide")

# Load shared models and encoders
@st.cache_resource
def load_models():
    placement_model = joblib.load("student_placement_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    career_model = joblib.load("career_model.joblib")
    career_scaler = joblib.load("career_scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    return placement_model, label_encoder, career_model, career_scaler, feature_names

placement_model, label_encoder, career_model, career_scaler, feature_names = load_models()

# Tabs for separation
tab1, tab2 = st.tabs(["📁 Career Path Forecasting", "🧠Personalized Career Path Recomendation"])

# --- TAB 1: PLACEMENT OUTCOME PREDICTION ---
with tab1:
    st.title("🎓 Career Path Forecasting")
    st.write("📁 Upload your Excel file to predict student post-graduation outcomes.")

    expected_columns = ['cgpa', 'internship', 'research papers', 'enterpreunership experience',
                        'skills for stratup', 'programming languages', 'coding']

    uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx", "xls"], key="placement")

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.subheader("📄 Uploaded Data Preview:")
            st.dataframe(df)

            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns in uploaded file: {', '.join(missing_cols)}")
                st.stop()

            input_df = df[expected_columns].copy()
            binary_map = {"yes": 1, "no": 0}
            for col in ['internship', 'research papers', 'enterpreunership experience', 'skills for stratup']:
                if col in input_df.columns:
                    input_df[col] = input_df[col].map(binary_map)

            predictions = placement_model.predict(input_df)
            decoded_predictions = label_encoder.inverse_transform(predictions)
            df['Prediction'] = decoded_predictions

            st.subheader("📈 Prediction Results:")
            st.dataframe(df)

            category_counts = df['Prediction'].value_counts().reset_index()
            category_counts.columns = ['Outcome', 'Count']
            fig = px.bar(category_counts, x='Outcome', y='Count', color='Outcome', text='Count',
                         title="📊 Distribution of Predicted Outcomes")
            st.plotly_chart(fig)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠ Error processing the file or predicting: {e}")

    st.markdown("---")
    st.header("📊 API Score Calculator")

    with st.form("api_calc_form"):
        cgpa = st.number_input("📚 CGPA (out of 10)", min_value=0.0, max_value=10.0, step=0.01)
        paid_internships = st.number_input("💸 Paid Internships", 0, 5)
        unpaid_internships = st.number_input("🧳 Unpaid Internships", 0, 5)

        st.markdown("### 🎓 Courses Completed")
        iit = st.number_input("🏫 IIT Courses", 0, 10)
        nit = st.number_input("🏛 NIT Courses", 0, 10)
        industry = st.number_input("🏭 Industry Courses", 0, 10)
        other = st.number_input("📘 Other Courses", 0, 10)

        certificates = st.number_input("🏅 Co-curricular Certificates", 0, 10)

        submit_api = st.form_submit_button("✅ Calculate API Score")

        if submit_api:
            cgpa_points = min((cgpa / 10) * 3, 3)
            internship_points = min(paid_internships * 2 + unpaid_internships, 4)
            course_points = min(iit * 0.5 + nit * 0.4 + industry * 0.3 + other * 0.2, 2)
            cert_points = min(certificates * 0.5, 1)
            total_score = round(cgpa_points + internship_points + course_points + cert_points, 2)

            st.success(f"🎯 API Score: *{total_score} / 10*")
            if total_score >= 8.5:
                st.info("🌟 Excellent Profile! You're well-prepared for placements or higher studies.")
            elif total_score >= 7:
                st.info("👍 Good Job! Keep boosting your experience and skillset.")
            elif total_score >= 5:
                st.info("📈 Fair. Focus on enhancing your profile with internships or courses.")
            else:
                st.warning("🚧 Needs Improvement. Engage more in academics and co-curriculars.")

# --- TAB 2: CAREER PATH ANALYSIS ---
with tab2:
    st.title("🧠 Personalized Career Path Recomendation")
    st.write("Analyze your skills, interests, and academic profile to predict the best career path.")

    with tab2:
     with st.expander("📘 View Input Guidelines"):
        with open("Career_Input_Guidelines.pdf", "rb") as f:
            st.download_button(
                label="📥 Download Career Input Guidelines (PDF)",
                data=f,
                file_name="Career_Input_Guidelines.pdf",
                mime="application/pdf"
            )


    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA (0-100)", 0, 100, 75)
        tech = st.slider("Technical Skills", 0, 100, 70)
        comm = st.slider("Communication Skills", 0, 100, 65)
        internships = st.slider("Number of Internships", 0, 5, 1)
        projects = st.slider("Number of Projects", 0, 10, 2)
        extra = st.slider("Extracurricular Score", 0, 100, 60)

    with col2:
        leader = st.slider("Leadership Skills", 0, 100, 60)
        creativity = st.slider("Creativity", 0, 100, 65)
        analytics = st.slider("Analytical Thinking", 0, 100, 70)
        research = st.slider("Research Interest", 0, 100, 60)
        business = st.slider("Business Interest", 0, 100, 55)
        tech_interest = st.slider("Technical Interest", 0, 100, 70)

    user_input = np.array([[cgpa, tech, comm, internships, projects, extra,
                            leader, creativity, analytics, research, business, tech_interest]])
    input_scaled = career_scaler.transform(user_input)

    if st.button("Analyze Career Path"):
        probas = career_model.predict_proba(input_scaled)[0]
        prediction = career_model.predict(input_scaled)[0]

        st.markdown("---")
        st.subheader("📊 Career Prediction Probabilities")

        for label, proba in zip(career_model.classes_, probas):
            st.metric(label, f"{proba:.1%}")
            st.progress(proba)

        st.success(f"✅ Recommended Career Path: *{prediction}*")

        st.subheader("🔍 Analysis Insights")
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': career_model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        user_feature_values = user_input.flatten()
        feature_influence = pd.DataFrame({
            'feature': feature_names,
            'user_value': user_feature_values,
            'importance': career_model.feature_importances_,
            'weighted_score': user_feature_values * career_model.feature_importances_
        })

        top_features = feature_influence.sort_values(by='weighted_score', ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_features, x='weighted_score', y='feature', palette='viridis')
        plt.title("Top 5 User-Specific Influential Features")
        st.pyplot(fig)

        st.markdown("""
        ### Career Characteristics:
        - **Higher Studies**: Research interest, high CGPA, analytical strength
        - **Entrepreneurship**: Business & leadership skills, creativity
        - **Job Placement**: Strong technical & communication skills, internships
        """)

        st.markdown("### 📌 Feature Descriptions and Recommendations")
        for _, row in top_features.iterrows():
            st.markdown(f"**{row['feature']}**: Value = {row['user_value']}, Importance = {row['importance']:.2f}")
            if row['feature'].lower() in ['research', 'cgpa', 'analytics']:
                st.write("→ Suggestion: Continue developing your academic and analytical strengths.")
            elif row['feature'].lower() in ['business', 'leadership', 'creativity']:
                st.write("→ Suggestion: Consider entrepreneurial roles or business-oriented studies.")
            elif row['feature'].lower() in ['tech', 'communication', 'internships']:
                st.write("→ Suggestion: Focus on technical certifications and real-world experience.")

        st.download_button("⬇ Download Career Insights as CSV", data=feature_influence.to_csv(index=False).encode(), file_name="career_insights.csv", mime="text/csv")
