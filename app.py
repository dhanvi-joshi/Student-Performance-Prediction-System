import streamlit as st
import numpy as np
import joblib
import pandas as pd

# PAGE CONFIG
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>

.stApp {
    background-color: #f5f0e6;
    color: #3e2723;
}

/* TITLE */
h1 {
    color: #6d4c41;
    text-align: center;
    font-size: 42px !important;
    font-weight: bold;
}

/* SUBHEADINGS */
h2, h3 {
    color: #5d4037;
}

/* BUTTON */
div.stButton > button {
    background: linear-gradient(
        90deg,
        #bcaaa4,
        #8d6e63
    );
    color: white;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size: 20px;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(
        90deg,
        #a1887f,
        #795548
    );
}

/* SLIDERS */
.stSlider label {
    color: #4e342e !important;
    font-weight: 500;
}

/* SELECTBOX */
.stSelectbox label {
    color: #4e342e !important;
    font-weight: 500;
}

/* MAIN CONTAINERS */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #6d4c41;
    font-size: 15px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# LOAD MODEL AND SCALER
model = joblib.load('student_performance_model.pkl')
scaler = joblib.load('scaler.pkl')

# TITLE
st.markdown(
    "<h1>🎓 Student Performance Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown("""
<div style='text-align:center;
font-size:18px;
margin-bottom:30px;
color:#5d4037;'>

AI-powered academic performance prediction
using Machine Learning techniques.

</div>
""", unsafe_allow_html=True)

# SECTION
st.markdown("---")
st.subheader("📘 Academic Information")

# INPUTS

StudyHours = st.slider("Study Hours", 0, 12, 4)

Attendance = st.slider("Attendance (%)", 0, 100, 75)

Resources = st.slider("Learning Resources", 0, 10, 5)

extra_option = st.selectbox(
    "Extracurricular Participation",
    ["No", "Yes"]
)
Extracurricular = 1 if extra_option == "Yes" else 0

Motivation = st.slider("Motivation Level", 0, 10, 5)

internet_option = st.selectbox(
    "Internet Access",
    ["No", "Yes"]
)
Internet = 1 if internet_option == "Yes" else 0

OnlineCourses = st.slider(
    "Online Courses Completed",
    0,
    20,
    5
)

Discussions = st.slider(
    "Discussion Participation",
    0,
    20,
    5
)

AssignmentCompletion = st.slider(
    "Assignment Completion (%)",
    0,
    100,
    70
)

edutech_option = st.selectbox(
    "Use of Educational Technology",
    ["No", "Yes"]
)
EduTech = 1 if edutech_option == "Yes" else 0

StressLevel = st.slider(
    "Stress Level",
    0,
    10,
    5
)

# INPUT ARRAY
input_data = np.array([[
    StudyHours,
    Attendance,
    Resources,
    Extracurricular,
    Motivation,
    Internet,
    OnlineCourses,
    Discussions,
    AssignmentCompletion,
    EduTech,
    StressLevel
]])

# SCALE INPUT
input_scaled = scaler.transform(input_data)

# PREDICTION
if st.button("🚀 Predict Performance"):

    # HYBRID RULE + ML SYSTEM

    if (
        StudyHours >= 10 and
        Attendance >= 90 and
        AssignmentCompletion >= 90 and
        Motivation >= 8 and
        StressLevel <= 3
    ):
        result = "Excellent"

    elif (
        StudyHours <= 2 and
        Attendance <= 40 and
        AssignmentCompletion <= 40 and
        Motivation <= 3
    ):
        result = "Poor"

    else:
        prediction = model.predict(input_scaled)[0]

        grade_map = {
            0: "Excellent",
            1: "Good",
            2: "Average",
            3: "Poor"
        }

        result = grade_map[prediction]

    # RESULT COLORS

    result_colors = {
        "Excellent": "#2e7d32",
        "Good": "#558b2f",
        "Average": "#ef6c00",
        "Poor": "#c62828"
    }

    result_color = result_colors[result]

    # RESULT BOX

    st.markdown(f"""
    <div style="
    padding:25px;
    border-radius:15px;
    background-color:{result_color};
    color:white;
    font-size:30px;
    font-weight:bold;
    text-align:center;
    margin-top:20px;
    margin-bottom:20px;
    ">
    🎯 Predicted Performance: {result}
    </div>
    """, unsafe_allow_html=True)

    # RECOMMENDATIONS

    st.subheader("📌 Recommendations")

    recommendations = []

    if Attendance < 75:
        recommendations.append(
            "Improve attendance consistency."
        )

    if StudyHours < 3:
        recommendations.append(
            "Increase study hours regularly."
        )

    if StressLevel > 7:
        recommendations.append(
            "Practice stress management techniques."
        )

    if AssignmentCompletion < 70:
        recommendations.append(
            "Complete assignments on time."
        )

    if Motivation < 5:
        recommendations.append(
            "Engage in motivational learning activities."
        )

    if Discussions < 5:
        recommendations.append(
            "Participate more actively in discussions."
        )

    if OnlineCourses < 3:
        recommendations.append(
            "Consider taking more online learning courses."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Excellent work. Maintain consistency and continue your good habits."
        )

    # DISPLAY RECOMMENDATIONS

    for rec in recommendations:
        st.markdown(f"""
        <div style="
        background-color:#fffaf3;
        padding:12px;
        border-radius:10px;
        margin-bottom:10px;
        border-left:6px solid #8d6e63;
        color:#4e342e;
        ">
        ✅ {rec}
        </div>
        """, unsafe_allow_html=True)

# FEATURE IMPORTANCE

st.markdown("---")

feature_importance = pd.DataFrame({
    'Feature': [
        'StudyHours',
        'Attendance',
        'Resources',
        'Extracurricular',
        'Motivation',
        'Internet',
        'OnlineCourses',
        'Discussions',
        'AssignmentCompletion',
        'EduTech',
        'StressLevel'
    ],
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

st.subheader("📊 Feature Importance")

st.bar_chart(
    feature_importance.set_index('Feature')
)

# PROJECT INSIGHTS

st.markdown("---")

st.subheader("🧠 Project Insights")

st.markdown("""
- Assignment completion and attendance were among the strongest predictors.
- Decision Tree achieved the best performance after optimization.
- Feature selection improved model generalization.
- Streamlit deployment enabled real-time prediction capability.
- Hybrid AI logic was used for handling extreme academic profiles.
""")

# FOOTER

st.markdown("""
<div class='footer'>

Developed using Streamlit, Scikit-learn and Machine Learning 🚀

</div>
""", unsafe_allow_html=True)