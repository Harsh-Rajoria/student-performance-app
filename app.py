import streamlit as st
import numpy as np
import pandas as pd
import requests
import zipfile
import io
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance Predictor", page_icon="📊", layout="wide")

genai.configure(api_key="AIzaSyCr0eILVz1l5W5Jdfo7VnsoGO7qcO8-fX0")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content), 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        df = pd.read_csv(z.open(csv_files[0]), sep=';') if len(csv_files) == 1 else pd.read_csv(z.open("student-mat.csv"), sep=';')
    df = df[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G3']]
    df.dropna(inplace=True)
    return df

df = load_data()
X = df.drop(columns=['G3'])
y = df['G3']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

def generate_insights_gemini(student_data):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    prompt = f"Predict the student's performance based on this data: {student_data}. Provide risk factors and study recommendations."
    response = model.generate_content(prompt)
    return response.text if response else "⚠️ No response from Gemini API"

def predict_performance(input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    insights = generate_insights_gemini(input_data)
    return prediction, insights

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Dashboard", "🎯 Predictions", "📂 Upload & Analysis", "📝 Interventions"])

if page == "🏠 Dashboard":
    st.title("📊 Student Performance Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Final Grade Distribution")
        st.bar_chart(df['G3'])
    with col2:
        st.subheader("📉 Study Time vs Final Grade")
        st.scatter_chart(df[['studytime', 'G3']])
    st.info("💡 Explore key student performance trends based on real-world data!")

elif page == "🎯 Predictions":
    st.title("🎯 Predict Student Performance")
    st.write("🔢 Enter student details to predict their final grade.")
    col1, col2 = st.columns(2)
    input_data = []
    with col1:
        for i in range(len(X.columns) // 2):
            input_data.append(st.number_input(f"{X.columns[i]}", value=0))
    with col2:
        for i in range(len(X.columns) // 2, len(X.columns)):
            input_data.append(st.number_input(f"{X.columns[i]}", value=0))
    if st.button("🔮 Predict Now"):
        prediction, insights = predict_performance(input_data)
        st.success(f"📊 **Predicted Final Grade: {round(prediction, 2)}**")
        st.info(f"💡 **AI Insights:** {insights}")

elif page == "📂 Upload & Analysis":
    st.title("📂 Upload Student Data")
    uploaded_file = st.file_uploader("Upload a CSV or ZIP file", type=["csv", "zip"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                df_uploaded = pd.read_csv(z.open(csv_files[0])) if len(csv_files) == 1 else pd.read_csv(z.open("student-mat.csv"))
        else:
            df_uploaded = pd.read_csv(uploaded_file)
        st.write("📊 **Uploaded Data Preview:**", df_uploaded.head())
        st.bar_chart(df_uploaded.select_dtypes(include=[np.number]))

elif page == "📝 Interventions":
    st.title("📝 Personalized Interventions")
    st.write("💡 **Enter student details to get AI-generated intervention strategies.**")
    student_profile = st.text_area("✏️ Enter student background details:")
    if st.button("✨ Generate Plan"):
        intervention_plan = generate_insights_gemini(student_profile)
        st.success("📄 **Intervention Plan Generated!**")
        st.write(intervention_plan)
