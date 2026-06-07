import os
import streamlit as st
from dotenv import load_dotenv
from file_utils import extract_file_content

load_dotenv()
st.set_page_config(page_title="Medical Assistant - Input", page_icon="🩺")

# ✅ Validate BOTH keys
groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
if not groq_key:
    st.error("❌ GROQ_API_KEY not found in .env")
    st.stop()
if not gemini_key:
    st.warning("⚠️ GEMINI_API_KEY not found. Image analysis will be unavailable.")

st.title("🩺 Medical Symptom Analyzer")
st.markdown("Provide your details below. Upload reports or images (PDF, DOCX, TXT, PNG, JPG).")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        pain_level = st.selectbox("Pain Level", ["Low", "Medium", "High"])
    with col2:
        days_suffering = st.number_input("Days Suffering", min_value=0, max_value=365, step=1)
        symptoms = st.text_area("Describe Your Symptoms", placeholder="e.g., persistent headache...", height=120)

    # ✅ ALL formats including images
    uploaded_file = st.file_uploader(
        "Upload Medical Report/Image (Optional)",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        help="Text docs processed by Llama (fast). Images analyzed by Gemini."
    )

    submitted = st.form_submit_button("Start Consultation", type="primary", use_container_width=True)

if submitted:
    if not symptoms.strip():
        st.warning("⚠️ Please describe your symptoms before continuing.")
    else:
        extracted = extract_file_content(uploaded_file) if uploaded_file else {
            "text": "", "image_bytes": None, "mime_type": "", "is_image": False
        }

        st.session_state.patient_data = {
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "days_suffering": days_suffering,
            "pain_level": pain_level,
            "report_text": extracted["text"],
            "report_image_bytes": extracted["image_bytes"],
            "report_mime_type": extracted["mime_type"],
            "has_initial_image": extracted["is_image"],
        }

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.switch_page("pages/chat.py")