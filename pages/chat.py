import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from groq import RateLimitError as GroqRateLimitError
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from file_utils import extract_file_content

load_dotenv()
st.set_page_config(page_title="Medical Assistant - Chat", page_icon="💬")

if "patient_data" not in st.session_state:
    st.error("❌ No patient data found. Please return to the input page.")
    if st.button("Go to Input Page"):
        st.switch_page("app.py")
    st.stop()

patient_data = st.session_state.patient_data

# ============================================================
# SYSTEM PROMPTS (shared structure, provider-specific notes)
# ============================================================
BASE_SYSTEM_PROMPT = """You are a professional AI medical assistant.

PATIENT CONTEXT:
- Age: {age}
- Gender: {gender}
- Initial Symptoms: {symptoms}
- Duration: {days_suffering} days
- Pain Level: {pain_level}
{report_section}

GUIDELINES:
1. ALWAYS start your first response with a disclaimer that you are an AI, NOT a substitute for professional medical advice.
2. Use ALL provided context including uploaded documents and images.
3. For images (X-rays, lab results, charts), describe relevant findings carefully and note limitations.
4. Never prescribe specific medications or dosages.
5. If pain is HIGH or emergency signs present, advise immediate medical attention.
6. Keep responses concise, empathetic, and professional.
"""

INITIAL_PROMPT = (
    "Based on all patient context (including any attached report/image), "
    "provide an initial health consultation. Do not ask user to repeat information."
)

# ============================================================
# PROVIDER INITIALIZATION
# ============================================================
@st.cache_resource
def get_groq_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("❌ GROQ_API_KEY missing"); st.stop()
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=key)

@st.cache_resource
def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None  # Graceful degradation: no image support
    return genai.Client(api_key=key)

groq_llm = get_groq_llm()
gemini_client = get_gemini_client()

# ============================================================
# GROQ CHAIN (text-only path)
# ============================================================
report_text = patient_data.get("report_text", "")
report_section = f"\nMEDICAL REPORT EXTRACT:\n{report_text}" if report_text.strip() else ""

groq_prompt = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
]).partial(
    age=patient_data["age"], gender=patient_data["gender"],
    symptoms=patient_data["symptoms"], days_suffering=patient_data["days_suffering"],
    pain_level=patient_data["pain_level"], report_section=report_section
)
groq_chain = groq_prompt | groq_llm | StrOutputParser()

# ============================================================
# GEMINI HELPERS (multimodal path)
# ============================================================
GEMINI_SYSTEM_INSTRUCTION = BASE_SYSTEM_PROMPT.format(
    age=patient_data["age"], gender=patient_data["gender"],
    symptoms=patient_data["symptoms"], days_suffering=patient_data["days_suffering"],
    pain_level=patient_data["pain_level"], report_section=report_section
)

def build_gemini_contents(user_input, image_bytes=None, mime_type=None, include_initial_image=False):
    contents = []
    if include_initial_image and patient_data.get("report_image_bytes"):
        contents.append(types.Part.from_bytes(
            data=patient_data["report_image_bytes"],
            mime_type=patient_data["report_mime_type"]
        ))
    for msg in get_chat_history():
        role = "user" if isinstance(msg, HumanMessage) else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
    parts = [types.Part.from_text(text=user_input)]
    if image_bytes and mime_type:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    contents.append(types.Content(role="user", parts=parts))
    return contents

def call_gemini(user_input, image_bytes=None, mime_type=None, include_initial_image=False):
    if gemini_client is None:
        return ("⚠️ Image analysis requires a Gemini API key. "
                "Please add GEMINI_API_KEY to your .env file, or describe the image content in text.")
    contents = build_gemini_contents(user_input, image_bytes, mime_type, include_initial_image)
    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=GEMINI_SYSTEM_INSTRUCTION, temperature=0.3
        )
    )
    return resp.text

# ============================================================
# UNIFIED ROUTER + RETRY
# ============================================================
def generate_hybrid(user_input, image_bytes=None, mime_type=None,
                    include_initial_image=False, max_retries=3):
    """Route to Gemini if images present, otherwise Groq. Retry on rate limits."""
    use_gemini = bool(image_bytes) or include_initial_image

    for attempt in range(max_retries):
        try:
            if use_gemini:
                return call_gemini(user_input, image_bytes, mime_type, include_initial_image)
            else:
                return groq_chain.invoke({
                    "input": user_input,
                    "history": get_chat_history()[:-1] if not include_initial_image else []
                })
        except (google_exceptions.ResourceExhausted, GroqRateLimitError):
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                provider = "Gemini" if use_gemini else "Groq"
                st.warning(f"⏳ {provider} rate limited. Retrying in {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        except Exception:
            raise

# ============================================================
# CHAT HISTORY
# ============================================================
def get_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history

def add_to_history(role: str, content: str):
    msg = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
    st.session_state.chat_history.append(msg)

# ============================================================
# AUTO-GENERATE INITIAL CONSULTATION
# ============================================================
if "initial_consultation_done" not in st.session_state:
    has_img = bool(patient_data.get("has_initial_image"))
    provider_label = "Gemini (image)" if has_img else "Llama 3.3 70B"
    with st.spinner(f"Generating initial consultation via {provider_label}..."):
        try:
            initial_response = generate_hybrid(
                INITIAL_PROMPT, include_initial_image=has_img
            )
            add_to_history("assistant", initial_response)
            st.session_state.initial_consultation_done = True
        except (google_exceptions.ResourceExhausted, GroqRateLimitError):
            st.error("❌ API quota exhausted. Please wait ~1 minute and refresh.")
        except Exception as e:
            st.error(f"Error: {e}")
    if st.session_state.get("initial_consultation_done"):
        st.rerun()

# ============================================================
# UI
# ============================================================
st.title("💬 Medical Consultation Chat")
has_report = bool(report_text.strip()) or bool(patient_data.get("report_image_bytes"))
st.caption(
    f"**Patient:** {patient_data['age']}y/o {patient_data['gender']} | "
    f"**Pain:** {patient_data['pain_level']} | "
    f"**Duration:** {patient_data['days_suffering']}d"
    + (" | 📎 Report/Image attached" if has_report else "")
    + " | 🦙 Llama + 🔷 Gemini Hybrid"
)

for msg in get_chat_history():
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Unified chat bar (Streamlit >= 1.42)
chat_response = st.chat_input(
    "Ask a follow-up question or attach a document/image...",
    accept_file=True
)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "png", "jpg", "jpeg"}

if chat_response:
    prompt_text = chat_response.text
    uploaded_files = chat_response.files or []

    valid_files, rejected_files = [], []
    for f in uploaded_files:
        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        (valid_files if ext in ALLOWED_EXTENSIONS else rejected_files).append(f.name if f in rejected_files else f)

    if rejected_names := [f.name for f in uploaded_files
                          if not (f.name.rsplit(".", 1)[-1].lower() if "." in f.name else "") in ALLOWED_EXTENSIONS]:
        st.warning(f"⚠️ Skipped unsupported: {', '.join(rejected_names)}")

    combined_context, img_bytes, img_mime = "", None, None
    for f in valid_files:
        extracted = extract_file_content(f)
        if extracted["is_image"]:
            img_bytes, img_mime = extracted["image_bytes"], extracted["mime_type"]
        if extracted["text"] and not extracted["text"].startswith("[Unsupported"):
            combined_context += f"\n\n[{f.name}]:\n{extracted['text']}"

    full_input = prompt_text + combined_context

    st.chat_message("user").write(prompt_text)
    if valid_files:
        labels = [f"{f.name}{'🖼️' if extract_file_content(f)['is_image'] else ''}" for f in valid_files]
        st.caption(f"📎 {', '.join(labels)}")
    add_to_history("user", full_input)

    route = "🔷 Gemini" if img_bytes else "🦙 Llama"
    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing via {route}..."):
            try:
                response = generate_hybrid(prompt_text, img_bytes, img_mime)
                st.write(response)
                add_to_history("assistant", response)
            except (google_exceptions.ResourceExhausted, GroqRateLimitError):
                st.error("❌ Rate limit exceeded. Wait ~1 min or upgrade tier.")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
if st.button("🔄 New Consultation"):
    for k in ["patient_data", "chat_history", "initial_consultation_done"]:
        st.session_state.pop(k, None)
    st.switch_page("app.py")