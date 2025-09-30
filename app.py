import streamlit as st
from google import genai
from google.genai import types
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import PyPDF2
import docx
import json
import re
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from IPython.display import Audio, display

# -------------------- 1. API Setup --------------------
API_KEY = "AIzaSyA922EECbgq-itfXphPsK7voRl-BsJDQH8"
client = genai.Client(api_key=API_KEY)

# -------------------- 2. Supported Languages --------------------
LANGUAGES = {
    "english":"en","tamil":"ta","hindi":"hi","telugu":"te",
    "korean":"ko","malayalam":"ml","bengali":"bn","chinese":"zh-CN","japanese":"ja"
}

# -------------------- 3. Extract text from uploaded files --------------------
def extract_text(file):
    filename = getattr(file, 'name', 'file.txt')

    if filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    elif filename.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith(".json"):
        data = json.load(file)
        if isinstance(data, dict):
            return json.dumps(data, indent=2)
        elif isinstance(data, list):
            return "\n".join([json.dumps(item) for item in data])
        else:
            return str(data)

    elif filename.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        return ""

# -------------------- 4. Gemini Query with Context --------------------
def ask_gemini_with_context(question: str, context: str) -> str:
    try:
        full_prompt = (
            f"You are a knowledgeable and patient teacher. "
            f"Use the following study material as context to answer questions:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer clearly, step-by-step, and in a teaching style suitable for beginners."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"[Gemini API error]: {e}"

# -------------------- 5. Clean Text for Speech --------------------
def clean_text_for_speech(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9.,?!\s\u0B80-\u0DFF\u0900-\u097F\u1100-\u11FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def speak_text(text: str, lang_code='en'):
    text = clean_text_for_speech(text)
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        temp_path = fp.name
        tts.save(temp_path)
        display(Audio(temp_path, autoplay=True))

# -------------------- 6. Translation --------------------
def safe_translate_natural(text, src='auto', target='en', max_chunk=4000):
    if not text.strip():
        return text
    if len(text.strip()) < 2:
        text += " "
    text = "Translate in natural, simple, conversational language suitable for a beginner student: " + text
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunks.append(GoogleTranslator(source=src, target=target).translate(chunk))
        except Exception as e:
            print(f"[Translation error]: {e}")
            translated_chunks.append(chunk)
    return " ".join(translated_chunks)

# -------------------- 7. Embeddings + FAISS --------------------
st.info("Loading embedding model... this may take a few seconds")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
st.success("Embedding model loaded!")

def embed_text(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def create_faiss_index(text_chunks: list):
    embeddings = np.vstack([embed_text(chunk) for chunk in text_chunks])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def semantic_search(question: str, text_chunks: list, index, top_k=3):
    q_vec = embed_text(question)
    distances, indices = index.search(q_vec, top_k)
    results = [text_chunks[i] for i in indices[0]]
    return " ".join(results)

# -------------------- 8. Streamlit UI --------------------
st.title("📚 Multi-language File-powered Tutor (Gemini + FAISS)")

user_lang = st.selectbox("Choose your language", list(LANGUAGES.keys()))

uploaded_file = st.file_uploader("Upload your study material file (PDF, DOCX, JSON, TXT)")

if uploaded_file is not None:
    file_text = extract_text(uploaded_file)
    text_chunks = [file_text[i:i+1000] for i in range(0, len(file_text), 1000)]
    
    index, embeddings = create_faiss_index(text_chunks)
    st.success("✅ File processed and FAISS index ready!")

    question = st.text_input("Ask anything from your file:")

    if question:
        translated_question = safe_translate_natural(question, src=LANGUAGES[user_lang], target='en') if user_lang!="english" else question
        context = semantic_search(translated_question, text_chunks, index, top_k=3)
        answer = ask_gemini_with_context(translated_question, context)
        if user_lang != "english":
            answer = safe_translate_natural(answer, src='en', target=LANGUAGES[user_lang])
        
        st.subheader("📘 Teacher's Answer")
        st.write(answer)
        # Optionally play voice
        # speak_text(answer, lang_code=LANGUAGES[user_lang])
