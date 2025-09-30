from google import genai
from google.genai import types
from gtts import gTTS
from IPython.display import Audio, display
from deep_translator import GoogleTranslator
import tempfile
import re
import PyPDF2
import docx
import json
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# -------------------- 1. Set up API Key --------------------
API_KEY = "AIzaSyA922EECbgq-itfXphPsK7voRl-BsJDQH8"
client = genai.Client(api_key=API_KEY)

# Supported languages
LANGUAGES = {
    "tamil": "ta",
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "korean": "ko",
    "malayalam": "ml",
    "bengali": "bn",
    "chinese": "zh-CN",
    "japanese": "ja"
}

# -------------------- 2. Extract text from different files --------------------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_json(json_file):
    data = json.load(json_file)
    if isinstance(data, dict):
        return json.dumps(data, indent=2)
    elif isinstance(data, list):
        return "\n".join([json.dumps(item) for item in data])
    else:
        return str(data)

def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with open(file_path, 'rb') as f:
            return extract_text_from_pdf(f)
    elif file_path.endswith(".docx"):
        with open(file_path, 'rb') as f:
            return extract_text_from_docx(f)
    elif file_path.endswith(".json"):
        with open(file_path, 'r', encoding="utf-8") as f:
            return extract_text_from_json(f)
    elif file_path.endswith(".txt"):
        with open(file_path, 'rb') as f:
            return extract_text_from_txt(f)
    else:
        return ""

# -------------------- 3. Gemini with context --------------------
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

# -------------------- 4. Clean text for speech --------------------
def clean_text_for_speech(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9.,?!\s\u0B80-\u0DFF\u0900-\u097F\u1100-\u11FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------- 5. Speak text --------------------
def speak_text(text: str, lang_code='en'):
    text = clean_text_for_speech(text)
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        temp_path = fp.name
        tts.save(temp_path)
        display(Audio(temp_path, autoplay=True))

# -------------------- 6. Safe translation --------------------
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

# -------------------- 7. Embedding and FAISS --------------------
print("Loading HuggingFace embedding model...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

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

# -------------------- 8. Main Tutor Loop --------------------
def main():
    print("📚 Multi-language File-powered Tutor with Embedding (Gemini + Voice + FAISS)")
    print("Supported languages:", ", ".join(LANGUAGES.keys()))

    user_lang_input = input("Choose your language (name or code): ").strip().lower()
    if user_lang_input in LANGUAGES.values():
        user_lang = [name for name, code in LANGUAGES.items() if code == user_lang_input][0]
    else:
        user_lang = user_lang_input

    if user_lang not in LANGUAGES:
        print("Language not supported, defaulting to English.")
        user_lang = "english"

    lang_code = LANGUAGES[user_lang]

    # Upload any type of file
    print("\n📤 Please upload your study material file (PDF, DOCX, JSON, TXT):")
    from google.colab import files
    uploaded = files.upload()

    file_text = ""
    for filename in uploaded.keys():
        file_text += extract_text(filename)

    # Split file_text into smaller chunks for embedding
    text_chunks = [file_text[i:i+1000] for i in range(0, len(file_text), 1000)]
    index, embeddings = create_faiss_index(text_chunks)
    print("✅ File loaded, embeddings created, FAISS index ready!\n")

    print("Type 'exit' to quit.\n")

    while True:
        question = input("You (ask anything from your file): ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if user_lang != "english":
            translated_question = safe_translate_natural(question, src=lang_code, target='en')
        else:
            translated_question = question

        # Use semantic search to get relevant context
        context = semantic_search(translated_question, text_chunks, index, top_k=3)

        # Get answer from Gemini
        answer = ask_gemini_with_context(translated_question, context)

        if user_lang != "english":
            answer = safe_translate_natural(answer, src='en', target=lang_code)

        print("\n📘 Teacher:\n", answer, "\n")
        speak_text(answer, lang_code=lang_code)

# -------------------- 9. Entry Point --------------------
if __name__ == "__main__":
    main()
