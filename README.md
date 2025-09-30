📚 Multi-file Multi-language Data Science Tutor (Gemini + Voice)

This project is an advanced AI-powered **Data Science tutor** that can answer questions from multiple study materials (PDF, DOCX, JSON, TXT) with **voice assistance** and **multi-language support**. It leverages **Google Gemini API**, **gTTS**, and **Deep Translator** to provide natural, beginner-friendly explanations.

---

## Features

- **Multi-file Support:** Upload multiple study files at once (`PDF`, `DOCX`, `JSON`, `TXT`).
- **PDF/Document Parsing:** Automatically extracts text from uploaded files.
- **AI Tutor:** Uses **Google Gemini API** to generate clear, step-by-step answers.
- **Voice Assistant:** Answers are read aloud using **Google Text-to-Speech (gTTS)**.
- **Multi-language Support:** Currently supports:
  - Tamil, English, Hindi, Telugu, Korean, Malayalam, Bengali, Chinese, Japanese
- **Natural Translation:** Converts answers and questions to a beginner-friendly style in the chosen language.
- **Beginner-friendly:** Answers are simplified and explained in a teaching style.

---

1)  Install required packages:
pip install deep-translator gtts PyPDF2 python-docx

Usage

2)  Run the notebook or Python script.

Select your language (name or code).

Upload your study files (PDF, DOCX, JSON, TXT). You can upload multiple files at once.

Ask any question related to the uploaded materials.

Get a text answer and a spoken answer in your chosen language.

Type exit to quit the tutor.

3)  Example

📚 Multi-file Multi-language Tutor (Gemini + Voice)
Supported languages: tamil, english, hindi, telugu, korean, malayalam, bengali, chinese, japanese

Choose your language (name or code): english

Please upload your study material files (PDF, DOCX, JSON, TXT):

You (ask anything from your files): What is supervised learning?

📘 Teacher:
Supervised learning is a type of machine learning where the algorithm learns from labeled data. The model is trained using input-output pairs to predict outcomes on new, unseen data...


4)  File Types Supported

PDF: .pdf

Word Documents: .docx

JSON: .json

Text files: .txt

5)  Dependencies

Google Gemini API

gTTS

Deep Translator

PyPDF2

python-docx

IPython


6)  Notes

The tutor is designed for beginner-friendly explanations.

Text-to-speech may slightly vary depending on the language.

Translation is simplified for readability in target languages.

