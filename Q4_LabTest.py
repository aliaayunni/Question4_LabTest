import streamlit as st
from PyPDF2 import PdfReader
import nltk

# Download tokenizer if not available
nltk.download("punkt")
from nltk.tokenize import sent_tokenize



st.set_page_config(page_title="Text Chunking using NLTK", layout="centered")
st.title("Text Chunking using NLTK Sentence Tokenizer")

st.write("Upload a PDF file to extract text and perform sentence-level semantic chunking.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
   
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text()

    st.success("PDF text extracted successfully!")

    # --------------------------------------------------
    # STEP 3: Sentence Tokenization
    # --------------------------------------------------
    sentences = sent_tokenize(full_text)

    st.subheader("Sample Extracted Sentences (Index 58–68)")

    sample_sentences = sentences[58:69]

    for i, sentence in enumerate(sample_sentences, start=58):
        st.write(f"**Sentence {i}:** {sentence}")

    # --------------------------------------------------
    # STEP 4: Semantic Sentence Chunking
    # --------------------------------------------------
    st.subheader("Semantic Sentence Chunking Output")

    chunked_text = "\n\n".join(sample_sentences)
    st.text_area("Chunked Text", chunked_text, height=300)

else:
    st.info("Please upload a PDF file to proceed.")
