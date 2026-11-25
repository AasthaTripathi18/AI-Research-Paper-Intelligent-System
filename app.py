import io

import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="AI Research Paper System", layout="wide")

st.title("AI Research Paper Intelligent System")
st.write("Upload a research paper PDF to view its extracted text.")

pdf_file = st.file_uploader("Choose a research paper", type=["pdf"])

if pdf_file is not None:
    reader = PdfReader(io.BytesIO(pdf_file.getvalue()))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    st.success(f"Loaded {len(reader.pages)} page(s).")

    if text.strip():
        st.subheader("Extracted paper text")
        st.text_area("Paper content", text, height=400)
    else:
        st.warning("No selectable text was found. The PDF may be scanned images.")
