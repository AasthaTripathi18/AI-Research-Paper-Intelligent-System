from __future__ import annotations

import re
from collections import Counter
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


st.set_page_config(page_title="AI Research Paper System", page_icon="📄", layout="wide")


def extract_pdf(file_bytes: bytes) -> tuple[str, dict[str, str], int]:
    reader = PdfReader(BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("This PDF is password protected. Please upload an unlocked copy.") from exc
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    metadata = reader.metadata
    details = {
        "Title": str(metadata.title or "Not provided") if metadata else "Not provided",
        "Author": str(metadata.author or "Not provided") if metadata else "Not provided",
    }
    return text, details, len(reader.pages)


def clean_text(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if len(s.split()) >= 7]


def make_summary(text: str, count: int = 5) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "Not enough readable text was extracted to create a summary."
    if len(sentences) <= count:
        return "\n\n".join(sentences)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(sentences)
    scores = matrix.sum(axis=1).A1
    chosen = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:count]
    return "\n\n".join(sentences[i] for i in sorted(chosen))


def extract_keywords(text: str, limit: int = 12) -> list[tuple[str, float]]:
    if len(text.split()) < 3:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
    try:
        matrix = vectorizer.fit_transform([text])
    except ValueError:
        return []
    terms = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]
    return [(str(terms[i]), float(scores[i])) for i in scores.argsort()[::-1][:limit]]


def find_sections(text: str) -> list[str]:
    pattern = re.compile(
        r"(?im)^\s*((?:\d+(?:\.\d+)*\s+)?(?:abstract|introduction|related work|background|"
        r"method(?:ology)?|approach|experiments?|results?|discussion|conclusion|references|limitations))\s*$"
    )
    return list(dict.fromkeys(match.group(1).strip() for match in pattern.finditer(text)))


st.title("AI Research Paper Intelligent System")
st.write("Upload a research paper PDF to extract its text, create a quick overview, and identify key terms.")

uploaded = st.file_uploader("Choose a research paper", type=["pdf"], help="Text-based PDFs work best. Scanned image PDFs need OCR.")

if uploaded:
    try:
        raw_text, metadata, page_count = extract_pdf(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
        st.stop()

    text = clean_text(raw_text)
    if not text:
        st.warning("No selectable text was found. This may be a scanned PDF; OCR support is not included in this version.")
        st.stop()

    st.success(f"Extracted text from {page_count} page(s).")
    left, middle, right = st.columns(3)
    left.metric("Pages", page_count)
    middle.metric("Words", f"{len(text.split()):,}")
    right.metric("Characters", f"{len(text):,}")

    title = metadata["Title"] if metadata["Title"] != "Not provided" else uploaded.name
    st.subheader(title)
    st.caption(f"Author: {metadata['Author']}")

    tabs = st.tabs(["Overview", "Keywords & sections", "Extracted text"])
    with tabs[0]:
        st.markdown("#### Quick summary")
        with st.spinner("Creating a concise extractive summary…"):
            try:
                st.write(make_summary(text))
            except ValueError:
                st.info("This paper is too short for automatic summarization.")
        st.caption("The summary selects representative sentences from the paper; it is not an AI-generated interpretation.")
    with tabs[1]:
        st.markdown("#### Important terms")
        keywords = extract_keywords(text)
        if keywords:
            st.write(" · ".join(f"**{term}**" for term, _ in keywords))
        else:
            st.info("Not enough text to extract keywords.")
        sections = find_sections(raw_text)
        st.markdown("#### Detected sections")
        st.write(" · ".join(sections) if sections else "No standard section headings were detected.")
    with tabs[2]:
        st.text_area("PDF text", value=raw_text, height=480)

    st.download_button(
        "Download extracted text",
        data=raw_text.encode("utf-8"),
        file_name=f"{uploaded.name.rsplit('.', 1)[0]}_extracted.txt",
        mime="text/plain",
    )
else:
    st.info("Start by uploading a research paper in PDF format.")

with st.sidebar:
    st.header("About")
    st.write("A lightweight research paper reader: PDF text extraction, a sentence-based summary, keyword extraction, and section detection.")
    st.caption("Your PDF is processed for this session and is not saved by this app.")
