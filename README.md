# AI Research Paper Intelligent System
**Semantic Search + Summarization + NER powered by NLP & Transformers**

An intelligent system that helps researchers and students discover relevant research papers and understand them faster using NLP, Transformers, and Vector Search.

---

### 📌 Problem 
With millions of research papers published every year, finding relevant papers and extracting key insights is slow and overwhelming.

This project tackles that by automating:
1. **Semantic Search** - Find papers based on meaning, not just keywords
2. **Auto Summarization** - Get short, readable summaries of abstracts
3. **Keyword Extraction** - Pull out important topics and phrases
4. **Technical NER** - Identify models, frameworks, libraries, and methods

---

### 📊 Dataset
**ML-ArXiv Papers** from Hugging Face  
~15,000 ML research papers with `Title` and `Abstract`

**Link**: [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)

---

### ✨ Key Features

#### 1. **Data Preprocessing**
Cleaned and merged `title + abstract` into a single text field for analysis.

#### 2. **Semantic Embeddings**
Used `all-MiniLM-L6-v2` Sentence Transformer to convert 15k papers into dense vectors.

#### 3. **Semantic Search with FAISS**
Built a FAISS index for fast similarity search.  
Example: `search_paper("deep learning for medical image analysis")`

#### 4. **Abstractive Summarization**
Used `facebook/bart-large-cnn` to generate concise summaries of top results.  
Example: `search_and_summarize("Deep learning in medical imaging", k=5)`

#### 5. **Keyword Extraction**
Used `KeyBERT` to extract single and multi-word key phrases like `medical imaging`, `deep learning`.

#### 6. **Named Entity Recognition**
Implemented 3 approaches to extract technical entities:
- **Rule-based NER**: spaCy EntityRuler for `BERT → MODEL`, `PyTorch → LIBRARY`
- **Zero-Shot NER**: BART-MNLI to classify unseen entities into `FRAMEWORK`, `METHOD`, `APPLICATION` etc.
- **Hybrid NER**: Extended rules + domain patterns for better accuracy  
  Example: `Sentence Transformers → LIBRARY`, `semantic similarity search → METHOD`

---

### 🛠️ Tech Stack
`Python` `Pandas` `NumPy` `Sentence-Transformers` `FAISS` `HuggingFace Transformers`  
`KeyBERT` `spaCy` `Scikit-learn`

---

### 📂 Project Structure
