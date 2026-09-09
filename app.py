import os
import re
import numpy as np
import streamlit as st
import faiss
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
 
# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="RAG Chat with your PDF", page_icon="📄", layout="wide")
 
# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # open-source, small & fast (384-dim)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
 
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
]
 
# ---------------------------------------------------------
# Cached resources
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)
 
 
# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)
 
 
def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, breaking on sentence boundaries where possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
 
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    current = ""
 
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if chunks and chunk_overlap > 0:
                overlap_text = chunks[-1][-chunk_overlap:]
                current = (overlap_text + " " + sentence).strip()
            else:
                current = sentence
 
    if current:
        chunks.append(current)
 
    return chunks
 
 
def build_faiss_index(chunks, embed_model):
    embeddings = embed_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
 
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)
    return index
 
 
def retrieve_relevant_chunks(query, embed_model, index, chunks, top_k=TOP_K):
    query_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, min(top_k, len(chunks)))
    results = [(chunks[i], float(scores[0][pos])) for pos, i in enumerate(indices[0]) if i != -1]
    return results
 
 
def build_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(chunk for chunk, _ in retrieved_chunks)
    prompt = f"""You are a helpful assistant that answers questions using ONLY the context provided below, \
which was extracted from a PDF document. If the answer cannot be found in the context, say so honestly \
instead of making something up.
 
Context:
{context}
 
Question: {query}
 
Answer:"""
    return prompt
 
 
def call_groq(client, model, prompt):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    return chat_completion.choices[0].message.content
 
 
# ---------------------------------------------------------
# Session state init
# ---------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
 
# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
 
    default_key = os.environ.get("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys",
    )
 
    selected_model = st.selectbox("Groq model", GROQ_MODELS, index=0)
 
    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
 
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary", use_container_width=True):
            with st.spinner("Extracting text..."):
                raw_text = extract_text_from_pdf(uploaded_file)
 
            if not raw_text.strip():
                st.error("No extractable text found in this PDF (it may be scanned/image-only).")
            else:
                with st.spinner("Chunking text..."):
                    chunks = chunk_text(raw_text)
 
                with st.spinner(f"Embedding {len(chunks)} chunks and building FAISS index..."):
                    embed_model = load_embedding_model()
                    index = build_faiss_index(chunks, embed_model)
 
                st.session_state.chunks = chunks
                st.session_state.faiss_index = index
                st.session_state.pdf_processed = True
                st.session_state.chat_history = []
                st.success(f"Done! Indexed {len(chunks)} chunks.")
 
    if st.session_state.pdf_processed:
        st.info(f"📄 PDF loaded — {len(st.session_state.chunks)} chunks indexed.")
 
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
 
# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------
st.title("📄 Chat with your PDF (RAG + Groq)")
st.caption(
    "Upload a PDF, then ask questions about it. Retrieval runs locally with FAISS; "
    "answers are generated by an open-source LLM served via the Groq API."
)
 
if not st.session_state.pdf_processed:
    st.warning("👈 Upload a PDF and click **Process PDF** in the sidebar to get started.")
else:
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)
 
    user_question = st.chat_input("Ask a question about your PDF...")
 
    if user_question:
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.markdown(user_question)
 
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    embed_model = load_embedding_model()
                    retrieved = retrieve_relevant_chunks(
                        user_question,
                        embed_model,
                        st.session_state.faiss_index,
                        st.session_state.chunks,
                    )
                    prompt = build_prompt(user_question, retrieved)
 
                    try:
                        client = Groq(api_key=groq_api_key)
                        answer = call_groq(client, selected_model, prompt)
                    except Exception as e:
                        answer = f"⚠️ Error calling Groq API: {e}"
 
                    st.markdown(answer)
 
                    with st.expander("Show retrieved context"):
                        for i, (chunk, score) in enumerate(retrieved, start=1):
                            st.markdown(f"**Chunk {i} (score: {score:.3f})**")
                            st.text(chunk)
 
            st.session_state.chat_history.append(("assistant", answer))
 
