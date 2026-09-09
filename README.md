Chat with your PDF (RAG + Groq)

A Retrieval-Augmented Generation (RAG) web app built with Streamlit. Upload a PDF, and ask questions about it in natural language. The app finds the most relevant parts of your document locally using open-source embeddings and FAISS, then sends only that context to an open-source LLM (served via the Groq API) to generate a grounded answer.

How it works
Extract — text is pulled out of the uploaded PDF using pypdf.
Chunk — the text is split into overlapping ~1000-character chunks, broken at sentence boundaries, so no single piece of context is too large or cuts a sentence in half.
Embed — each chunk is converted into a numeric vector using the open-source sentence-transformers model all-MiniLM-L6-v2. This runs locally — no external API call.
Index — the vectors are stored in a local FAISS vector index for fast similarity search.
Retrieve — when you ask a question, it's embedded the same way, and FAISS returns the top 4 most relevant chunks from your PDF.
Generate — those chunks are inserted into a prompt that instructs the model to answer using only that context, and sent to an open-source LLM running on Groq's infrastructure. The generated answer is streamed back to the chat.

Everything except the final answer-generation step (extraction, chunking, embeddings, vector search) runs locally/open-source. Only step 6 calls out to Groq's API, because generating fluent natural-language answers requires a full-scale LLM that can't run on a laptop or a free-tier server.

Features
Upload any text-based PDF and chat with it
Runs entirely on open-source components except the final generation step
Choice of Groq-hosted models from the sidebar
Expandable "Show retrieved context" panel under each answer, so you can see exactly which chunks of your PDF were used to generate it
Chat history persists during your session; "Clear chat" resets it
No data stored anywhere outside your session — nothing is written to a database
Project structure
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── runtime.txt         # Pins the Python version on Streamlit Cloud (3.12)
Setup — Run locally
1. Prerequisites
Python 3.10–3.12 (3.12 recommended)
A free Groq API key from console.groq.com/keys
2. Clone and install
bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
3. Set your API key (optional but convenient)

Create a file at .streamlit/secrets.toml:

toml
GROQ_API_KEY = "gsk_your_actual_key_here"

Alternatively, skip this and just paste your key into the sidebar's "Groq API Key" field once the app is running.

4. Run it
bash
streamlit run app.py

The app opens at http://localhost:8501.

Setup — Deploy on Streamlit Cloud
Push app.py, requirements.txt, and runtime.txt to a GitHub repository.
Go to share.streamlit.io and sign in with GitHub.
Click Create app → Deploy a public app from GitHub.
Select your repo, branch, and app.py as the main file.
Under Advanced settings:
Add a secret: GROQ_API_KEY = "gsk_your_actual_key_here"
Explicitly select Python 3.12 from the Python version dropdown (don't rely on runtime.txt alone — Streamlit Cloud has been known to ignore it).
Click Deploy. First build takes a few minutes since it installs sentence-transformers/torch.
Usage
Open the app and paste your Groq API key in the sidebar (if not already set via secrets).
Upload a PDF using the sidebar file uploader.
Click Process PDF — this extracts, chunks, embeds, and indexes the document.
Once processing finishes, type your question in the chat box at the bottom.
Expand "Show retrieved context" under any answer to see the exact PDF passages used to generate it.
Upload a new PDF or click Clear chat to start over.
Configuration

These are set as constants near the top of app.py and can be adjusted directly in the code:

Setting	Default	Description
EMBED_MODEL_NAME	all-MiniLM-L6-v2	Open-source embedding model (384-dimensional)
CHUNK_SIZE	1000	Target characters per chunk
CHUNK_OVERLAP	150	Characters carried over between consecutive chunks
TOP_K	4	Number of chunks retrieved per question
GROQ_MODELS	see app.py	Dropdown list of Groq-hosted models available in the sidebar
Tech stack
Purpose	Tool
Frontend / app framework	Streamlit
PDF text extraction	pypdf
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Vector search	FAISS (faiss-cpu)
LLM inference	Groq API
Troubleshooting

Error calling Groq API: ... model does not exist or you do not have access to it Groq periodically deprecates models. Check console.groq.com/docs/models for the current list of production models available on your plan, and update the GROQ_MODELS list in app.py accordingly. As of this writing, openai/gpt-oss-20b and openai/gpt-oss-120b are current, openly-available production models.

No solution found when resolving dependencies / faiss-cpu has no wheels (on Streamlit Cloud) This happens when Streamlit Cloud's build machine uses a very new Python version that an exactly-pinned package doesn't have a wheel for. Make sure requirements.txt uses minimum-version constraints (>=) rather than exact pins, and explicitly select Python 3.12 in Advanced settings when deploying.

"No extractable text found in this PDF" The uploaded PDF is likely scanned/image-only rather than containing real text. This app does not include OCR — you'd need to run the PDF through an OCR tool first to extract text from it.

Answers seem to ignore the PDF / hallucinate Open "Show retrieved context" to check whether the right chunks were actually retrieved. If they're irrelevant, try rephrasing your question, or increase TOP_K for a broader retrieval net.

Notes on data & privacy
The FAISS index and processed chunks live only in Streamlit's session_state — they reset on app restart or when a new user opens the app. Each user re-uploads and re-processes their own PDF each session; nothing is persisted to disk or a database.
Your PDF's retrieved chunks (not the whole document) are sent to Groq's API as part of the prompt when generating an answer — review Groq's data policy if this matters for sensitive documents.
