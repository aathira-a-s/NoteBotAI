# NoteBot AI 🎓 | Academic RAG Tutor

NoteBot AI is a cloud-native, high-performance Retrieval-Augmented Generation (RAG) conversational platform designed to parse dense academic text documents and provide immediate contextual answers. Featuring a striking, high-contrast **Dark Neo-Brutalist** design aesthetic, the application operates with a fully serverless backend layout optimized for fast document intelligence.

🔗 **Live Deployment:** [notebotai.streamlit.app](https://notebotai.streamlit.app)

---

## ⚡ Key Architectural Features

* **High-Speed Cloud Computation Layer:** Uses the Groq API running `llama-3.1-8b-instant` to generate contextual analysis with near-zero latency.
* **In-Memory Embedding Vectorization:** Completely serverless chunk vectorization powered by HuggingFace's open-source `all-MiniLM-L6-v2` transformers model.
* **FAISS Vector Matrix:** Leverages the Meta AI FAISS (Facebook AI Similarity Search) localized C++ engine for lightning-fast similarity lookups.
* **Dynamic Route Controller:** Features an isolated view-state routing pipeline seamlessly managing the user transitions between the Document Library grid and active conversational sessions.
* **Custom Dark Neo-Brutalist UI:** Styled explicitly via low-level CSS injections, replacing soft curves with flat saturated color anchors, monospace text structures, and heavy layout borders.

---

## 🛠️ Built With

* **Frontend Framework:** Streamlit (v1.x)
* **Orchestration Matrix:** LangChain / LangChain-Classic Core
* **Compute Engine:** Groq Cloud Systems
* **Embeddings Source:** HuggingFace Hub
* **Database Layer:** FAISS (CPU Optimized)

---

## ⚙️ Local Development Setup

To replicate this environment locally, make sure you have Python 3.10+ configured on your system, then follow these instructions:

### 1. Clone the Repository
```bash
git clone [https://github.com/aathira-a-s/NoteBotAI.git](https://github.com/aathira-a-s/NoteBotAI.git)
cd NoteBotAI