# NoteBot AI  | Academic RAG Tutor

NoteBot AI is a cloud-native, high-performance Retrieval-Augmented Generation (RAG) conversational platform designed to parse dense academic text documents and provide immediate contextual answers. Featuring a striking, high-contrast **Dark Neo-Brutalist** design aesthetic, the application operates with a fully serverless backend layout optimized for fast document intelligence.

---

##  Key Architectural Features

* **High-Speed Cloud Computation Layer:** Uses the Groq API running `llama-3.1-8b-instant` to generate contextual analysis with near-zero latency.
* **In-Memory Embedding Vectorization:** Completely serverless chunk vectorization powered by HuggingFace's open-source `all-MiniLM-L6-v2` transformers model.
* **FAISS Vector Matrix:** Leverages the Meta AI FAISS (Facebook AI Similarity Search) localized C++ engine for lightning-fast similarity lookups.
* **Dynamic Route Controller:** Features an isolated view-state routing pipeline seamlessly managing the user transitions between the Document Library grid and active conversational sessions.
* **Custom Dark Neo-Brutalist UI:** Styled explicitly via low-level CSS injections, replacing soft curves with flat saturated color anchors, monospace text structures, and heavy layout borders.

---

##  Built With

* **Frontend Framework:** Streamlit (v1.x)
* **Orchestration Matrix:** LangChain / LangChain-Classic Core
* **Compute Engine:** Groq Cloud Systems
* **Embeddings Source:** HuggingFace Hub
* **Database Layer:** FAISS (CPU Optimized)

---

## System Architecture Diagram

```mermaid
graph TD
    %% Styling Configuration
    classDef default fill:#121212,stroke:#00ffcc,stroke-width:2px,color:#fff;
    classDef user fill:#00ffcc,stroke:#000,stroke-width:2px,color:#000;
    classDef external fill:#222,stroke:#ff0055,stroke-width:2px,color:#ff0055;

    %% Workflow Nodes & Styling
    User([User Interaction])
    class User user;
    
    StreamlitUI[Streamlit Dark Neo-Brutalist UI]

    %% Connections
    User -->|Uploads PDF| StreamlitUI
    
    subgraph Data Ingestion Pipeline
        StreamlitUI --> PyPDF[PyPDF Document Loader]
        PyPDF --> TextSplitter[LangChain Text Splitter]
        TextSplitter -->|Text Chunks| HFEmbed[HuggingFace Hub: all-MiniLM-L6-v2]
        HFEmbed -->|Vector Embeddings| FAISS[(FAISS Vector Matrix)]
    end

    User -->|Submits Question| StreamlitUI
    StreamlitUI -->|User Query| SimSearch{FAISS Similarity Search}
    FAISS -.-> SimSearch
    
    subgraph Contextual Inference Engine
        SimSearch -->|Top Matching Chunks| PromptTemplate[LangChain ChatPromptTemplate]
        PromptTemplate -->|Structured Payload| GroqAPI[Groq Inference Engine]
        GroqAPI -->|llama-3.1-8b-instant| GenResponse[Contextual Answer]
    end

    GenResponse -->|Render Text Layer| StreamlitUI
