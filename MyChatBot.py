import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# Cloud replacements for local Ollama components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 1. Page Config
st.set_page_config(
    page_title="NoteBot AI | Academic Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject CSS for High-Contrast Dark Neo-Brutalist UI
st.markdown("""
    <style>
    /* Google Fonts for a raw, technical layout look */
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Lexend:wght@400;600;800&display=swap');

    /* Root Structure Overrides */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0B0C10 !important;
        font-family: 'Lexend', sans-serif !important;
        color: #FFFFFF !important;
    }

    /* Neo-Brutalist Monospace Typography for Specific Headers */
    h1, h2, h3 {
        font-family: 'Courier Prime', monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        color: #00FFAB !important; /* Neon Mint Mint Accent */
        letter-spacing: -1px;
    }

    /* Sidebar Background Panel */
    [data-testid="stSidebar"] {
        background-color: #12131C !important;
        border-right: 4px solid #000000 !important;
    }

    /* Primary Action Buttons (Thick borders + Flat crisp offset shadow) */
    .stButton>button {
        background-color: #FFDE4D !important; /* Cyberpunk Yellow */
        color: #000000 !important;
        font-family: 'Courier Prime', monospace !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border: 3px solid #000000 !important;
        border-radius: 0px !important; /* Sharp corners */
        box-shadow: 4px 4px 0px #000000 !important;
        transition: all 0.1s ease !important;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px #000000 !important;
    }
    .stButton>button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 1px 1px 0px #000000 !important;
    }

    /* Library Cards System Override */
    .doc-card {
        background-color: #181A26 !important;
        border: 3px solid #FFFFFF !important;
        border-radius: 0px !important;
        padding: 24px;
        margin-bottom: 12px;
        box-shadow: 6px 6px 0px #00FFAB !important; /* Neon drop shadow shadow */
    }
    .doc-card h4 {
        color: #FFFFFF !important;
        font-family: 'Lexend', sans-serif !important;
    }

    /* Native Streamlit Input Field Frame Adjustments */
    div[data-testid="stChatInput"] {
        border-radius: 0px !important;
    }
    div[data-testid="stChatInput"] textarea {
        border: 3px solid #000000 !important;
        background-color: #181A26 !important;
        color: #FFFFFF !important;
        border-radius: 0px !important;
        box-shadow: 4px 4px 0px #00FFAB !important;
    }

    /* Native Chat Message Frame Styling Blocks */
    div[data-testid="stChatMessage"] {
        border-radius: 0px !important;
        border: 3px solid #000000 !important;
        margin-bottom: 10px !important;
        box-shadow: 4px 4px 0px #000000 !important;
    }

    /* Differentiating User messages and Assistant containers visually via accents */
    div[data-testid="stChatMessageUser"] {
        background-color: #24273A !important;
        border-left: 8px solid #FFDE4D !important;
    }
    div[data-testid="stChatMessageAssistant"] {
        background-color: #181A26 !important;
        border-left: 8px solid #00FFAB !important;
    }

    /* Custom File Drag and Drop Box Wrap styling overrides */
    div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #00FFAB !important;
        border-radius: 0px !important;
        background-color: #181A26 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Securely pull API keys from Streamlit Cloud Secrets Manager
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# 3. Initialize Session State variables for dynamic routing
if "view" not in st.session_state:
    st.session_state.view = "library"  # Options: "library" or "chat"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================================================================
# 📂 SIDEBAR PANEL
# ==============================================================================
with st.sidebar:
    st.markdown("### NOTEBOT AI<br><small style='color:#00FFAB; font-family:monospace;'>[ ACADEMIC TUTOR ]</small>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation button to toggle views
    if st.button("📁 View Library", use_container_width=True):
        st.session_state.view = "library"
        st.rerun()

    st.divider()
    st.markdown("### 📥 UPLOAD DESK")
    uploaded_file = st.file_uploader("Upload notes PDF", type="pdf", label_visibility="collapsed")

    # Dynamic processing hook when a new file is uploaded
    if uploaded_file is not None and st.session_state.current_file != uploaded_file.name:
        if not GROQ_API_KEY:
            st.error("Missing GROQ_API_KEY! Please set it up in Streamlit Cloud Settings.")
        else:
            with st.status("⚡ CHUNKING & INDEXING...", expanded=True) as status:
                # Step A: Extract PDF Text
                pdf_reader = PdfReader(uploaded_file)
                text = "".join([page.extract_text() for page in pdf_reader.pages])

                # Step B: Semantic Chunking
                splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                chunks = splitter.split_text(text)

                # Step C: Free Cloud Vector Generation using HuggingFace Models
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []  # Clear historical chat frames for new doc
                st.session_state.view = "chat"  # Automatically route straight into chat view
                status.update(label="INDEX COMPLETE!", state="complete")
            st.rerun()

# ==============================================================================
# 🏛️ VIEW CONTROLLER (DYNAMIC ROUTING)
# ==============================================================================

# --- VIEW A: THE DOCUMENT LIBRARY GRID ---
if st.session_state.view == "library":
    st.markdown("## 📁 DOCUMENT LIBRARY")
    st.caption("Access and synthesize your loaded research materials.")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.current_file is None:
        st.info("Your library is currently empty. Drop a file into the sidebar uploader to launch your first session!")
    else:
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown(f"""
                <div class='doc-card'>
                    <span style='color:#FFDE4D; font-weight:bold; font-family:monospace; font-size:12px;'>[ ACTIVE BOOKSHELF ]</span>
                    <h4 style='margin-top:8px;'>📄 {st.session_state.current_file}</h4>
                    <p style='color:#A0AEC0; font-size:13px; font-family:monospace;'>Vector index successfully constructed and saved within active cloud runtime cache context.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("OPEN CHAT SESSION ➔", use_container_width=True):
                st.session_state.view = "chat"
                st.rerun()

# --- VIEW B: THE INTERACTIVE CHAT ENVIRONMENT ---
elif st.session_state.view == "chat":
    st.markdown(f"## 💬 CHATTING WITH: `{st.session_state.current_file}`")
    st.caption("Your high-speed cloud assistant tutor is active. Ask anything below.")

    # Render historical messages beautifully using Streamlit's native chat frames
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture query inputs dynamically
    user_query = st.chat_input("Ask NoteBot anything about your document...")

    if user_query:
        # 1. Render User Input immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # 2. Run contextual backend pipeline
        with st.chat_message("assistant"):
            with st.spinner("RUNNING RETRIEVAL MATRIX..."):
                # Similarity context lookup matching user vectors
                matching_chunks = st.session_state.vector_store.similarity_search(user_query)

                # Cloud LLM via free Groq API
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.1-8b-instant",
                    temperature=0
                )

                customized_prompt = ChatPromptTemplate.from_template(
                    """You are an elite academic assistant tutor. Answer the question directly based on the context provided.
                    If you cannot find the exact details within the context, simply say "I don't know".

                    Context: {context}
                    Question: {input}"""
                )

                # Run Chain Assembly Engine
                chain = create_stuff_documents_chain(llm, customized_prompt)
                output = chain.invoke({"input": user_query, "context": matching_chunks})

                st.markdown(output)

        # 3. Cache answer frames to chat memory array
        st.session_state.chat_history.append({"role": "assistant", "content": output})