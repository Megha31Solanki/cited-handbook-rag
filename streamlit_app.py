import os
import fitz
import streamlit as st

from streamlit_pdf_viewer import pdf_viewer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import OllamaLLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Handbook RAG Bot",
    page_icon="📚",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    .stChatMessage {
        border-radius: 18px;
        padding: 12px;
        margin-bottom: 12px;
        border: 1px solid #2A2F3A;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        color: white;
    }

    .subtitle {
        color: #A0A0A0;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .source-box {
        padding: 10px;
        border-radius: 10px;
        background-color: #1A1D24;
        margin-bottom: 8px;
        border: 1px solid #333;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class='title'>📚 Handbook RAG Bot</div>
    <div class='subtitle'>
    AI-powered PDF Question Answering with Citations, Summaries & Highlighted Sources
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- PATHS ----------------
DATA_PATH = "data/docs"
VECTOR_PATH = "data/vectorstore"

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("⚙️ Control Panel")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    
    st.markdown("---")

    st.markdown("### ✨ Features")

    st.markdown("✅ Semantic Search")
    st.markdown("✅ Local LLM")
    st.markdown("✅ PDF Highlighting")
    st.markdown("✅ Citations")
    st.markdown("✅ Persistent Knowledge Base")
   

# ---------------- PDF PROCESSING ----------------
if uploaded_file:

    save_path = os.path.join(
        DATA_PATH,
        uploaded_file.name
    )

    # save uploaded file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("PDF Uploaded Successfully ✅")

    # load pdf
    loader = PyPDFLoader(save_path)

    docs = loader.load()

    # chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    # update vector database
    if os.path.exists(VECTOR_PATH):

        existing_db = FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        existing_db.add_documents(chunks)

        vectorstore = existing_db

    else:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    # save db
    vectorstore.save_local(VECTOR_PATH)

    st.sidebar.success("Knowledge Base Updated 🚀")

# ---------------- LOAD VECTOR DB ----------------
vectorstore = FAISS.load_local(
    VECTOR_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------- RETRIEVER ----------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ---------------- LOCAL LLM ----------------
llm = OllamaLLM(
    model="llama3"
)



# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:

    st.session_state.messages = []

# ---------------- CHAT INPUT ----------------
question = st.chat_input(
    "Ask anything from your uploaded PDFs..."
)

if question:

    # save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # retrieve docs
    docs = retriever.invoke(question)

    context = ""

    sources = []

    highlight_data = []

    for doc in docs:

        source_path = doc.metadata.get(
            "source",
            "Unknown"
        )

        source = os.path.basename(source_path)

        page = doc.metadata.get(
            "page",
            0
        )

        chunk_text = doc.page_content

        context += chunk_text + "\n\n"

        sources.append(
            f"{source} | Page {page}"
        )

        highlight_data.append(
            {
                "source": source,
                "page": page,
                "chunk_text": chunk_text
            }
        )

    # prompt
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not found in context,
say exactly:
"I don't know based on the documents."

Context:
{context}

Question:
{question}
"""

    # generate answer
    with st.spinner("Thinking..."):

        response = llm.invoke(prompt)

    # save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": list(set(sources)),
            "highlight_data": highlight_data
        }
    )

# ---------------- DISPLAY CHAT ----------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        # assistant extras
        if message["role"] == "assistant":

            # sources
            with st.expander("📄 Show Sources"):

                for src in message["sources"]:

                    st.markdown(
                        f"<div class='source-box'>{src}</div>",
                        unsafe_allow_html=True
                    )

            # highlighted pdf
            with st.expander("✨ View Highlighted PDF"):

                for item in message["highlight_data"]:

                    pdf_path = os.path.join(
                        DATA_PATH,
                        item["source"]
                    )

                    page_num = item["page"]

                    chunk_text = item["chunk_text"]

                    try:

                        pdf_doc = fitz.open(pdf_path)

                        page = pdf_doc[page_num]

                        search_text = chunk_text[:100]

                        matches = page.search_for(search_text)

                        for match in matches:

                            highlight = page.add_highlight_annot(match)

                            highlight.update()

                        highlighted_file = "highlighted.pdf"

                        pdf_doc.save(highlighted_file)

                        pdf_doc.close()

                        st.markdown(
                            f"### 📘 {item['source']} | Page {page_num}"
                        )

                        pdf_viewer(highlighted_file)

                    except Exception as e:

                        st.error(
                            f"Highlight Error: {e}"
                        )


