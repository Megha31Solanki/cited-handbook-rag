import os
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = "data/docs"
VECTOR_PATH = "data/vectorstore"

all_docs = []

# load all PDFs
for file in os.listdir(DATA_PATH):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DATA_PATH, file)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        all_docs.extend(docs)

print(f"Loaded {len(all_docs)} pages")

# chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print(f"Created {len(chunks)} chunks")

# embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create vector database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# save locally
vectorstore.save_local(VECTOR_PATH)

print("\nFAISS Vector Database Created Successfully")