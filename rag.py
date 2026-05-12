from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import OllamaLLM

VECTOR_PATH = "data/vectorstore"

# embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load vector database
vectorstore = FAISS.load_local(
    VECTOR_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# local LLM
llm = OllamaLLM(
    model="llama3"
)

print("\nRAG Chat Ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("Ask Question: ")

    if question.lower() == "exit":
        break

    # retrieve relevant chunks
    docs = retriever.invoke(question)

    context = ""

    sources = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        sources.append(f"{source} | Page {page}")

        context += doc.page_content + "\n\n"

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

    # generate response
    response = llm.invoke(prompt)

    print("\nANSWER:\n")

    print(response)

    print("\nSOURCES:\n")

    for src in set(sources):

        print(src)