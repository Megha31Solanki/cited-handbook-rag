# 📚 Handbook RAG Bot

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and ask natural-language questions with grounded answers and source citations.

---

## 🚀 Features

- 📄 PDF Upload System
- 🧠 Semantic Search using FAISS
- 🤖 Local LLM Support with Ollama
- 📚 Citation-based Answers
- 🚫 Hallucination Guardrails
- 💬 Modern Chat-style UI
- 📂 Multi-PDF Knowledge Base
- ✨ Highlighted PDF References
- 🌙 Professional Dark Theme UI

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS Vector Database
- HuggingFace Embeddings
- Ollama
- Llama3 / Phi3
- PyMuPDF

---

## 📦 Installation

Clone the repository:
Go to project folder:
cd cited-handbook-rag
Install dependencies:
pip install -r requirements.txt

▶️ Run the Application
Start Ollama locally:
ollama run phi3
Then run Streamlit app:
streamlit run streamlit_app.py

📌 How It Works
PDF Upload   ↓Chunking   ↓Embeddings   ↓FAISS Vector Database   ↓Semantic Retrieval   ↓LLM Response Generation   ↓Answer + Citations

🔥 Key AI Features


Retrieval-Augmented Generation (RAG)


Semantic Document Search


Grounded Answer Generation


Citation-based Responses


Local LLM Integration


Multi-document Retrieval


PDF Highlighting


AI Guardrails for Hallucination Reduction



💡 Example Questions


What is file I/O?


Explain exception handling.


What are the common file modes?


Summarize this topic.


Explain this concept in simple words.



📷 UI Preview
Upload PDFs, ask questions, view citations, and inspect highlighted PDF references through a modern Streamlit interface.

📌 Use Cases


Study Assistant


Research PDF Chatbot


Company Handbook Assistant


Documentation Search


Interview Preparation


Notes & Concept Revision



🚀 Future Improvements


Inline Citations


Hybrid Search


Reranking


OCR Support


Evaluation Dashboard


Streaming Responses


Deployment Support



👩‍💻 Built By
Megha Solanki



git clone <your-repo-link>
