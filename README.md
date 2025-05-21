# 📄 Document Chat with RAG

A powerful Streamlit-based application that lets you **chat with your documents** using a **Retrieval-Augmented Generation (RAG)** pipeline. This app supports **PDF, DOCX, and TXT** files, powered by **Google Gemini (via LangChain)** and **FAISS vector search**. Built with scalability, transparency, and usability in mind.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔍 **Ask Questions** | Chat with uploaded documents using natural language. |
| 📄 **Multi-document support** | Upload multiple PDFs, DOCX, or TXT files at once. |
| 📚 **Citations with Metadata** | Each answer includes inline citations like (DocumentName, Page X, Chunk Y). |
| 📖 **Chunk Preview Panel** | Shows a few extracted document chunks with metadata after processing. |
| 📁 **Multi-source Attribution** | Answers clarify which document(s) information came from. |
| 💾 **Download Chat History** | Download your entire Q&A conversation as `.txt` or `.md`. |
| ❗ **Graceful API Error Handling** | Descriptive feedback for invalid API keys or exceeded quotas. |
| ⚙️ **Configurable Parameters** | Adjust chunk size, overlap, and retrieved document count. |
| 🔐 **Secure API Input** | Gemini API key is entered securely from the sidebar. |

---

## 🚀 How it Works

### 1. Upload Files  
Supported formats: `.pdf`, `.docx`, `.txt`  
The app reads and extracts raw text from each document.

### 2. Chunk Splitting  
Text is split into overlapping chunks using LangChain’s `RecursiveCharacterTextSplitter`, preserving context.

### 3. Embeddings & Vector Store  
Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a **FAISS** vector database.

### 4. Conversational Retrieval Chain  
A `ConversationalRetrievalChain` queries documents using Google Gemini (via LangChain wrapper). It:
- Reformulates user questions using history.
- Retrieves top-k chunks.
- Generates a contextual answer with **citations**.

### 5. Metadata & Citation Injection  
Each chunk carries metadata (filename, page, chunk ID), which is used in the answer to show sources clearly.

---

## 🧠 Technologies Used

- [Streamlit](https://streamlit.io/) - UI and interactivity
- [LangChain](https://python.langchain.com/) - RAG pipeline
- [Google Gemini](https://ai.google.dev/) - LLM for generation
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- `pypdf`, `docx2txt`, `io` - Document loaders

---
 Made with ❤️ by **Dhruv Tambekar**

