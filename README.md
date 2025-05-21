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

## 🖼️ Sample Usage Flow

1. **Upload** multiple `.pdf`, `.docx`, or `.txt` files using the sidebar.
2. Click **"Process Documents"** to extract and vectorize the content.
3. Preview **2–3 extracted text chunks** for transparency and debugging.
4. Start chatting with prompts like:
   - _“What is the summary of the first document?”_
   - _“What does document 2 say about AI?”_
   - _“Compare views on sustainability across all documents”_
5. View **inline citations** like:

   > _"The company will achieve carbon neutrality by 2030_ **(ClimatePolicy.pdf, Page 5, Chunk 3)**"

6. Click **Download Chat History** to save the full session as `.txt` or `.md`.

---

## 📌 Notes & Tips

- ✅ **Recommended chunk size:** Keep it around **800–1000** for best performance.
- 🧽 **Clean documents work best:** Use **text-based PDFs** or properly **OCR-processed** files.
- 🚫 **API quota issues?** Errors are shown in red with helpful debugging messages.
- 📝 **Answers are Markdown-formatted:** Responses support formatting like headings, bullet points, bold, and italics for better clarity.

---
 Made with ❤️ by **Dhruv Tambekar**

