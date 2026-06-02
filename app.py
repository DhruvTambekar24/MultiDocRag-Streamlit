import streamlit as st
import os
import io
import time
import datetime
import hashlib
from pathlib import Path
from collections import Counter
import pypdf
import docx2txt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

st.set_page_config(
    page_title="DocMind · RAG Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:        #0c0e14;
    --surface:   #13161f;
    --surface2:  #1a1e2a;
    --border:    #252a38;
    --accent:    #6ee7b7;
    --accent2:   #818cf8;
    --accent3:   #f472b6;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --user-bub:  #1e2a3a;
    --ai-bub:    #141a26;
    --radius:    14px;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.docmind-header {
    display: flex; align-items: center; gap: 14px;
    padding: 0 0 1.4rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.4rem;
}
.docmind-header .logo {
    font-family: var(--font-head); font-size: 1.9rem; font-weight: 800;
    letter-spacing: -0.5px; line-height: 1;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.docmind-header .tagline {
    font-size: 0.78rem; color: var(--muted); font-weight: 300;
    letter-spacing: 0.06em; text-transform: uppercase;
}

.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1rem; }
.stat-pill {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 999px; padding: 4px 14px; font-size: 0.78rem;
    color: var(--muted); display: flex; align-items: center; gap: 6px;
}
.stat-pill span { color: var(--accent); font-weight: 600; }

.chat-wrap { display: flex; flex-direction: column; gap: 18px; margin-bottom: 1.5rem; }

.bubble-user, .bubble-ai {
    max-width: 80%; padding: 14px 18px; border-radius: var(--radius);
    font-size: 0.92rem; line-height: 1.65; position: relative;
    animation: popIn 0.25s ease;
}
@keyframes popIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.bubble-user {
    background: var(--user-bub); border: 1px solid #2a3a52;
    align-self: flex-end; border-bottom-right-radius: 4px;
}
.bubble-ai {
    background: var(--ai-bub); border: 1px solid var(--border);
    align-self: flex-start; border-bottom-left-radius: 4px;
}
.bubble-label {
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 6px; opacity: 0.55;
}

.citations-box {
    background: var(--surface2); border-left: 3px solid var(--accent2);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 10px 14px; margin-top: 10px; font-size: 0.78rem;
    color: var(--muted); line-height: 1.7;
}
.citations-box strong { color: var(--accent); }

.chunk-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 14px 16px;
    margin-bottom: 10px; font-size: 0.83rem; line-height: 1.65;
}
.chunk-meta {
    font-size: 0.72rem; color: var(--accent2); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0c0e14 !important; border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-head) !important; font-weight: 700 !important;
    letter-spacing: 0.03em !important; padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--text) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(110,231,183,0.15) !important;
}

.stSlider > div { color: var(--accent) !important; }

[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

.stSelectbox > div > div {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--text) !important;
}

details { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
summary { color: var(--text) !important; font-weight: 500 !important; }

.stSuccess { background: rgba(110,231,183,0.08) !important; border: 1px solid var(--accent) !important; border-radius: var(--radius) !important; }
.stError   { background: rgba(244,114,182,0.08) !important; border: 1px solid var(--accent3) !important; border-radius: var(--radius) !important; }
.stWarning { background: rgba(129,140,248,0.08) !important; border: 1px solid var(--accent2) !important; border-radius: var(--radius) !important; }

.stProgress > div > div { background: var(--accent) !important; }
hr { border-color: var(--border) !important; }

.sb-section {
    font-family: var(--font-head); font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--muted); margin: 1.2rem 0 0.5rem 0;
}

.empty-state { text-align: center; padding: 4rem 2rem; color: var(--muted); }
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state h3 { font-family: var(--font-head); font-size: 1.2rem; color: var(--text); margin-bottom: 0.5rem; }
.empty-state p  { font-size: 0.88rem; line-height: 1.6; }

.stDownloadButton > button {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; font-family: var(--font-body) !important;
    font-size: 0.82rem !important; border-radius: var(--radius) !important;
}
.stDownloadButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULTS = {
    "chat_history":      [],    # [{role, content, citations, time}]
    "lc_messages":       [],    # [HumanMessage | AIMessage] for LangChain context
    "vector_store":      None,
    "retrieval_chain":   None,  # create_retrieval_chain instance
    "processed_docs":    [],
    "chunk_previews":    [],
    "total_chunks":      0,
    "doc_hash":          "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _file_hash(files) -> str:
    h = hashlib.md5()
    for f in sorted(files, key=lambda x: x.name):
        h.update(f.name.encode())
        h.update(str(f.size).encode())
    return h.hexdigest()


def extract_text_from_file(uploaded_file) -> list:
    name = uploaded_file.name
    ext  = Path(name).suffix.lower()
    docs = []

    if ext == ".pdf":
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": name, "page": page_num},
                ))

    elif ext == ".docx":
        text = docx2txt.process(io.BytesIO(uploaded_file.read()))
        for i, start in enumerate(range(0, len(text), 3000), start=1):
            chunk = text[start:start + 3000]
            if chunk.strip():
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": name, "page": i},
                ))

    elif ext == ".txt":
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        for i, start in enumerate(range(0, len(text), 3000), start=1):
            chunk = text[start:start + 3000]
            if chunk.strip():
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": name, "page": i},
                ))

    return docs


def build_vector_store(all_docs: list, chunk_size: int, overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in all_docs:
        splits = splitter.split_documents([doc])
        for idx, split in enumerate(splits, start=1):
            split.metadata["chunk"] = idx
        chunks.extend(splits)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks

SYSTEM_PROMPT = """You are DocMind, an expert document analyst.
Answer the user's question using ONLY the provided context.
After every factual claim append an inline citation: (SourceFile, Page X, Chunk Y).
If the answer spans multiple documents, clearly attribute each part.
If the context doesn't contain the answer, say so honestly.
Format your answer in clean Markdown (bold, bullet lists, headings where helpful).

Context:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_chain(vector_store, groq_api_key: str, top_k: int):
    """
    Uses the modern create_retrieval_chain + create_stuff_documents_chain pattern.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.3,
    )

    llm.invoke("hi")

    stuff_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": top_k}),
        stuff_chain
    )

    return retrieval_chain


def format_citations(source_docs) -> str:
    seen, lines = set(), []
    for doc in source_docs:
        m   = doc.metadata
        key = (m.get("source","?"), m.get("page","?"), m.get("chunk","?"))
        if key not in seen:
            seen.add(key)
            lines.append(f"📄 **{key[0]}** · Page {key[1]} · Chunk {key[2]}")
    return "\n".join(lines)


def chat_history_to_text(history: list) -> str:
    lines = [f"DocMind Chat Export — {datetime.datetime.now():%Y-%m-%d %H:%M}\n{'='*60}\n"]
    for msg in history:
        role = "You" if msg["role"] == "user" else "DocMind"
        lines.append(f"[{msg.get('time','')}] {role}:\n{msg['content']}")
        if msg.get("citations"):
            lines.append(f"\nSources:\n{msg['citations']}")
        lines.append("")
    return "\n".join(lines)


def chat_history_to_md(history: list) -> str:
    lines = [f"# DocMind Chat Export\n*{datetime.datetime.now():%Y-%m-%d %H:%M}*\n\n---\n"]
    for msg in history:
        role = "**You**" if msg["role"] == "user" else "**🧠 DocMind**"
        lines.append(f"### {role} `{msg.get('time','')}`\n\n{msg['content']}\n")
        if msg.get("citations"):
            lines.append(f"> **Sources:**\n> {msg['citations'].replace(chr(10), chr(10)+'> ')}\n")
        lines.append("---\n")
    return "\n".join(lines)


with st.sidebar:
    st.markdown(
        """<div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;
                    background:linear-gradient(135deg,#6ee7b7,#818cf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    margin-bottom:0.2rem;">🧠 DocMind</div>
        <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:1.2rem;">RAG · Document Intelligence</div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section">🔐 Groq API Key</div>', unsafe_allow_html=True)
    groq_api_key = st.text_input(
        "Groq API Key", type="password", placeholder="gsk_...",
        label_visibility="collapsed",
        help="Get from https://console.groq.com/keys",
    )

    st.markdown('<div class="sb-section">📁 Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop files here", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    st.markdown('<div class="sb-section">⚙️ Settings</div>', unsafe_allow_html=True)
    chunk_size   = st.slider("Chunk size (tokens)", 400, 2000, 900, 50)
    overlap      = st.slider("Chunk overlap", 0, 400, 120, 20)
    top_k        = st.slider("Retrieved chunks (k)", 2, 15, 5, 1)

    st.markdown("---")
    process_btn = st.button("⚡ Process Documents", use_container_width=True)

    if st.button("🗑️ Clear Session", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v if not isinstance(v, list) else []
        st.rerun()

    if st.session_state["chat_history"]:
        st.markdown('<div class="sb-section">💾 Export Chat</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📄 .txt",
                data=chat_history_to_text(st.session_state["chat_history"]),
                file_name="docmind_chat.txt", mime="text/plain",
                use_container_width=True)
        with c2:
            st.download_button("📝 .md",
                data=chat_history_to_md(st.session_state["chat_history"]),
                file_name="docmind_chat.md", mime="text/markdown",
                use_container_width=True)

    if st.session_state["processed_docs"]:
        st.markdown('<div class="sb-section">📚 Loaded Documents</div>', unsafe_allow_html=True)
        for d in st.session_state["processed_docs"]:
            st.markdown(
                f"""<div class="chunk-card" style="margin-bottom:6px;padding:10px 12px;">
                    <div class="chunk-meta">{d['name']}</div>
                    {d['pages']} pages · {d['chunks']} chunks
                </div>""",
                unsafe_allow_html=True,
            )

if process_btn:
    if not groq_api_key:
        st.error("🔑 Please enter your Groq API key in the sidebar.")
    elif not uploaded_files:
        st.warning("📂 Please upload at least one document.")
    else:
        new_hash = _file_hash(uploaded_files)
        if new_hash != st.session_state["doc_hash"]:
            with st.spinner("Reading and vectorising your documents…"):
                progress    = st.progress(0, text="Extracting text…")
                all_docs    = []
                doc_summary = []

                for i, f in enumerate(uploaded_files):
                    f.seek(0)
                    docs = extract_text_from_file(f)
                    all_docs.extend(docs)
                    doc_summary.append({"name": f.name, "pages": len(docs), "chunks": 0})
                    progress.progress((i + 1) / len(uploaded_files), text=f"Parsed {f.name}")

                progress.progress(0.7, text="Building FAISS index…")
                try:
                    vs, chunks = build_vector_store(all_docs, chunk_size, overlap)
                except Exception as e:
                    st.error(f"❌ Embedding error: {e}")
                    st.stop()

                counts = Counter(c.metadata.get("source") for c in chunks)
                for d in doc_summary:
                    d["chunks"] = counts.get(d["name"], 0)

                progress.progress(0.9, text="Initialising Groq chain…")
                try:
                    chain = build_chain(vs, groq_api_key, top_k)
                except Exception as e:
                    st.error(f"❌ LLM init error: {e}")
                    st.stop()

                st.session_state["vector_store"]    = vs
                st.session_state["retrieval_chain"] = chain
                st.session_state["processed_docs"]  = doc_summary
                st.session_state["total_chunks"]    = len(chunks)
                st.session_state["chunk_previews"]  = chunks[:4]
                st.session_state["doc_hash"]        = new_hash
                st.session_state["chat_history"]    = []
                st.session_state["lc_messages"]     = []

                progress.progress(1.0, text="Ready!")
                time.sleep(0.4)
                progress.empty()

            st.success(f"✅ {len(uploaded_files)} document(s) processed · {len(chunks):,} chunks indexed")
        else:
            st.info("ℹ️ Same files detected — using existing index. Clear session to reprocess.")


st.markdown(
    """<div class="docmind-header">
        <div>
            <div class="logo">🧠 DocMind</div>
            <div class="tagline">Retrieval-Augmented Generation · Document Intelligence</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

if st.session_state["processed_docs"]:
    st.markdown(
        f"""<div class="stat-row">
            <div class="stat-pill">📄 Docs: <span>{len(st.session_state['processed_docs'])}</span></div>
            <div class="stat-pill">🧩 Chunks: <span>{st.session_state['total_chunks']:,}</span></div>
            <div class="stat-pill">💬 Messages: <span>{len(st.session_state['chat_history'])}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

if st.session_state["chunk_previews"]:
    with st.expander("🔍 Document Chunk Preview (transparency panel)", expanded=False):
        for i, chunk in enumerate(st.session_state["chunk_previews"], 1):
            m = chunk.metadata
            st.markdown(
                f"""<div class="chunk-card">
                    <div class="chunk-meta">Chunk {i} · {m.get('source','?')} · Page {m.get('page','?')}</div>
                    {chunk.page_content[:420]}{'…' if len(chunk.page_content) > 420 else ''}
                </div>""",
                unsafe_allow_html=True,
            )

chat_container = st.container()
with chat_container:
    if not st.session_state["chat_history"]:
        if st.session_state["vector_store"]:
            st.markdown(
                """<div class="empty-state">
                    <div class="icon">💬</div>
                    <h3>Documents ready — start asking!</h3>
                    <p>Try asking for a summary, comparing documents,<br>or asking specific questions about the content.</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="empty-state">
                    <div class="icon">📂</div>
                    <h3>Upload & process documents to begin</h3>
                    <p>Supported: <strong>PDF · DOCX · TXT</strong><br>
                    Add your Groq API key, upload files, then click <em>Process Documents</em>.</p>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f"""<div class="bubble-user">
                        <div class="bubble-label">You · {msg.get('time','')}</div>
                        {msg['content']}
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                cite_html = ""
                if msg.get("citations"):
                    cite_html = f'<div class="citations-box">📌 Sources<br>{msg["citations"]}</div>'
                st.markdown(
                    f"""<div class="bubble-ai">
                        <div class="bubble-label">🧠 DocMind · {msg.get('time','')}</div>
                        {msg['content']}
                        {cite_html}
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
if st.session_state["vector_store"] and not st.session_state["chat_history"]:
    st.markdown(
        "<div style='font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;'>💡 Suggested prompts</div>",
        unsafe_allow_html=True,
    )
    for col, sug in zip(st.columns(3), [
        "Summarise all documents",
        "Compare the main themes",
        "List key facts and figures",
    ]):
        with col:
            if st.button(sug, use_container_width=True):
                st.session_state["_pending_query"] = sug
                st.rerun()

col_input, col_send = st.columns([9, 1])
with col_input:
    user_query = st.text_input(
        "Ask anything…", key="user_input",
        label_visibility="collapsed",
        placeholder="Ask anything about your documents…",
        value=st.session_state.pop("_pending_query", ""),
    )
with col_send:
    send_btn = st.button("➤", use_container_width=True)

if (send_btn or user_query) and user_query.strip():
    if not st.session_state["retrieval_chain"]:
        st.error("⚠️ Please process documents first.")
    elif not groq_api_key:
        st.error("🔑 Groq API key required.")
    else:
        ts = datetime.datetime.now().strftime("%H:%M")
        q  = user_query.strip()

        st.session_state["chat_history"].append({"role": "user", "content": q, "time": ts})

        with st.spinner("🧠 Thinking…"):
            try:
                result = st.session_state["retrieval_chain"].invoke({
                    "input":        q,
                    "chat_history": st.session_state["lc_messages"],
                })
                answer      = result.get("answer", "")
                source_docs = result.get("context", [])
                citations   = format_citations(source_docs)

                # Update LangChain message history for next turn
                st.session_state["lc_messages"].extend([
                    HumanMessage(content=q),
                    AIMessage(content=answer),
                ])
                # Keep context window manageable (last 10 exchanges)
                if len(st.session_state["lc_messages"]) > 20:
                    st.session_state["lc_messages"] = st.session_state["lc_messages"][-20:]

                st.session_state["chat_history"].append({
                    "role": "assistant", "content": answer,
                    "citations": citations,
                    "time": datetime.datetime.now().strftime("%H:%M"),
                })

            except Exception as e:
                err = str(e)
                if "API_KEY" in err.upper() or "invalid api key" in err.lower() or "401" in err:
                    label = " **Invalid API key.** Double-check your Groq key in the sidebar."
                elif "403" in err or "permission" in err.lower():
                    label = "**Permission denied.** Your key may not have Groq API access enabled."
                elif "quota" in err.lower() or "429" in err or "resource_exhausted" in err.lower():
                    label = " **Quota exceeded.** Wait a moment or check your Groq account plan."
                elif "model" in err.lower() and "not found" in err.lower():
                    label = " **Model not found.** Check your Groq model name or account settings."
                else:
                    label = "**Unexpected error.**"
                msg = f"{label}\n\n```\n{err}\n```"
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": msg, "citations": "", "time": ts}
                )

        st.rerun()
