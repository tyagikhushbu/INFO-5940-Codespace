
---

# ref-log.md

```markdown
# Reference Log (ref-log.md)

## External Libraries / Tools
- **LangChain** (`langchain`, `langchain-community`, `langchain-openai`)
  - Text splitting (`RecursiveCharacterTextSplitter`), vector store adapter (Chroma), retriever interface, chat model wrappers.
- **ChromaDB** (`chromadb`)
  - Vector database with persistence; built using `Chroma.from_documents(...)` and `persist_directory`.
- **Streamlit**
  - Chat UI (`st.chat_message`, `st.chat_input`), file uploader, sidebar controls.
- **pypdf**
  - Text extraction from **text-based** PDFs.
- **OpenAI** via `langchain-openai`
  - `ChatOpenAI` for chat; `OpenAIEmbeddings` for embeddings (if your key has access to the requested models).
- **Sentence Transformers** (`sentence-transformers`) *(optional but recommended fallback)*
  - Local embeddings (`all-MiniLM-L6-v2`) that avoid API dependency and unblock testing.

## Patterns / Docs Consulted
- LangChain retriever usage in versions ≥ 0.2 (`retriever.invoke(query)` instead of `.get_relevant_documents`).
- ChromaDB: persistence and simple from-documents flow.
- Streamlit chat UX patterns (`st.chat_message`, `st.chat_input`).

## Implementation Details (matching `chat_with_pdf.py`)
- **Chunking**: `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`.
- **Embeddings**: `OpenAIEmbeddings(model="openai.text-embedding-3-large")` (⚠️ see README Troubleshooting for correct OpenAI model names or local fallback).
- **Retriever**: `vector_db.as_retriever(search_kwargs={"k": 4})`, with multi-turn history forwarded to the LLM.
- **Grounding**: System prompt enforces “answer only from context,” with source attributions in the UI.
- **Persistence**: `persist_dir` defaults to `./.chroma`, set in the sidebar.

## GenAI Usage (What & Why)
- Used ChatGPT to help scaffold RAG app structure, ensure compatibility with newer LangChain retriever APIs (`.invoke()`), draft documentation (README + this reference log), and enumerate robust fallbacks for model/embedding issues.
- **Rationale**: speed up boilerplate and avoid common integration pitfalls. All code and docs were reviewed and adapted to match the exact `chat_with_pdf.py`.

## Known Caveats / Next Steps
- **Model access** varies by API key/org. If you see `Invalid model name`, pick a model your key exposes (for embeddings: `text-embedding-3-small/-large`; for chat: `gpt-4o`, etc.), or switch embeddings to local `sentence-transformers`.
- **OCR** is not included; scanned PDFs will not extract text.
- Consider adding a **model fallback utility** (e.g., try requested model, then fallback list) and/or **local embeddings by default** to make grading smoother.
