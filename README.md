<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=00D9FF&center=true&vCenter=true&width=650&lines=PDF+RAG+Chatbot+%F0%9F%93%84%F0%9F%A4%96;Chat+with+any+PDF+using+LLMs;Multi-Document+%7C+Conversational+%7C+Cited+Answers;Built+with+LangChain+%2B+FAISS+%2B+Groq" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-openai%2Fgpt--oss--20b-F55036?style=for-the-badge)](https://groq.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)

[![Stars](https://img.shields.io/github/stars/Shambhuraje1919/pdf-rag-chatbot?style=social)](https://github.com/Shambhuraje1919/pdf-rag-chatbot/stargazers)
[![Forks](https://img.shields.io/github/forks/Shambhuraje1919/pdf-rag-chatbot?style=social)](https://github.com/Shambhuraje1919/pdf-rag-chatbot/network/members)
[![Last Commit](https://img.shields.io/github/last-commit/Shambhuraje1919/pdf-rag-chatbot?color=blue)](https://github.com/Shambhuraje1919/pdf-rag-chatbot/commits/main)

**Answers stay grounded in the source PDF, with page-level citations — and a clearly labeled general-knowledge fallback when the answer isn't in the document.**

[Live Demo](https://pdf-rag-chatbot-shambhuraje.streamlit.app/#pdf-reader-chat-bot) · [Report Bug](https://github.com/Shambhuraje1919/pdf-rag-chatbot/issues) · [Request Feature](https://github.com/Shambhuraje1919/pdf-rag-chatbot/issues)

</div>

<br/>

## 📖 About

**PDF RAG Chatbot** is a multi-document Retrieval-Augmented Generation application that lets you upload one or more PDFs and have a natural, cited conversation with their contents. It combines a `LangChain` retrieval pipeline, `FAISS` vector search, and Groq's blazing-fast `openai/gpt-oss-20b` inference behind a clean `Streamlit` interface.

Every answer comes with a **page-accurate citation**, and follow-up questions are automatically reformulated into standalone queries so the retriever never loses context mid-conversation.

<br/>

## ✨ Features

| | |
|---|---|
| 📚 **Multi-Document Ingestion** | Upload and query across multiple PDFs in a single session |
| 🔍 **Semantic Search** | `all-MiniLM-L6-v2` embeddings + FAISS for fast, relevant chunk retrieval |
| 💬 **Conversational Memory** | A query-reformer rewrites follow-up questions into standalone queries using chat history |
| 📌 **Accurate Citations** | Fixed 0-index/1-index page bug — citations point to the *exact* source page |
| 🚫 **Honest Abstention + Fallback** | States clearly when an answer isn't in the document, then offers a clearly-labeled general-knowledge answer instead of leaving you stuck |
| ⚡ **Fast Inference** | Powered by Groq's `openai/gpt-oss-20b` for near-instant responses |
| 🐳 **Docker Ready** | Includes a `Dockerfile` for containerized deployment |

<br/>

## 🏗️ How It Works

```mermaid
flowchart LR
    A[📄 Upload PDF] --> B[PyPDFLoader]
    B --> C[RecursiveCharacterTextSplitter]
    C --> D[MiniLM-L6-v2 Embeddings]
    D --> E[(FAISS Vector Store)]
    F[💬 User Question] --> G{Follow-up?}
    G -- Yes --> H[Query Reformer<br/>uses chat history]
    G -- No --> I[Standalone Query]
    H --> I
    I --> E
    E --> J[Top-k Relevant Chunks]
    J --> K{Found in Document?}
    K -- Yes --> L[Groq · openai/gpt-oss-20b]
    L --> M[✅ Answer + Page Citation]
    K -- No --> N[⚠️ Not found in document]
    N --> O[General-knowledge answer<br/>clearly labeled as such]
```

<br/>

## 🧪 Accuracy Testing

Retrieval and citation accuracy weren't just assumed — they were measured. A 10-page synthetic policy document was used as a controlled test corpus, and 16 questions spanning three categories were run against the deployed app.

<div align="center">

| Category | Description | Result |
|---|---|---|
| Single-chunk | Answer lives in one section | ✅ Pass |
| Multi-hop | Answer spans two sections | ✅ Pass |
| Abstention | Answer deliberately absent from doc | ✅ Correctly flagged as not found |

### **16 / 16 correct answers · 16 / 16 correct citations · 100% accuracy**

</div>

A real bug was caught in the process: page numbers from `PyPDFLoader` are 0-indexed and section names weren't stored in metadata, so citations were silently falling back to LLM guesses. This was fixed by 1-indexing pages and extracting real section headings into the chunk metadata — all results above reflect the app *after* the fix.

<br/>

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/-FAISS-0467DF?style=flat-square)
![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Groq](https://img.shields.io/badge/-Groq-F55036?style=flat-square)
![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

<br/>

## 🚀 Getting Started

Just want to try it out? Skip all of this and use the **[Live Demo](https://pdf-rag-chatbot-shambhuraje.streamlit.app/#pdf-reader-chat-bot)** — no setup or API key needed.

### Prerequisites
- Python 3.10+

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Shambhuraje1919/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** and start uploading PDFs.

<br/>

## 📂 Project Structure

```
pdf-rag-chatbot/
├── app.py                  # Streamlit UI & chat orchestration
├── rag.py                  # RAG pipeline: loading, chunking, embedding, retrieval
├── testing/                 # RAG accuracy testing suite (16-question eval)
│   ├── TechNova_Employee_Handbook.pdf
│   ├── personalized_results.csv
│   └── RAG_Accuracy_Report.pdf
├── requirements.txt
├── Dockerfile
└── .devcontainer/
```

<br/>

## 🗺️ Roadmap

- [ ] Support for DOCX / TXT ingestion
- [ ] Persistent vector store across sessions
- [ ] Streaming token-by-token responses
- [ ] Multi-user / multi-session isolation

Contributions and suggestions are welcome — open an [issue](https://github.com/Shambhuraje1919/pdf-rag-chatbot/issues) or a PR!

<br/>

## 👤 Author

**Shambhuraje Jagadale**
B.Tech (E&TC) · Ex LLM Post-Training Intern @ Ethara AI · Kaggle Expert

[![GitHub](https://img.shields.io/badge/-Shambhuraje1919-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Shambhuraje1919)
[![Email](https://img.shields.io/badge/-shambhurajejagadale%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:shambhurajejagadale@gmail.com)

<br/>

<div align="center">

⭐ **If this project helped you, consider giving it a star!** ⭐

</div>
