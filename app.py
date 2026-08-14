import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage , HumanMessage


load_dotenv()

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("PDF Reader ChatBot")

uploaded_files = st.file_uploader(
    "Upload Your File Here",
    type=["pdf"],
    accept_multiple_files=True
)

all_documents = []

if uploaded_files:
    file_key = tuple(f.name for f in uploaded_files)

    if st.session_state.get("loaded_files_key") != file_key:
        loaded_documents = []
        for uploaded_file in uploaded_files:
            st.success(f"Uploaded: {uploaded_file.name} Successfully")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                pdf_path = temp_file.name

            loader = PyPDFLoader(pdf_path)
            pdf = loader.load()
            os.remove(pdf_path)  # cleanup temp file

            for doc in pdf:
                doc.metadata["source"] = uploaded_file.name

            loaded_documents.extend(pdf)

            st.write(f"📄 {uploaded_file.name}: {len(pdf)} pages loaded")

        st.session_state.all_documents = loaded_documents
        st.session_state.loaded_files_key = file_key

    all_documents = st.session_state.all_documents

    st.write(f"📚 Total pages loaded: {len(all_documents)}")




class JinaEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        self.url = "https://api.jina.ai/v1/embeddings"
        self.model = "jina-embeddings-v3"

    def embed_documents(self, texts):
        all_embeddings = []

        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]

            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": batch,
                    "task": "retrieval.passage",
                    "dimensions": 768
                },
                timeout=120
            )

            if not response.ok:
                print("Jina Status:", response.status_code)
                print("Jina Response:", response.text)

            response.raise_for_status()

            embeddings = [
                item["embedding"]
                for item in response.json()["data"]
            ]

            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text):
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": [text],
                "task": "retrieval.query",
                "dimensions": 768
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json()["data"][0]["embedding"]


@st.cache_resource
def create_vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embedding = JinaEmbeddings()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )

    return vector_store, len(chunks)

if all_documents:

    vector_store, chunk_count = create_vector_store(all_documents)

    st.write(f"Chunks created: {chunk_count}")

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    st.success("✅ Vector database created!")
    #st.audio("success.mp3", autoplay=True)

    

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template="""You are a document question-answering assistant.

Always check the provided context first.

If the information is found in the context:
- Answer using the context.
- Keep the answer concise.
- Do not add unnecessary information from your own knowledge.

If the information is NOT found in the context:
- Clearly say that the information was not found in the provided document.
- Then provide a general-knowledge answer.
- Clearly distinguish the general-knowledge answer from the document-based answer.

Always add available metadata at the end:

📖 Source: <source>
📌 Section: <section>
📄 Page: <page>

Do not repeat the page or section inside the answer.

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Section: {doc.metadata.get('section', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"{doc.page_content}"
            for doc in docs
    )

    parser = StrOutputParser()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | parser
    )

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    question = st.chat_input("Ask something about your PDF...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        history = "\n".join(
            f"User: {m.content}" if isinstance(m, HumanMessage)
            else f"Assistant: {m.content}"
            for m in st.session_state.chat_history
        )

        contextual_question_prompt = PromptTemplate(
        template="""You are a question rewriting assistant.

            Use the conversation history to understand the user's latest question.

            Rewrite the latest question into a clear, standalone question that preserves the user's meaning.

            If the latest question is a follow-up such as:
            - why is it important?
            - what are its types?
            - how does it work?
            - explain this
            - what about that?
            - ohh

            then use the previous conversation to identify what "it", "this", "that", or similar words refer to.

            If the latest question is already clear, return it unchanged.

            If the conversation does not contain enough information to understand the latest question,
            return the latest question unchanged. Do not invent a topic.

            Conversation history:
            {history}

            Latest question:
            {question}

            Return ONLY the standalone question.""",
        input_variables=["history", "question"]

            )

        rewrite_chain = (
            contextual_question_prompt
            | llm
            | StrOutputParser()
        )

        standalone_question = rewrite_chain.invoke({
            "history": history,
            "question": question
        })
        st.write("🔄 Rewritten:", standalone_question)
        answer = chain.invoke(standalone_question)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append(
            HumanMessage(content=question)
        )

        st.session_state.chat_history.append(
            AIMessage(content=answer)
        )