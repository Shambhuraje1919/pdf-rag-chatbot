from dotenv import load_dotenv
import os
import requests
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings

load_dotenv()

loader = PyPDFLoader("project/Programming PyTorch for Deep Learning (2020).pdf")
pdf = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(pdf)

class JinaEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        self.url = "https://api.jina.ai/v1/embeddings"
        self.model = "jina-embeddings-v3"

    def embed_documents(self, texts):
        all_embeddings = []

        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]

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

        if not response.ok:
            print("Status:", response.status_code)
            print("Response:", response.text)

        response.raise_for_status()

        return response.json()["data"][0]["embedding"]

vector_store = FAISS.from_documents(
    embedding=JinaEmbeddings(),
    documents=chunks
)

retriver = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate(
    template="""You are a document question-answering assistant.

Always add the page number or topic name if available in metadata.

Warning: Don't add too much information from your own knowledge. Always check the context first and try to answer from that context.

If the information is not available in the context, clearly say that it was not found in the provided document, and then you may answer using your general knowledge.

Also add the metadata as a reference from the book.

Always add metadata at the end using new lines:

📖 Source: ()
📌 Section: ()
📄 Page:

Do not repeat the page or section inside the answer and then again in the metadata. Keep the answer clean and show the citation separately.

Context:
{context}

Question:
{que}

Answer:""",
    input_variables=["context", "que"]
)

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source')}, "
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

chain = (
    {
        "context": retriver | format_docs,
        "que": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

print(chain.invoke("What is Neural Network?"))
print(chain.invoke("Who is Modi?"))