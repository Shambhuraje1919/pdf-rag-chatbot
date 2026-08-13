from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

loader = PyPDFLoader("Applied DL PyTorch.pdf")
pdf = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(pdf)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    embedding=embedding,
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