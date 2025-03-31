import os
from pypdf import PDFReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

HUGGINGFACE_API_KEY=os.getenv('HUGGINGFACEHUB_API_TOKEN')
EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL')

embeddings=HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device":"cpu"}
)

def extract_text(file):
    data=PDFReader(file)
    text=""
    for page in data.pages:
        try:
            text+=page.extract_text()
        except Exception as e:
            print(e)
    return text

def chunk_text(text):
    splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=splitter.split_text(text)

    return chunks
    

def store_to_vectordb(chunks):
    vectorstore=Chroma.from_documents(
        documents=chunks,
        collection_name="rag_chroma",
        embedding=embeddings
    )
    return vectorstore

def create_retriever():
    retriever=store_to_vectordb().as_retriever()
    