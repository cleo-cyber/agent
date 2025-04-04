from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

class RetrieverModule:
    def __init__(self ,embeddings: str):
        self.embeddings = embeddings

    def build_retriever(self,documents):
        """
        Create a retriever tool using FAISS.
        Args:
            documents (list): List of text chunks.
        Returns:
            ToolNode: The retriever tool.
        """
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        retriever_tool = create_retriever_tool(retriever, 
                                               "retrieve_research_content",
                                               "Search and return relevant content from the research document based on user query and do not extend or compress the response")
        return retriever_tool