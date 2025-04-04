
from langchain_groq import ChatGroq
from graph.rag_agent import AgentBuilder
from retievers.retriever import RetrieverModule
from processors.document_processor import DocumentProcessor    
from langchain_huggingface import HuggingFaceEmbeddings

class RAGOrchestration:
    def __init__(self,config):
        self.config=config
        self.llm=ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name=config.MODEL_NAME, streaming=True,temperature=0)
        self.embeddings=HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device":"cpu"}
        )
    print("RAG Orchestration initialized")

    def run(self)->str:
        """
        Run the RAG pipeline with the provided query.
        Args:
           None
        Returns:
            str: The response from the LLM.
        """

        print("Running RAG Orchestration")
        doc_processor=DocumentProcessor("test file")

        text=doc_processor.extract_text()

        chunks=doc_processor.chunk_text()

        retriever_tool=RetrieverModule(self.embeddings).build_retriever(chunks)

        agent=AgentBuilder(self.llm,retriever_tool)

        graph=agent.create_graph()

        return graph