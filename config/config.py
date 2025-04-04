import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    """Configuration class for the application."""
    print("Loading environment variables...")

    
    HUGGINGFACE_API_KEY=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL')
    GROQ_API_KEY=os.getenv('GQOQ_API_KEY')
    MODEL_NAME=os.getenv('MODEL_NAME')

    os.environ['LANGSMITH_ENDPOINT']=os.getenv('LANGSMITH_ENDPOINT')
    os.environ['LANGSMITH_API_KEY']=os.getenv('LANGSMITH_API_KEY')
    os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')
    os.environ['GQOQ_API_KEY']=os.getenv('GQOQ_API_KEY')
    os.environ['TAVILY_API_KEY']=os.getenv('TAVILY_API_KEY')
    