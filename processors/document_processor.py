from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = ""
        self.chunks = []
        self.documents = []

        print("File path:", self.file_path)
    
    print("DocumentProcessor initialized")
    
    def extract_text(self) -> str:
        """
        Extract text from the PDF file.
        Returns:
            str: The extracted text.
        """
        print("Extracting text from PDF")

        data = PdfReader(self.file_path)
        for page in data.pages:
            try:
                self.text += page.extract_text()
            except Exception as e:
                print(e)
        if not self.text:
            raise ValueError("No text found in the PDF file.")
        print("Text extraction complete")
        print(f"Extracted text: {self.text[:100]}...")
        
        return self.text
    
    def chunk_text(self) -> list:
        """
        Chunk the extracted text into smaller pieces.
        Returns:
            list: List of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.chunks = splitter.split_text(self.text)
        self.documents = [Document(page_content=chunk) for chunk in self.chunks]
        return self.documents
    