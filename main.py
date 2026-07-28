from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings # Lub OpenAIEmbeddings/HuggingFace

class KnowledgeAgent:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.vector_store = None

    def ingest_document(self, file_path: str):
        loader = TextLoader(file_path)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        # Mock embeddings to demonstrate pipeline structure
        embeddings = FakeEmbeddings(size=384)
        self.vector_store = Chroma.from_documents(chunks, embeddings)
        return len(chunks)

    def query(self, prompt: str, k: int = 3):
        if not self.vector_store:
            raise ValueError("No documents ingested yet.")
        results = self.vector_store.similarity_search(prompt, k=k)
        return [doc.page_content for doc in results]

if __name__ == "__main__":
    agent = KnowledgeAgent()
    print("Knowledge Agent initialized successfully.")
