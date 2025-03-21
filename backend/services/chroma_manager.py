from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
class ChromaManager:
    def __init__(self,persist_directory: str = "./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            collection_name="knowledge_base",
            embedding_function=self.embeddings,
        )
    
    def add_texts(self,texts:List[str],source:str) -> List[str]:
       docs = [Document(page_content=t, metadata={"source": source}) for t in texts]
       return self.vectordb.add_texts(texts)
    
    def persist(self):
        self.vectordb.persist()


    def get_retriever(self,k: int = 3):
        return self.vectordb.as_retriever(search_kwargs={"k": k})