# test_chroma.py
from chroma_manager import ChromaManager

def test_chroma():
    manager = ChromaManager()
    
    # 添加文本
    texts = ["Machine learning is fun", "Python is a popular language"]
    doc_ids = manager.add_texts(texts)
    print("Added IDs:", doc_ids)  # 应输出类似 ['xxxx','yyyy']
    
    # 检索测试
    retriever = manager.get_retriever(k=1)
    results = retriever.get_relevant_documents("What is Python?")
    print("Top result:", results[0].page_content)  # 应返回 "Python is a popular language"

if __name__ == "__main__":
    test_chroma()