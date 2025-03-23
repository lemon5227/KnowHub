from fastapi import APIRouter, UploadFile, File, HTTPException,Query
import tempfile
import os
import pdfplumber 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.chroma_manager import ChromaManager
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq




router = APIRouter()  

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""] 
)


@router.get("/ask")
async def ask(question:str=Query(...,description="The question to ask"),
              k: int = Query(3,description="The number of blocks to query")
              ):
    try:
        #Initialization Retriever and LLM
        chroma = ChromaManager()
        retriever = chroma.get_retriever(k=k)

        #groq model
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192",  
        )
        
        # build the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            input_key="query"
        )

        #Ask the question
        result = qa_chain.invoke({"query": question})

        #handle possible null values
        sources = []
        if "source_documents" in result:
            sources = list({doc.metadata.get("source", "") for doc in result["source_documents"]})
        
        return {
            "question": question,
            "answer": result.get("result", "no answer"),
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(500, f"Answer error: {str(e)}")

@router.post("/upload")  
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    tmp_file_path = None
    all_chunks = [] 
    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            while content := await file.read(1024 * 1024):  # 1MB分块读取
                tmp.write(content)
            tmp_file_path = tmp.name
        
        with pdfplumber.open(tmp_file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks = text_splitter.split_text(text)
                    all_chunks.extend(chunks)
            chroma = ChromaManager()
            chroma.add_texts(all_chunks, source=file.filename)


            return {
                "filename": os.path.basename(tmp_file_path),  
                "total_pages": len(pdf.pages),
                "chunks": [
                    {"id": i, "text": chunk} 
                    for i, chunk in enumerate(all_chunks)
                ]
            }  
    
    except IOError as e:
        raise HTTPException(500, f"I/O Error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        await file.close()