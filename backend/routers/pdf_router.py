from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
import pdfplumber 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.chroma_manager import ChromaManager
router = APIRouter()  

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""] 
)

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
            chroma.persist()

            return {
                "filename": os.path.basename(tmp_file_path),  
                "total_pages": len(pdf.pages),
                "chunks": [
                    {"id": i, "text": chunk} 
                    for i, chunk in enumerate(all_chunks)
                ]
            }  
    
    except pdfplumber.PDFSyntaxError:
        raise HTTPException(422, "PDF is encrypted or corrupted")
    except IOError as e:
        raise HTTPException(500, f"I/O Error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        await file.close()