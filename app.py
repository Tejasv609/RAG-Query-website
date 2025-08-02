from fastapi import FastAPI, File, UploadFile, HTTPException
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceSplitter
import torch
from transformers import AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")  # Store HF Token in .env file

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LLaMA model (Use Mistral for local testing)
system_prompt = "You are a Q&A assistant. Answer based on the uploaded document."
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name='meta-llama/Llama-2-7b-chat-hf',  # Use a smaller model
    model_name='meta-llama/Llama-2-7b-chat-hf',
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16}
)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")


# Create ServiceContext
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=1024)
service_context = Settings

index = None
  # Global index object

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles file upload and indexing"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        global index
        documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)


        return {"message": f"File '{file.filename}' uploaded and indexed successfully."}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Handles querying of indexed documents"""
    global index

    print(f"Received question: {request.question}")

    if index is None:
        return {"error": "No document uploaded. Please upload a file first."}

    try:
        query_engine = index.as_query_engine()  # Get the query engine
        response = query_engine.query(request.question)

        print(f"Answer generated: {response}")
        return {"answer": str(response)}

    except Exception as e:
        print(f"ERROR during query: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)