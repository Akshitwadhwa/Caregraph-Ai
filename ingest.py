import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from config import get_faiss_dir

def build_medical_knowledge_base():
    # 1. Initialize Free Local Embeddings
    print("--- Initializing Embeddings (HuggingFace) ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Setup Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    all_docs = []
    data_dir = "./data"
    
    # Check if folder exists
    if not os.path.exists(data_dir):
        print(f"Error: The folder '{data_dir}' does not exist. Please create it.")
        return

    # 3. Process each PDF
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files:
        print("No PDFs found in the data folder!")
        return

    for file in files:
        try:
            print(f"Reading: {file}...")
            file_path = os.path.join(data_dir, file)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Split the text into chunks
            chunks = text_splitter.split_documents(docs)
            all_docs.extend(chunks)
            print(f"Successfully processed {file} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"Could not read {file}: {e}")

    # 4. Create and persist the Vector Database
    if all_docs:
        print(f"--- Creating Vector Database with {len(all_docs)} total chunks ---")
        vector_db = FAISS.from_documents(
            documents=all_docs,
            embedding=embeddings
        )
        faiss_dir = get_faiss_dir()
        vector_db.save_local(faiss_dir)
        print(f"--- Success! {faiss_dir} folder created. ---")
    else:
        print("No document content found to index.")

if __name__ == "__main__":
    build_medical_knowledge_base()
