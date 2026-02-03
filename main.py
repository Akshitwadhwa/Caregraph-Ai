import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. SETUP API KEY
os.environ["GOOGLE_API_KEY"] = "AIzaSyB-FiviAAK4TBO4BnYk50oIlevra_f14ek"

def start_caregraph():
    print("--- Loading CareGraph reasoning engine ---")
    
    # 2. LOAD THE BRAIN (FAISS Index)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if the medical_db exists first
    if not os.path.exists("medical_db"):
        print("Error: medical_db folder not found. Please run ingest.py first!")
        return

    vector_db = FAISS.load_local("medical_db", embeddings, allow_dangerous_deserialization=True)
    
    # 3. INITIALIZE GEMINI (Using the full model path)
    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash", 
            temperature=0.1
        )
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return

    # 4. CREATE A CLINICAL PROMPT
    template = """
    You are 'CareGraph AI', a professional clinical decision support assistant.
    Use the following pieces of medical context to answer the user's question.
    
    CONTEXT FROM GUIDELINES:
    {context}
    
    USER QUESTION: 
    {question}
    
    INSTRUCTIONS:
    - If the context doesn't have the answer, say you don't know based on the current docs.
    - Start with a medical disclaimer.
    
    FINAL RESPONSE:
    """
    
    custom_prompt = ChatPromptTemplate.from_template(template)

    # 5. SETUP THE RETRIEVAL CHAIN using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | custom_prompt
        | llm
        | StrOutputParser()
    )

    # 6. RUN A TEST QUERY
    print("\nCareGraph is ready. Ask a question (e.g., 'What is diabetes?') or type 'exit'")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            print("\nCareGraph: Searching medical guidelines...")
            response = qa_chain.invoke(user_input)
            print(f"\n{response}")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Tip: Check if your API Key is valid and you have 'Generative AI' enabled in Google AI Studio.")

if __name__ == "__main__":
    start_caregraph()
