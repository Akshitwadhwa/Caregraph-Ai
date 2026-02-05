import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

_QA_CHAIN = None
_INIT_ERROR = None


def _build_chain():
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

    if not os.path.exists("medical_db"):
        raise FileNotFoundError("medical_db folder not found. Please run ingest.py first.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("medical_db", embeddings, allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )

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

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


def get_chain():
    global _QA_CHAIN, _INIT_ERROR
    if _QA_CHAIN is not None:
        return _QA_CHAIN
    if _INIT_ERROR is not None:
        raise _INIT_ERROR

    try:
        _QA_CHAIN = _build_chain()
        return _QA_CHAIN
    except Exception as exc:
        _INIT_ERROR = exc
        raise
