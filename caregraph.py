import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from config import get_faiss_dir

_QA_CHAIN = None
_INIT_ERROR = None


class CareGraphResponse(BaseModel):
    disclaimer: str = Field(description="Medical disclaimer")
    rationale: str = Field(description="Reasoning grounded in guidelines")
    ok_report: str = Field(description="Concise clinical summary and next steps")


def _build_chain():
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

    faiss_dir = get_faiss_dir()
    if not os.path.exists(faiss_dir):
        raise FileNotFoundError(f"{faiss_dir} folder not found. Please run ingest.py first.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )

    parser = JsonOutputParser(pydantic_object=CareGraphResponse)

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
    - Output must be valid JSON and match this schema:
    {format_instructions}

    JSON RESPONSE:
    """

    prompt = ChatPromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions()
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
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
