"""
CareGraph AI — Phase 3: Multimodal Vision + RAG
Analyzes a patient lab report image, extracts biomarkers,
searches the FAISS vector database for clinical context,
and produces a combined medical reasoning response.
"""

import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from config import get_faiss_dir

# ── Config ────────────────────────────────────────────────────────────
IMAGE_PATH = "test_lab.jpg"
FAISS_DIR = get_faiss_dir()
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VISION_MODEL = "models/gemini-2.5-flash"
REASONING_MODEL = "models/gemini-2.5-flash"


def encode_image_to_base64(path: str) -> str:
    """Read a local image file and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_biomarkers(llm, image_b64: str) -> str:
    """Send the lab report image to Gemini Vision and extract biomarkers."""
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "You are a clinical lab-report reader. "
                    "Examine this blood test report image carefully and extract "
                    "every numerical biomarker you can find. "
                    "For each biomarker, return the name, value, and unit on its own line. "
                    "Example format:\n"
                    "  Glucose (Fasting): 126 mg/dL\n"
                    "  HbA1c: 7.2 %\n"
                    "If a reference range is visible, include it in parentheses. "
                    "Only output the extracted values, nothing else."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                },
            },
        ]
    )

    response = llm.invoke([message])
    return response.content


def search_faiss(query: str, k: int = 4):
    """Load the FAISS index and return the top-k relevant document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(
        FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )
    results = vector_db.similarity_search(query, k=k)
    return results


def build_reasoning_response(llm, lab_values: str, context_docs) -> str:
    """Combine extracted lab values with retrieved guidelines for a final response."""
    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    prompt = (
        "You are 'CareGraph AI', a clinical decision-support assistant.\n\n"
        "A patient's lab report has been analyzed and the following biomarkers "
        "were extracted:\n\n"
        f"{lab_values}\n\n"
        "Relevant excerpts from clinical guidelines:\n\n"
        f"{context_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Start with a brief medical disclaimer.\n"
        "2. Identify which biomarkers are abnormal and explain why.\n"
        "3. Cross-reference the abnormal values with the guideline excerpts.\n"
        "4. Provide a concise clinical summary and recommended next steps.\n"
        "5. If the guidelines do not cover a specific biomarker, state that clearly.\n\n"
        "RESPONSE:"
    )

    message = HumanMessage(content=prompt)
    response = llm.invoke([message])
    return response.content


def main():
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file '{IMAGE_PATH}' not found in project root.")
        return

    if not os.path.exists(FAISS_DIR):
        print(f"Error: FAISS index directory '{FAISS_DIR}' not found.")
        return

    # ── Step 1: Encode image ──────────────────────────────────────────
    print("--- Step 1: Encoding lab report image ---")
    image_b64 = encode_image_to_base64(IMAGE_PATH)
    print(f"  Image encoded ({len(image_b64)} chars of base64)")

    # ── Step 2: Extract biomarkers via Gemini Vision ──────────────────
    print("--- Step 2: Extracting biomarkers with Gemini Vision ---")
    vision_llm = ChatGoogleGenerativeAI(
        model=VISION_MODEL,
        temperature=0.1,
    )
    lab_values = extract_biomarkers(vision_llm, image_b64)
    print(f"\nExtracted Lab Values:\n{lab_values}\n")

    # ── Step 3: Search FAISS for relevant clinical guidelines ─────────
    print("--- Step 3: Searching FAISS vector database ---")
    docs = search_faiss(lab_values, k=4)
    print(f"  Retrieved {len(docs)} relevant guideline chunks\n")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] {source} — {doc.page_content[:80]}...")

    # ── Step 4: Combined reasoning ────────────────────────────────────
    print("\n--- Step 4: Generating clinical reasoning ---")
    reasoning_llm = ChatGoogleGenerativeAI(
        model=REASONING_MODEL,
        temperature=0.2,
    )
    final_response = build_reasoning_response(reasoning_llm, lab_values, docs)
    print(f"\n{'='*60}")
    print("CareGraph AI — Multimodal Vision + RAG Report")
    print(f"{'='*60}\n")
    print(final_response)


if __name__ == "__main__":
    main()
