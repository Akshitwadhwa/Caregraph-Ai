# CareGraph AI: Agentic Multimodal RAG for Clinical Support

**CareGraph AI** is a specialized Clinical Decision Support System (CDSS) designed to bridge the gap between messy patient data and gold-standard medical literature. Using an **Agentic RAG (Retrieval-Augmented Generation)** architecture, the system interprets patient lab reports (images/PDFs) and cross-references them with official clinical guidelines from the NIH, CDC, and ADA.

---

## 🚀 Key Features

*   **Multimodal Analysis:** Processes both text-based queries and visual lab reports (JPG/PNG) using **Google Gemini 1.5 Flash**.
*   **Privacy-First Architecture:** Utilizes **local HuggingFace embeddings** (`all-MiniLM-L6-v2`) to ensure semantic processing of medical data remains on the local environment.
*   **Grounded Reasoning:** Prevents hallucinations by forcing the LLM to derive answers strictly from the context provided by ingested medical guidelines.
*   **High-Performance Retrieval:** Uses **FAISS (Facebook AI Similarity Search)** for high-speed vector indexing, optimized for handling clinical documents in Python 3.14.
*   **Clinical Guardrails:** Implements specific system prompts that enforce medical disclaimers and prevent the AI from suggesting specific drug dosages.

---

## 🛠️ Tech Stack

*   **Orchestration:** [LangChain](https://github.com/langchain-ai/langchain)
*   **LLM (Reasoning & Vision):** Google Gemini 1.5 Flash
*   **Vector Database:** FAISS (Facebook AI Similarity Search)
*   **Embeddings:** HuggingFace Sentence-Transformers (Local)
*   **Data Ingestion:** PyPDF for high-fidelity clinical guideline parsing
*   **Environment:** Python 3.14 (Stable implementation)

---

## 🏗️ System Architecture

1.  **Ingestion Layer:** Clinical PDFs (Diabetes, Oncology, Heart Health) are parsed, split into 800-token chunks with overlap, and indexed via FAISS.
2.  **Vision Layer:** Patient lab results (images) are processed via Gemini's multimodal vision capabilities to extract structured biomarkers (e.g., A1C, Glucose, BMI).
3.  **Retrieval Layer:** The system performs a similarity search in the local vector store to find relevant diagnostic thresholds from official guidelines.
4.  **Synthesis Layer:** The LLM synthesizes a final clinical response, grounding every answer in the provided source material while maintaining a strict clinical persona.

---

## 📂 Project Structure

```text
CareGraph/
├── data/                # Source PDFs (NIH, CDC, ADA Guidelines)
├── faiss_index/         # Local Vector Store (Generated)
├── ingest.py            # Data pipeline & semantic indexing logic
├── main.py              # Text-based RAG reasoning engine
├── vision_main.py       # Multimodal Lab Report analysis engine
├── .env                 # API keys (Google Gemini)
└── requirements.txt     # Dependency list
