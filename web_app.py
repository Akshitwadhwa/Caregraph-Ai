import shutil
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from caregraph import CareGraphResponse, get_chain
from ingest import build_medical_knowledge_base
from vision_main import encode_image_to_base64, extract_biomarkers, search_faiss

app = FastAPI(title="CareGraph AI")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
DATA_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask")
async def ask(payload: dict):
    question = (payload or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Please enter a question."}, status_code=400)

    try:
        chain = get_chain()
        answer = chain.invoke(question)
        if isinstance(answer, dict):
            return answer
        return JSONResponse({"error": "Model did not return valid JSON."}, status_code=500)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    stamp = int(time.time())
    return f"{stamp}_{name}"


def _build_vision_json_response(lab_values: str, docs) -> dict:
    parser = JsonOutputParser(pydantic_object=CareGraphResponse)
    prompt = (
        "You are 'CareGraph AI', a clinical decision-support assistant.\n\n"
        "A patient's lab report has been analyzed and the following biomarkers "
        "were extracted:\n\n"
        f"{lab_values}\n\n"
        "Relevant excerpts from clinical guidelines:\n\n"
        + "\n\n---\n\n".join(doc.page_content for doc in docs)
        + "\n\nINSTRUCTIONS:\n"
        "1. Start with a brief medical disclaimer.\n"
        "2. Identify which biomarkers are abnormal and explain why.\n"
        "3. Cross-reference the abnormal values with the guideline excerpts.\n"
        "4. Provide a concise clinical summary and recommended next steps.\n"
        "5. If the guidelines do not cover a specific biomarker, state that clearly.\n"
        "6. Output must be valid JSON and match this schema:\n"
        f"{parser.get_format_instructions()}\n\n"
        "JSON RESPONSE:"
    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.2,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return parser.parse(response.content)


@app.post("/api/upload")
async def upload(
    image: Optional[UploadFile] = File(default=None),
    pdfs: Optional[List[UploadFile]] = File(default=None),
):
    if not image and not pdfs:
        return JSONResponse({"error": "Please upload a PDF and/or an image."}, status_code=400)

    saved_pdfs = []
    if pdfs:
        for pdf in pdfs:
            if not pdf.filename.lower().endswith(".pdf"):
                return JSONResponse({"error": "Only PDF files are allowed."}, status_code=400)
            filename = _safe_filename(pdf.filename)
            target = DATA_DIR / filename
            with target.open("wb") as f:
                shutil.copyfileobj(pdf.file, f)
            saved_pdfs.append(str(target))

    if saved_pdfs:
        try:
            build_medical_knowledge_base()
        except Exception as exc:
            return JSONResponse({"error": f"Ingest failed: {exc}"}, status_code=500)

    if image:
        if not image.content_type or not image.content_type.startswith("image/"):
            return JSONResponse({"error": "Only image files are allowed."}, status_code=400)
        filename = _safe_filename(image.filename or "report.jpg")
        target = UPLOAD_DIR / filename
        with target.open("wb") as f:
            shutil.copyfileobj(image.file, f)

        try:
            image_b64 = encode_image_to_base64(str(target))
            vision_llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash",
                temperature=0.1,
            )
            lab_values = extract_biomarkers(vision_llm, image_b64)
            docs = search_faiss(lab_values, k=4)
            answer = _build_vision_json_response(lab_values, docs)
            return answer
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    return {"status": "PDFs uploaded and indexed successfully."}
