from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from caregraph import get_chain

app = FastAPI(title="CareGraph AI")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


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
        return {"answer": answer}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
