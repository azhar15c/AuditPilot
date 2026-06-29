import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.workflow import run_workflow

app = FastAPI(title="AuditPilot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ALLOWED_TYPES = {"application/pdf", "text/plain"}
_ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    message = str(exc)

    if _is_rate_limit(message):
        return JSONResponse(
            status_code=429,
            content={
                "error": "HuggingFace rate limit reached.",
                "detail": (
                    "The free-tier inference API has hit its rate limit. "
                    "Wait a few minutes and try again, or upgrade your HF plan."
                ),
            },
        )

    if _is_model_loading(message):
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model is loading.",
                "detail": (
                    "The model is warming up on HuggingFace Serverless. "
                    "This typically takes 20–60 seconds on first request. Please retry."
                ),
            },
        )

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error.", "detail": message},
    )


@app.post("/audit/run")
async def audit_run(file: UploadFile = File(...)) -> JSONResponse:
    _validate_file(file)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = os.path.splitext(file.filename or "upload")[1].lower() or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        state = run_workflow(tmp_path)
    finally:
        os.unlink(tmp_path)

    if state.get("error"):
        raise HTTPException(status_code=422, detail=state["error"])

    return JSONResponse({
        "entities": state.get("entities", []),
        "employee_records": state.get("employee_records", []),
        "ncci_suggestions": state.get("ncci_suggestions", []),
        "audit_report": state.get("audit_report", ""),
        "completeness_flags": state.get("completeness_flags", {}),
    })


@app.get("/audit/health")
async def audit_health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "models": ["bert-NER", "bge-large", "llama-3.3-70b"],
    })


def _validate_file(file: UploadFile) -> None:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.content_type not in _ALLOWED_TYPES and ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a PDF or plain text file.",
        )


def _is_rate_limit(message: str) -> bool:
    lower = message.lower()
    return any(p in lower for p in ("rate limit", "429", "too many requests", "quota"))


def _is_model_loading(message: str) -> bool:
    lower = message.lower()
    return any(p in lower for p in ("loading", "model is currently loading", "503"))


from app import demo   # noqa: E402  (app.py applies HfFolder + gradio_client patches first)
import gradio as gr    # noqa: E402  (gradio already cached; this is a no-op)

gr.mount_gradio_app(app, demo, path="/")
