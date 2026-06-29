import re
import pdfplumber
from agents.state import AuditState

# Patterns used for completeness checking
_PAYROLL_RE = re.compile(
    r'\$[\d,]+\.?\d*|wages?|salaries|salary|payroll|compensation|earnings?',
    re.IGNORECASE,
)
_NAME_RE = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')
_DATE_RE = re.compile(
    r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'
    r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
    re.IGNORECASE,
)


def intake_node(state: AuditState) -> AuditState:
    path = state.get("uploaded_file_path", "")
    if not path:
        return {**state, "error": "No file path provided.", "current_step": "intake"}

    try:
        raw_text = _extract_text(path)
    except Exception as exc:
        return {**state, "error": f"Failed to read file: {exc}", "current_step": "intake"}

    if not raw_text.strip():
        return {
            **state,
            "error": "Document is empty or contains no extractable text. "
                     "Scanned/image-only PDFs are not supported in this release.",
            "current_step": "intake",
        }

    flags = _check_completeness(raw_text)
    return {
        **state,
        "raw_text": raw_text,
        "completeness_flags": flags,
        "current_step": "intake",
    }


def _extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return _extract_pdf(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_pdf(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _check_completeness(text: str) -> dict:
    has_payroll = bool(_PAYROLL_RE.search(text))
    has_names = bool(_NAME_RE.search(text))
    has_dates = bool(_DATE_RE.search(text))

    missing = []
    if not has_payroll:
        missing.append("payroll data")
    if not has_names:
        missing.append("employee names")
    if not has_dates:
        missing.append("dates")

    # Critical = both payroll and names are absent (document is uninformative)
    critical_missing = not has_payroll and not has_names

    return {
        "has_payroll_data": has_payroll,
        "has_employee_names": has_names,
        "has_dates": has_dates,
        "missing_fields": missing,
        "critical_missing": critical_missing,
    }
