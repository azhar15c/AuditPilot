import os
import pandas as pd

# huggingface_hub >=0.17 removed HfFolder. Gradio 4.x oauth.py still imports
# it, so inject a stub before `import gradio` runs. No-op when Gradio 5 is
# installed because Gradio 5 doesn't reference HfFolder at all.
import huggingface_hub as _hfhub
if not hasattr(_hfhub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(_t): pass
        @staticmethod
        def delete_token(): pass
    _hfhub.HfFolder = _HfFolder

import gradio as gr

# Gradio 4.x had a bug where json_schema_to_python_type crashed on
# additionalProperties=True (bool). Guard it in case it still exists.
try:
    import gradio_client.utils as _gcu
    _orig_schema_to_type = _gcu.json_schema_to_python_type
    def _safe_schema_to_type(schema, defs=None):
        try:
            return _orig_schema_to_type(schema, defs)
        except TypeError:
            return "Any"
    _gcu.json_schema_to_python_type = _safe_schema_to_type
except Exception:
    pass

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_payroll_register.pdf")

_AUDITOR_INSTRUCTIONS = """
### Before Finalising This Worksheet

1. **Verify NCCI codes** — Cross-check each suggested code against the NCCI Basic Manual for your state. Codes marked LOW confidence require manual review before submission.
2. **Confirm wages** — Validate reported wages against the IRS 941 and payroll register. Document any discrepancy.
3. **Subcontractor COIs** — Any subcontractor without a valid Certificate of Insurance must have their payments reclassified as insured payroll, which increases premium.
4. **Completeness flags** — Resolve any issues noted in the Document Intake tab before submitting the audit.
5. **This is a draft** — AI suggestions are decision support only. The auditor of record is responsible for the final worksheet.
"""

_EMPTY_DF = pd.DataFrame(
    columns=["name", "job_title", "suggested_ncci_code", "classification", "payroll", "confidence", "rationale"]
)


# ── Event handlers ────────────────────────────────────────────────────────────

def _run_pipeline(filepath: str, progress) -> tuple:
    """Call the LangGraph pipeline directly — no HTTP roundtrip."""
    try:
        from agents.workflow import run_workflow
        progress(0.3, desc="Running intake and extraction...")
        state = run_workflow(filepath)
        if state.get("error"):
            gr.Warning(f"Pipeline error: {state['error']}")
        progress(0.9, desc="Building results...")
        result = _build_dataframe(state), state.get("audit_report", "")
        progress(1.0, desc="Done.")
        return result
    except Exception as exc:
        gr.Warning(f"Pipeline failed: {exc}")
        return _EMPTY_DF.copy(), ""


def run_audit(file, progress=gr.Progress()):
    if file is None:
        gr.Warning("Please upload a file before running the audit.")
        return _EMPTY_DF.copy(), ""
    progress(0.1, desc="Uploading document...")
    # Gradio 5 returns filepath as str; Gradio 4 returned a file-like object
    filepath = file if isinstance(file, str) else file.name
    return _run_pipeline(filepath, progress)


def run_sample_audit(progress=gr.Progress()):
    progress(0.1, desc="Loading sample document...")
    return _run_pipeline(SAMPLE_PATH, progress)


def export_csv(df):
    if df is None or df.empty:
        gr.Warning("No classification data to export.")
        return None
    path = "/tmp/auditpilot_classifications.csv"
    df.to_csv(path, index=False)
    return path


def generate_report_pdf(report_text: str, df):
    if not report_text and (df is None or df.empty):
        gr.Warning("Run an audit first before downloading the report.")
        return None

    import tempfile
    from datetime import datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors as rl
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle,
        Paragraph, Spacer, HRFlowable,
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    RED    = rl.HexColor("#CC0000")
    DARK   = rl.HexColor("#1E2428")
    MGRAY  = rl.HexColor("#4A5568")
    LGRAY  = rl.HexColor("#F2F4F7")
    BORDER = rl.HexColor("#CBD5E0")
    BLACK  = rl.HexColor("#1A202C")

    PW, PH = letter
    M = 0.75 * inch
    CONTENT_W = PW - 2 * M

    def _s(name, **kw):
        return ParagraphStyle(name=name, **kw)

    def _p(text, s):
        return Paragraph(text, s)

    def _section(label):
        cell = _p(f"  {label}", _s("SB", fontName="Helvetica-Bold", fontSize=8, textColor=rl.white))
        t = Table([[cell]], colWidths=[CONTENT_W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), MGRAY),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    def _page_deco(c, doc):
        c.saveState()
        c.setFillColor(RED)
        c.rect(0, PH - 5, PW, 5, fill=1, stroke=0)
        c.setFont("Helvetica", 6.5)
        c.setFillColor(MGRAY)
        c.drawCentredString(
            PW / 2, 0.32 * inch,
            f"DRAFT — AI-generated. Auditor of record must review before submission.  "
            f"Generated: {datetime.now().strftime('%m/%d/%Y %H:%M')}"
        )
        c.drawRightString(PW - M, 0.32 * inch, f"Page {doc.page}")
        c.restoreState()

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = Table([[
        [
            _p("<b>AuditPilot</b>", _s("AP", fontName="Helvetica-Bold", fontSize=16, textColor=RED)),
            _p("AI-Assisted Workers' Compensation Premium Audit",
               _s("AP2", fontName="Helvetica", fontSize=8, textColor=MGRAY)),
        ],
        [
            _p("<b>PREMIUM AUDIT WORKSHEET</b>",
               _s("WT", fontName="Helvetica-Bold", fontSize=12, textColor=DARK, alignment=TA_RIGHT)),
            _p(f"DRAFT — {datetime.now().strftime('%B %d, %Y')}",
               _s("WD", fontName="Helvetica-Bold", fontSize=9, textColor=RED, alignment=TA_RIGHT)),
        ],
    ]], colWidths=[CONTENT_W * 0.55, CONTENT_W * 0.45])
    hdr.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story += [hdr, HRFlowable(width="100%", thickness=2, color=RED, spaceAfter=10)]

    # ── Classification table ──────────────────────────────────────────────────
    if df is not None and not df.empty:
        story += [_section("EMPLOYEE CLASSIFICATION RESULTS"), Spacer(1, 6)]

        col_w = [CONTENT_W * w for w in [0.14, 0.12, 0.08, 0.15, 0.10, 0.08, 0.33]]

        def _th(t):
            return _p(t, _s("TH", fontName="Helvetica-Bold", fontSize=7,
                            textColor=rl.white, alignment=TA_CENTER))
        def _td(t):
            return _p(str(t or ""), _s("TD", fontName="Helvetica", fontSize=8, textColor=BLACK, leading=11))

        rows = [[_th("EMPLOYEE"), _th("JOB TITLE"), _th("NCCI CODE"),
                 _th("CLASSIFICATION"), _th("PAYROLL"), _th("CONF."), _th("RATIONALE")]]
        for _, row in df.iterrows():
            rows.append([
                _td(row.get("name", "")),
                _td(row.get("job_title", "")),
                _td(row.get("suggested_ncci_code", "")),
                _td(row.get("classification", "")),
                _td(row.get("payroll", "")),
                _td(row.get("confidence", "")),
                _td(row.get("rationale", "")),
            ])

        row_bg = [
            ("BACKGROUND", (0, i), (-1, i), LGRAY if i % 2 == 0 else rl.white)
            for i in range(1, len(rows))
        ]
        cl = Table(rows, colWidths=col_w, repeatRows=1)
        cl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), DARK),
            ("LINEBELOW",     (0, 0), (-1,  0), 1.5, RED),
            ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            *row_bg,
        ]))
        story += [cl, Spacer(1, 14)]

    # ── Audit narrative ───────────────────────────────────────────────────────
    if report_text and report_text.strip():
        story += [_section("AUDIT NARRATIVE"), Spacer(1, 6)]

        import re as _re
        head  = _s("NH", fontName="Helvetica-Bold", fontSize=10, textColor=DARK, leading=14, spaceBefore=6)
        body  = _s("NB", fontName="Helvetica",      fontSize=9,  textColor=BLACK, leading=14)
        bullet = _s("BU", fontName="Helvetica",     fontSize=9,  textColor=BLACK, leading=14, leftIndent=12)

        for line in report_text.strip().split("\n"):
            stripped = line.strip()
            if not stripped or stripped in ("---", "***"):
                story.append(Spacer(1, 4))
            elif stripped.startswith("# "):
                story.append(_p(stripped.lstrip("# ").strip(), head))
            elif stripped.startswith("## "):
                story.append(_p(stripped.lstrip("# ").strip(), head))
            elif stripped.startswith("### "):
                story.append(_p(stripped.lstrip("# ").strip(), head))
            elif stripped.startswith("|"):
                # skip markdown table rows — classification table already rendered above
                continue
            elif stripped.startswith("- ") or stripped.startswith("* "):
                clean = _re.sub(r'\*\*(.+?)\*\*', r'\1', stripped[2:])
                story.append(_p(f"•  {clean}", bullet))
            elif _re.match(r'^\d+\.', stripped):
                clean = _re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)
                story.append(_p(clean, bullet))
            else:
                clean = _re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)
                clean = _re.sub(r'\*(.+?)\*',     r'\1', clean)
                story.append(_p(clean, body))

        story.append(Spacer(1, 14))

    # ── Auditor checklist ─────────────────────────────────────────────────────
    story += [_section("AUDITOR REVIEW CHECKLIST"), Spacer(1, 6)]
    chk = _s("Chk", fontName="Helvetica", fontSize=9, textColor=BLACK, leading=17)
    for item in [
        "☐  Verify NCCI codes against the NCCI Basic Manual for Texas. LOW confidence codes require manual review.",
        "☐  Validate reported wages against IRS 941 and the source payroll register. Document any discrepancy.",
        "☐  Subcontractor COIs — Any missing COI requires payments to be reclassified as insured payroll.",
        "☐  Resolve all completeness flags before submitting the audit.",
        "☐  Auditor of record signature required below before this worksheet is submitted.",
    ]:
        story.append(_p(item, chk))

    story.append(Spacer(1, 20))

    sig = Table([[
        _p("Auditor of Record: _______________________________",
           _s("SG", fontName="Helvetica", fontSize=9, textColor=MGRAY)),
        _p("Date: _______________",
           _s("SG2", fontName="Helvetica", fontSize=9, textColor=MGRAY, alignment=TA_RIGHT)),
        _p("License #: _______________",
           _s("SG3", fontName="Helvetica", fontSize=9, textColor=MGRAY, alignment=TA_RIGHT)),
    ]], colWidths=[CONTENT_W * 0.5, CONTENT_W * 0.25, CONTENT_W * 0.25])
    sig.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN",        (0, 0), (-1, -1), "BOTTOM"),
    ]))
    story.append(sig)

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", prefix="auditpilot_report_", dir="/tmp"
    )
    tmp_path = tmp.name
    tmp.close()

    doc = SimpleDocTemplate(
        tmp_path, pagesize=letter,
        leftMargin=M, rightMargin=M,
        topMargin=0.75 * inch, bottomMargin=0.65 * inch,
    )
    doc.build(story, onFirstPage=_page_deco, onLaterPages=_page_deco)
    return tmp_path


# ── Helpers ───────────────────────────────────────────────────────────────────



def _build_dataframe(data: dict) -> pd.DataFrame:
    records     = data.get("employee_records", [])
    suggestions = data.get("ncci_suggestions", [])
    by_name     = {s["employee"]: s for s in suggestions}

    rows = []
    for r in records:
        name = r.get("name", "")
        s    = by_name.get(name, {})
        job  = r.get("job_description") or []
        rows.append({
            "name":                name,
            "job_title":           ", ".join(job) if isinstance(job, list) else str(job),
            "suggested_ncci_code": s.get("ncci_code") or "",
            "classification":      s.get("classification") or "",
            "payroll":             r.get("wages") or "",
            "confidence":          s.get("confidence") or "",
            "rationale":           s.get("rationale") or "",
        })

    return pd.DataFrame(rows) if rows else _EMPTY_DF.copy()


# ── Theme & CSS ───────────────────────────────────────────────────────────────

_CSS = """
/* container */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: #f1f5f9 !important;
}

/* hero banner */
.ap-hero {
    background: linear-gradient(135deg, #0f2744 0%, #1a4a8a 55%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 28px 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ap-hero-title {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
.ap-hero-sub {
    color: #bfdbfe;
    font-size: 0.875rem;
    line-height: 1.6;
    margin: 0;
    max-width: 560px;
}
.ap-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    white-space: nowrap;
    margin-left: 24px;
    flex-shrink: 0;
}

/* upload card */
.upload-card {
    background: #ffffff !important;
    border: 2px solid #dbeafe !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(30,64,175,0.08) !important;
}

/* sample card */
.sample-card {
    background: linear-gradient(150deg, #f0fdf4, #eff6ff) !important;
    border: 2px solid #86efac !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* report panel */
.report-panel {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 28px 32px !important;
    min-height: 320px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    color: #1a202c !important;
}
.report-panel p,
.report-panel h1, .report-panel h2, .report-panel h3, .report-panel h4,
.report-panel li, .report-panel td, .report-panel th,
.report-panel strong, .report-panel em, .report-panel span {
    color: #1a202c !important;
}
.report-panel code {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
}
.report-panel table {
    width: 100% !important;
    border-collapse: collapse !important;
}
.report-panel th {
    background: #f8fafc !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding: 8px 12px !important;
    text-align: left !important;
    font-weight: 600 !important;
}
.report-panel td {
    border-bottom: 1px solid #f1f5f9 !important;
    padding: 7px 12px !important;
}

/* section divider */
.section-divider {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 20px 0 16px;
}
"""

_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="AuditPilot", theme=_THEME, css=_CSS) as demo:

    # hero
    gr.HTML("""
        <div class="ap-hero">
            <div>
                <div class="ap-hero-title">⚡ AuditPilot</div>
                <div class="ap-hero-sub">
                    AI-assisted workers' compensation premium audit — upload a payroll register,
                    IRS 941 form, or COI to extract employees, suggest NCCI class codes, and
                    generate a draft audit worksheet.
                </div>
            </div>
            <span class="ap-badge">WC Premium Audit · Texas</span>
        </div>
    """)

    with gr.Tabs(selected=0) as tabs:

        # ── Tab 1: Audit Workspace ─────────────────────────────────────────
        with gr.Tab("📄 Audit Workspace", id=0):

            with gr.Row():
                with gr.Column(scale=1, elem_classes=["upload-card"]):
                    gr.HTML('<p style="font-size:1rem;font-weight:700;color:#1e3a5f;margin:0 0 12px 0;">📤 Upload Your Document</p>')
                    file_input = gr.File(
                        label="Payroll register, IRS 941, or COI (PDF or TXT)",
                        file_types=[".pdf", ".txt"],
                    )
                    run_btn = gr.Button("▶  Run Audit Analysis", variant="primary", size="lg")

                with gr.Column(scale=1, elem_classes=["sample-card"]):
                    gr.HTML("""
                        <p style="font-size:1rem;font-weight:700;color:#14532d;margin:0 0 10px 0;">🧪 Try the Sample</p>
                        <p style="font-size:0.84rem;color:#111827;margin:0 0 8px 0;">
                            📄 <code style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:4px;font-size:0.79rem;font-weight:600;">sample_payroll_register.pdf</code>
                        </p>
                        <p style="font-size:0.82rem;color:#111827;line-height:1.75;margin:0;">
                            Summit Builders LLC · Q1 2024<br>
                            3 employees · Roofer · Electrician · Office Manager<br>
                            Includes a subcontractor with a missing COI
                        </p>
                    """)
                    sample_btn = gr.Button("▶  Run Sample Audit", variant="secondary", size="lg")

            gr.HTML("""
                <div style="margin:24px 0 10px;">
                    <span style="background:#1e3a5f;color:#fff;font-size:0.72rem;font-weight:700;
                                 letter-spacing:0.8px;text-transform:uppercase;
                                 padding:6px 16px;border-radius:6px;">
                        📊 Employee Classification Table
                    </span>
                </div>
            """)

            df_out = gr.Dataframe(
                headers=["name", "job_title", "suggested_ncci_code", "classification", "payroll", "confidence", "rationale"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                label="Employee Classification Table",
                interactive=False,
                wrap=True,
            )

            export_btn = gr.Button("⬇  Export CSV", size="sm")
            export_file = gr.File(label="Download CSV", visible=False)

            gr.HTML('<hr class="section-divider">')
            report_btn = gr.Button("📋  Generate Audit Report  →", variant="secondary", size="lg")

        # ── Tab 2: Audit Report ────────────────────────────────────────────
        with gr.Tab("📋 Audit Report", id=1):
            report_out = gr.Markdown(
                value="*Run an audit on the Workspace tab to generate the worksheet.*",
                label="Draft Audit Worksheet",
                elem_classes=["report-panel"],
            )
            gr.HTML('<div style="height:12px;"></div>')
            with gr.Row():
                pdf_btn = gr.Button("⬇  Download Report PDF", variant="primary")
                pc_btn  = gr.Button("☁  Push to PolicyCenter", variant="secondary")
            pdf_file_out = gr.File(label="Audit Report PDF", visible=False)
            gr.HTML("""
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;
                            padding:10px 14px;font-size:0.8rem;color:#92400e;margin-top:4px;">
                    ⚠ <strong>PolicyCenter integration coming soon.</strong>
                    MCP server scaffold is in <code>mcp_servers/policycenter.py</code>.
                    Connect your Guidewire tenant URL + credentials in <code>.env</code> to activate.
                </div>
            """)
            gr.Markdown(_AUDITOR_INSTRUCTIONS)

    # ── Event wiring ──────────────────────────────────────────────────────────

    _run_outputs = [df_out, report_out]

    run_btn.click(fn=run_audit, inputs=[file_input], outputs=_run_outputs)
    sample_btn.click(fn=run_sample_audit, inputs=[], outputs=_run_outputs)

    report_btn.click(fn=lambda: gr.update(selected=1), inputs=[], outputs=[tabs])

    pc_btn.click(
        fn=lambda: gr.Info("PolicyCenter integration not yet connected. See mcp_servers/policycenter.py."),
        inputs=[],
        outputs=[],
    )

    export_btn.click(
        fn=export_csv,
        inputs=[df_out],
        outputs=[export_file],
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[export_file],
    )

    pdf_btn.click(
        fn=generate_report_pdf,
        inputs=[report_out, df_out],
        outputs=[pdf_file_out],
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[pdf_file_out],
    )


def _ensure_knowledge_base() -> None:
    """Ingest RAG source PDFs if ChromaDB is empty.
    Runs automatically on HuggingFace Spaces where chroma_db/ is not committed."""
    import chromadb as _cdb
    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    try:
        col = _cdb.PersistentClient(path=chroma_path).get_or_create_collection("ncci_codes")
        if col.count() > 0:
            return
    except Exception:
        pass

    from rag.ingest import ingest
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pdfs = ["tx_wc_basic_manual.pdf", "tx_wc_alpha_index.pdf"]
    found = [p for p in pdfs if os.path.exists(os.path.join(data_dir, p))]

    if not found:
        print("RAG source PDFs not found in data/ — classification will use LLM knowledge only.")
        print("Upload tx_wc_basic_manual.pdf and tx_wc_alpha_index.pdf via the HF Space Files tab to enable RAG.")
        return

    print(f"Knowledge base empty — ingesting {len(found)} PDF(s) (first launch only, ~2 min)...")
    for pdf in found:
        print(f"  Ingesting {pdf} ...")
        ingest(os.path.join(data_dir, pdf))
    print("Knowledge base ready.")


if __name__ == "__main__":
    _ensure_knowledge_base()
    if os.getenv("SPACE_ID"):
        # HuggingFace Spaces: bind to all interfaces so the proxy can reach the app
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        # Local: FastAPI + Gradio via uvicorn (API docs at /docs)
        import uvicorn
        uvicorn.run("api.main:app", host="0.0.0.0", port=7860, reload=True)
