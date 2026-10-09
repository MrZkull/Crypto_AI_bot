
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = (ROOT / "dashboard.py").read_text(encoding="utf-8")
INDEX = (ROOT / "dashboard_static" / "index.html").read_text(encoding="utf-8")

def test_history_result_and_separate_paper_pdf():
    assert "<th>Result + PnL</th>" in INDEX
    assert "history-result-pnl" in INDEX
    assert '@app.route("/api/download_report_pdf"' in DASH
    assert '@app.route("/api/download_paper_report_pdf"' in DASH
    assert "def generate_pdf_bytes(scope: str, summary: dict, trades: list)" in DASH

def test_paper_reporting_ui():
    assert 'id="paperPdfBtn"' in INDEX
    assert "Paper Risk &amp; Health Analytics" in INDEX
    assert "function downloadPaperPDF()" in INDEX
    assert "function renderPaperAnalytics(data)" in INDEX

def test_existing_history_report_frontend_is_still_present():
    assert "function printPDFReport()" in INDEX
    assert "/api/download_report_pdf" in INDEX
