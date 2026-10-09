from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / 'dashboard_static' / 'index.html').read_text(encoding='utf-8')

def test_paper_history_has_result_and_signed_pnl_column():
    assert 'id="paperClosedBody"' in INDEX
    assert 'class="paper-result-pnl"' in INDEX
    assert '<th>Result + PnL</th>' in INDEX
    assert "const outcomeLabel = net == null ? 'UNVERIFIED'" in INDEX
    assert 'const outcomeAmount = net == null' in INDEX

def test_paper_zero_and_unverified_outcomes_are_not_mislabeled():
    assert "net == null ? 'UNVERIFIED' : net > 0 ? 'WIN' : net < 0 ? 'LOSS' : 'BREAKEVEN'" in INDEX
    assert 'USDT`;' in INDEX

def test_existing_deribit_history_and_pdf_hooks_remain_present():
    assert 'function renderHistoryFiltered()' in INDEX
    assert 'function printPDFReport()' in INDEX
    assert '/api/download_report_pdf' in INDEX
