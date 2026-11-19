from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from scripts.verify_f08_firewall import is_f08_locked_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    [
        "phase2d_f08_restore_guard.py",
        "tests/test_phase2d_observation_ledger.py",
        "research_outputs/F08_AUDIT_ARCHIVE/README.md",
        "feature_engineering.py",
        "market_data_integrity.py",
        r"research_outputs\\F08_AUDIT_ARCHIVE\\snapshot.json",
        "research_outputs/F08_AUDIT_ARCHIVE/.ignored_snapshot_cache",
        "tests/test_phase2d_f08_new_gate.py",
    ],
)
def test_f08_firewall_rejects_locked_paths(path):
    assert is_f08_locked_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "dashboard.py",
        "deribit_client.py",
        "exchange_environment_guard.py",
        "tests/test_exchange_environment_guard.py",
        "docs/DERIBIT_TESTNET_READONLY_SOAK.md",
        "docs/F09_MACRO_REGIME_PREREGISTRATION.md",
    ],
)
def test_f08_firewall_allows_unrelated_reliability_paths(path):
    assert not is_f08_locked_path(path)


def test_emergency_close_script_is_retired_and_has_no_network_mutation_imports():
    source = (ROOT / "emergency_close.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [node])
    }
    assert "requests" not in imports
    assert "DERIBIT_CLIENT_SECRET" not in source
    assert "cancel_all_by_currency" not in source
    assert "private/buy" not in source
    assert "private/sell" not in source
    assert "retired" in source.lower()


def test_dashboard_source_has_unconditional_read_only_release_gate():
    source = (ROOT / "dashboard.py").read_text(encoding="utf-8")
    assert "DASHBOARD_MUTATIONS_ENABLED = False" in source
    assert "and not DASHBOARD_MUTATIONS_ENABLED" in source
    assert '"execution_controls_enabled": False' in source


def test_scanner_no_longer_dispatches_manual_auditor_after_every_scan():
    workflow = (ROOT / ".github/workflows/crypto_bot.yml").read_text(encoding="utf-8")
    assert "Trigger Auditor Workflow" not in workflow
    assert "DERIBIT_CLIENT_ID:" not in workflow
    assert "DERIBIT_CLIENT_SECRET:" not in workflow
    assert "DERIBIT_TESTNET_CLIENT_ID:" not in workflow
    assert "DERIBIT_TESTNET_CLIENT_SECRET:" not in workflow
    assert "DERIBIT_TESTNET_MUTATION_ACK:" not in workflow
    assert workflow.count("DERIBIT_TESTNET_MUTATIONS_ENABLED: 'false'") == 2
    assert "group: cryptobot-state-writers" in workflow


def test_only_one_workflow_declares_live_scanner_order_writer_mode():
    matches = []
    workflows = ROOT / ".github" / "workflows"
    for path in workflows.glob("*.y*ml"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "DERIBIT_TESTNET_MUTATIONS_ENABLED" in text:
            matches.append(path.name)
    assert matches == ["crypto_bot.yml"]


def test_dashboard_html_is_monitoring_only_without_exchange_buttons():
    html = (ROOT / "dashboard_static" / "index.html").read_text(encoding="utf-8")
    assert "MONITORING ONLY" in html
    assert 'onclick="triggerScan()"' not in html
    assert 'onclick="openKillModal()"' not in html
    assert 'onclick="closeTrade(' not in html
    assert "healTradeBracket(" not in html
    assert "EMERGENCY KILL SWITCH MODAL" not in html



def test_scanner_workflow_uses_shared_state_writer_group_with_fifo_pending_queue():
    workflow = Path(".github/workflows/crypto_bot.yml").read_text(encoding="utf-8")
    assert "group: cryptobot-state-writers" in workflow
    assert "queue: max" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "repository_dispatch:" not in workflow
    assert "DERIBIT_TESTNET_MUTATIONS_ENABLED: 'false'" in workflow
    assert "DERIBIT_TESTNET_CLIENT_SECRET" not in workflow
    assert "DERIBIT_CLIENT_SECRET" not in workflow


def test_dashboard_html_has_no_manual_probation_cooldown_or_proposal_mutations():
    html = (ROOT / "dashboard_static" / "index.html").read_text(encoding="utf-8")
    forbidden = (
        "manualBenchCoin(",
        "releaseProbation(",
        "releaseCooldown(",
        "approveProposal(",
        "rejectProposal(",
        "openEmailModal(",
        "sendEmailReport(",
        "/api/probation/bench",
        "/api/probation/release",
        "/api/cooldown/release",
        "/api/adaptation_proposals/${id}/approve",
        "/api/adaptation_proposals/${id}/reject",
        "/api/send_report",
    )
    for fragment in forbidden:
        assert fragment not in html

    # In-memory PDF exports remain available without changing trading state.
    assert "downloadPaperPDF()" in html
    assert "download_paper_report_pdf" in html
    assert "download_report_pdf" in html
    assert "MONITORING ONLY" in html


def test_dashboard_server_allows_only_the_two_in_memory_pdf_post_endpoints():
    source = (ROOT / "dashboard.py").read_text(encoding="utf-8")
    assert '"api_download_report_pdf"' in source
    assert '"api_download_paper_report_pdf"' in source
    assert '"api_send_report"' not in source[source.index("# Hard read-only boundary"):source.index("@app.after_request")]


def test_dashboard_github_sync_uses_read_only_token_and_disables_content_writes():
    dashboard_source = (ROOT / "dashboard.py").read_text(encoding="utf-8")
    render_source = (ROOT / "render.yaml").read_text(encoding="utf-8")

    assert 'os.getenv("GH_READ_TOKEN"' in dashboard_source
    assert "GH_PAT_TOKEN" not in dashboard_source
    assert "GH_TOKEN" not in dashboard_source
    push_start = dashboard_source.index("def gh_push(")
    push_end = dashboard_source.index("def get(filename", push_start)
    push_body = dashboard_source[push_start:push_end]
    assert "return False" in push_body
    assert "requests.put" not in push_body

    assert "GH_READ_TOKEN" in render_source
    assert "GH_PAT_TOKEN" not in render_source


def test_html_unsafe_requests_are_limited_to_in_memory_pdf_exports():
    html = (ROOT / "dashboard_static" / "index.html").read_text(encoding="utf-8")
    import re

    post_urls = set(re.findall(
        r"fetch\(['\"]([^'\"]+)['\"],\s*\{\s*method\s*:\s*['\"]POST['\"]",
        html,
        flags=re.IGNORECASE,
    ))
    assert post_urls == {
        "/api/download_paper_report_pdf",
        "/api/download_report_pdf",
    }


def test_unknown_order_recovery_includes_historical_order_and_trade_by_order_queries():
    source = (ROOT / "trade_executor.py").read_text(encoding="utf-8")
    client_source = (ROOT / "deribit_client.py").read_text(encoding="utf-8")
    assert "get_order_history_by_instrument_name" in source
    assert "get_user_trades_by_order" in source
    assert '"historical": bool(historical)' in client_source
    assert "absence from either remains UNKNOWN" in source
    assert "Empty historical results do not prove rejection or no fill" in source


def test_every_non_pdf_dashboard_mutation_handler_is_a_route_level_noop():
    source = (ROOT / "dashboard.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    expected = {
        "api_send_report", "api_trades_heal", "api_probation_release",
        "api_probation_bench", "api_cooldown_release", "api_approve_proposal",
        "api_reject_proposal", "api_scan", "api_kill_switch", "api_close_trade",
    }
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in expected:
        node = functions[name]
        body = ast.get_source_segment(source, node) or ""
        assert "_dashboard_read_only_response(" in body, name
        for forbidden in (
            "place_market_order(", "cancel_order(", "cancel_all_by_currency(",
            "gh_push(", "requests.post(", "write_text(", "_store_report(",
            "send_email", "client.set_leverage(",
        ):
            assert forbidden not in body, (name, forbidden)


def test_all_scanner_and_auditor_state_writers_share_one_concurrency_group():
    paths = (
        ROOT / ".github/workflows/crypto_bot.yml",
        ROOT / ".github/workflows/hourly_research_pipeline.yml",
        ROOT / ".github/workflows/audit_workflow.yml",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "group: cryptobot-state-writers" in source, path.name
        assert "queue: max" in source, path.name
        assert "cancel-in-progress: false" in source, path.name


def test_render_dashboard_has_only_read_scoped_integrations():
    source = (ROOT / "render.yaml").read_text(encoding="utf-8")
    for forbidden in (
        "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "BINANCE_API_KEY", "BINANCE_SECRET",
        "CRYPTOPANIC_KEY", "GH_PAT_TOKEN", "GH_TOKEN", "DERIBIT_CLIENT_SECRET",
        "DERIBIT_MAINNET_CLIENT_SECRET", "DERIBIT_TESTNET_MUTATION_ACK",
    ):
        assert forbidden not in source
    assert "GH_READ_TOKEN" in source
    assert "DERIBIT_TESTNET_READONLY_CLIENT_ID" in source
    assert "DERIBIT_TESTNET_READONLY_CLIENT_SECRET" in source


def test_readonly_testnet_workflow_is_main_only_scoped_and_never_mutates():
    source = (ROOT / ".github/workflows/deribit_testnet_readonly.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in source
    assert "if: github.ref == 'refs/heads/main'" in source
    assert "permissions:\n  contents: read" in source
    assert "environment: deribit-testnet-readonly" in source
    assert "DERIBIT_TESTNET_READONLY_CLIENT_ID" in source
    assert "DERIBIT_TESTNET_READONLY_CLIENT_SECRET" in source
    assert "DERIBIT_TESTNET_MUTATIONS_ENABLED" not in source
    assert "DERIBIT_TESTNET_MUTATION_ACK" not in source
    assert "private/buy" not in source and "private/sell" not in source
    assert "testnet_readonly_check.py" in source
    assert "testnet_readonly_evidence.json" in source
    assert "retention-days: 14" in source


def test_postgres_driver_is_optional_and_not_in_default_runtime_requirements():
    base_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    optional_requirements = (ROOT / "requirements-postgres.txt").read_text(encoding="utf-8")
    assert "psycopg" not in base_requirements.lower()
    assert "psycopg[binary]>=3.2,<4" in optional_requirements
