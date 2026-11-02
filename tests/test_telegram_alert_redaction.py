import requests
import telegram_alert


def test_telegram_timeout_redacts_token(monkeypatch, capsys):
    token = "TEST_TOKEN_SHOULD_NOT_LEAK"
    monkeypatch.setattr(telegram_alert, "BOT_TOKEN", token)
    monkeypatch.setattr(telegram_alert, "CHAT_ID", "123456")

    def fail(*args, **kwargs):
        raise requests.Timeout(f"https://api.telegram.org/bot{token}/sendMessage")

    monkeypatch.setattr(telegram_alert.requests, "post", fail)
    assert telegram_alert.send_message("test") is False
    output = capsys.readouterr().out
    assert token not in output
    assert "Timeout" in output


def test_telegram_http_failure_redacts_response(monkeypatch, capsys):
    monkeypatch.setattr(telegram_alert, "BOT_TOKEN", "test-token")
    monkeypatch.setattr(telegram_alert, "CHAT_ID", "123456")

    class Response:
        status_code = 401
        def json(self):
            return {"ok": False, "description": "token=test-token"}

    monkeypatch.setattr(telegram_alert.requests, "post", lambda *a, **k: Response())
    assert telegram_alert.send_message("test") is False
    output = capsys.readouterr().out
    assert "HTTP 401" in output
    assert "test-token" not in output


def test_telegram_success_requires_ok_true(monkeypatch):
    monkeypatch.setattr(telegram_alert, "BOT_TOKEN", "test-token")
    monkeypatch.setattr(telegram_alert, "CHAT_ID", "123456")

    class Response:
        status_code = 200
        def json(self):
            return {"ok": True, "result": {"message_id": 1}}

    monkeypatch.setattr(telegram_alert.requests, "post", lambda *a, **k: Response())
    assert telegram_alert.send_message("test") is True
