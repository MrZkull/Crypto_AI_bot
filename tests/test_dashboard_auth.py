import base64
import os
import unittest
from unittest.mock import patch

import dashboard


USERNAME = "cryptobot-admin"
PASSWORD = "test-only-dashboard-passphrase-long-enough"
PUBLIC_ORIGIN = "https://cryptobot-test.example"
AUTH_ENV = {
    "DASHBOARD_AUTH_USERNAME": USERNAME,
    "DASHBOARD_AUTH_PASSWORD": PASSWORD,
    "DASHBOARD_PUBLIC_ORIGIN": PUBLIC_ORIGIN,
}


def basic_auth_header(username=USERNAME, password=PASSWORD):
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


class TestDashboardAuthentication(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, AUTH_ENV)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.client = dashboard.app.test_client()

    def test_health_probe_is_public_and_fails_readiness_if_auth_missing(self):
        with patch.dict(os.environ, {
            "DASHBOARD_AUTH_USERNAME": "",
            "DASHBOARD_AUTH_PASSWORD": "",
            "DASHBOARD_PUBLIC_ORIGIN": "",
        }):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(payload["status"], "not_ready")
        self.assertEqual(set(payload), {"status", "time"})

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_missing_auth_configuration_fails_closed(self):
        with patch.dict(os.environ, {
            "DASHBOARD_AUTH_USERNAME": "",
            "DASHBOARD_AUTH_PASSWORD": "",
            "DASHBOARD_PUBLIC_ORIGIN": "",
        }):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "dashboard_auth_not_configured")

    def test_unauthenticated_dashboard_and_api_are_challenged(self):
        for path, method in (("/", "GET"), ("/api/status", "GET"), ("/api/scan", "POST")):
            with self.subTest(path=path, method=method):
                response = self.client.open(path, method=method)
                self.assertEqual(response.status_code, 401)
                self.assertIn("Basic realm=", response.headers.get("WWW-Authenticate", ""))

    def test_wrong_credentials_are_rejected(self):
        response = self.client.get("/", headers=basic_auth_header("not-admin", PASSWORD))
        self.assertEqual(response.status_code, 401)
        response = self.client.get("/", headers=basic_auth_header(USERNAME, "incorrect-password"))
        self.assertEqual(response.status_code, 401)

    def test_valid_credentials_allow_dashboard(self):
        response = self.client.get("/", headers=basic_auth_header())
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CryptoBot AI", response.data)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")

    def test_cross_origin_mutation_is_blocked_before_handler(self):
        with patch.object(dashboard, "deribit_client") as client_factory:
            response = self.client.post(
                "/api/close_trade",
                headers={**basic_auth_header(), "Origin": "https://attacker.example"},
                json={"symbol": "ETHUSDT"},
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["error"], "cross_origin_request_rejected")
        client_factory.assert_not_called()

    def test_mutation_without_origin_or_referer_is_blocked(self):
        with patch.object(dashboard, "deribit_client") as client_factory:
            response = self.client.post(
                "/api/close_trade", headers=basic_auth_header(), json={"symbol": "ETHUSDT"}
            )
        self.assertEqual(response.status_code, 403)
        client_factory.assert_not_called()

    def test_same_origin_mutation_passes_guard_then_reaches_handler(self):
        with patch.object(dashboard, "deribit_client", return_value=None) as client_factory:
            response = self.client.post(
                "/api/close_trade",
                headers={**basic_auth_header(), "Origin": PUBLIC_ORIGIN},
                json={"symbol": "ETHUSDT"},
            )
        self.assertEqual(response.status_code, 500)
        client_factory.assert_called_once()

    def test_weak_password_and_invalid_origin_fail_closed(self):
        with patch.dict(os.environ, {"DASHBOARD_AUTH_PASSWORD": "too-short"}):
            response = self.client.get("/", headers=basic_auth_header())
        self.assertEqual(response.status_code, 503)
        with patch.dict(os.environ, {
            "DASHBOARD_PUBLIC_ORIGIN": "https://trusted.example/not-an-origin",
        }):
            response = self.client.get("/", headers=basic_auth_header())
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
