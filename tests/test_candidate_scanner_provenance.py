import json
import unittest
from unittest.mock import patch

import candidate_scanner


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestCandidateScannerProvenance(unittest.TestCase):
    def test_fetch_klines_never_uses_binance_futures(self):
        calls = []

        payload = [[
            1_700_000_000_000,
            "100.0",
            "101.0",
            "99.0",
            "100.5",
            "10.0",
            1_700_000_899_999,
            "1000.0",
            10,
            "5.0",
            "500.0",
            "0",
        ]]

        def fake_urlopen(request, timeout=10):
            url = request.full_url
            calls.append(url)
            self.assertNotIn("fapi.binance.com", url)
            return FakeResponse(payload)

        with patch(
            "candidate_scanner.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = candidate_scanner.fetch_klines(
                "ETHUSDT",
                "15m",
                limit=1,
            )

        self.assertFalse(result.empty)
        self.assertEqual(result.attrs["logical_symbol"], "ETHUSDT")
        self.assertEqual(result.attrs["interval"], "15m")
        self.assertEqual(result.attrs["observation_market"], "BINANCE_SPOT")
        self.assertEqual(
            result.attrs["observation_source"],
            {"exchange": "binance", "market_type": "spot"},
        )
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
