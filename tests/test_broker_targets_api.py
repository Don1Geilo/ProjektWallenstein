from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wallenstein.broker_targets as bt


def test_fmp_get_request_is_made_once_with_api_key(requests_mock, monkeypatch):
    monkeypatch.setattr(bt, "FMP_API_KEY", "test-key")
    url = "https://financialmodelingprep.com/api/v4/price-target-consensus"
    payload = {"foo": "bar"}
    requests_mock.get(url, json=payload)

    result = bt._fmp_get("price-target-consensus", {"symbol": "AAPL"})

    assert result == payload
    assert requests_mock.call_count == 1
    req = requests_mock.last_request
    assert req.qs["apikey"] == ["test-key"]

