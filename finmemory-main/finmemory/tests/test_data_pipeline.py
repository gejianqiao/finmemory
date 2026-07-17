from unittest.mock import Mock, patch

import pandas as pd

from finmemory.data.loaders.finnhub_loader import FinnhubNewsLoader
from finmemory.data.loaders.sec_loader import SECFilingLoader
from finmemory.data.loaders.yahoo_loader import YahooFinanceLoader
from finmemory.data.processors.price_processor import process_price_data
from finmemory.utils.api_clients import SECClient
from finmemory.main import run_market_signal_agents
import finmemory.main as main_module


def _price_frame():
    index = pd.DatetimeIndex(["2025-03-03", "2025-03-04"], name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=index,
    )


def test_yahoo_loader_preserves_datetime_index(tmp_path):
    loader = YahooFinanceLoader(data_dir=tmp_path)
    ticker = Mock()
    ticker.history.return_value = _price_frame()

    with patch("finmemory.data.loaders.yahoo_loader.yf.Ticker", return_value=ticker):
        result = loader.download_single("AAPL", "2025-03-03", "2025-03-05", True)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tolist() == result["date"].tolist()
    assert not result.isna().any().any()

    cached = YahooFinanceLoader(data_dir=tmp_path).download_single(
        "AAPL", "2025-03-03", "2025-03-05"
    )
    assert isinstance(cached.index, pd.DatetimeIndex)


def test_price_processor_accepts_yahoo_column_names():
    result = process_price_data(_price_frame(), "AAPL")
    assert result.index.equals(_price_frame().index)
    assert "simple_return" in result.columns


def test_finnhub_loader_uses_api_cache_contract(tmp_path):
    loader = FinnhubNewsLoader(data_dir=tmp_path, cache_enabled=False, api_key="test")
    loader.client.get_company_news = Mock(return_value=[])

    first = loader.get_company_news("AAPL", "2025-03-03", "2025-03-05")
    second = loader.get_company_news("AAPL", "2025-03-03", "2025-03-05")

    assert first.empty and second.empty
    loader.client.get_company_news.assert_called_once()


def test_sec_submissions_url_and_shape(tmp_path):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-25-000001"],
                "filingDate": ["2025-03-05"],
                "reportDate": ["2025-03-01"],
                "form": ["10-Q"],
            }
        }
    }

    client = SECClient(user_agent="FinMemory tests test@example.com")
    with patch("finmemory.utils.api_clients.requests.get", return_value=response) as get:
        payload = client.get_submissions("320193")
    assert "/submissions/CIK0000320193.json" in get.call_args.args[0]
    assert "/api/xbrl/submissions/" not in get.call_args.args[0]
    assert payload["filings"]["recent"]["form"] == ["10-Q"]

    loader = SECFilingLoader(data_dir=tmp_path, cache_enabled=False)
    loader.get_company_cik = Mock(return_value="0000320193")
    loader.sec_client.get_submissions = Mock(return_value=response.json())
    filings = loader.get_submissions("AAPL")
    assert filings == [
        {
            "accessionNumber": "0000320193-25-000001",
            "filingDate": "2025-03-05",
            "reportDate": "2025-03-01",
            "form": "10-Q",
        }
    ]


def test_sec_ticker_registry_uses_cik_str():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }
    client = SECClient(user_agent="FinMemory tests test@example.com")
    with patch("finmemory.utils.api_clients.requests.get", return_value=response):
        assert client._get_cik_from_ticker("aapl") == "0000320193"


def test_deterministic_market_signal_preparation_uses_all_sources():
    result = run_market_signal_agents(
        ticker="AAPL",
        company_news=pd.DataFrame([{"datetime": "2025-03-03", "headline": "AAPL launch", "summary": "New product"}]),
        macro_news=pd.DataFrame([{"datetime": "2025-03-03", "headline": "Rates", "summary": "Rate news"}]),
        filings={
            "10-K": [{"text": "annual", "filing_date": "2025-03-03", "report_date": "2024-12-31", "form_type": "10-K"}],
            "10-Q": [{"text": "quarter", "filing_date": "2025-03-04", "report_date": "2025-03-01", "form_type": "10-Q"}],
        },
        dates=pd.date_range("2025-03-03", periods=3),
    )

    assert len(result["pending_memory"]) == 4
    assert result["source_counts"] == {"company_news": 1, "macro_news": 1, "sec_filings": 2}


def test_single_stock_loop_runs_with_mock_llm_and_real_components(monkeypatch):
    prices = _price_frame()
    prices["date"] = prices.index
    prices["adj_close"] = prices["Close"]
    direction = Mock()
    direction.decide_direction.return_value = {
        "investment_decision": "hold", "summary_reason": "test",
        "memory_indices": [], "reflection_analysis": "none",
    }
    direction.get_cost_summary.return_value = {}
    quantity = Mock()
    quantity.decide_quantity.return_value = {
        "order_size": 0, "summary_reason": "test", "memory_indices": []
    }
    quantity.get_cost_summary.return_value = {}
    context = Mock()
    context.analyze_context.return_value = {
        "market_summary": "test", "sentiment_score": 0.0, "importance_score": 1,
        "key_catalysts": [], "key_risks": [],
    }
    context.get_cost_summary.return_value = {}
    agents = {
        "market_context_agent": context,
        "direction_agent": direction,
        "quantity_risk_agent": quantity,
    }
    snapshot = Mock()
    snapshot.to_dict.return_value = {"avg_sentiment": 0.0, "signals": []}
    processor = Mock()
    processor.get_signal_snapshot.return_value = snapshot

    monkeypatch.setattr(main_module, "load_data", lambda **kwargs: {
        "AAPL": {"prices": prices, "news": pd.DataFrame(), "macro_news": pd.DataFrame(), "filings": {}}
    })
    monkeypatch.setattr(main_module, "initialize_agents", lambda *args, **kwargs: agents)
    monkeypatch.setattr(main_module, "run_market_signal_agents", lambda **kwargs: {
        "signal_processor": processor, "snapshots": [], "pending_memory": [],
        "source_counts": {"company_news": 0, "macro_news": 0, "sec_filings": 0},
    })

    result = main_module.run_single_stock("AAPL", "2025-03-03", "2025-03-05")
    assert len(result["episode_log"]) == 1
    assert result["reflection_statistics"]["total_outcomes"] == 1
    direction.decide_direction.assert_called_once()
    quantity.decide_quantity.assert_called_once()
    context.analyze_context.assert_called_once()
