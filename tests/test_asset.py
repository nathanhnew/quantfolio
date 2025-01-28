import os
from unittest.mock import Mock, patch

import pytest

from quantfolio.asset import Asset


def test_asset_init() -> None:
    # All 4 should fail with Value error
    with pytest.raises(ValueError):
        Asset(ticker=None, backend="FMP")
    with pytest.raises(ValueError):
        Asset(ticker="AAPL", backend="FMP")
    with pytest.raises(ValueError):
        Asset(ticker="AAPL", backend="Fails")
    with pytest.raises(ValueError):
        Asset(ticker="AAPL", backend="FMP")

    with patch.dict(os.environ, {"FMP_API_KEY": "abc123"}, clear=True):
        aapl = Asset(ticker="AAPL", backend="FMP")
        assert aapl.ticker == "AAPL"
        assert aapl.history is None
        assert aapl.backend.api_key == "abc123"
