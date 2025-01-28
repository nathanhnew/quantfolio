import os
from unittest.mock import patch

import pytest

from quantfolio.portfolio import Portfolio


def test_portfolio_init_errors() -> None:
    # Portfolio cannot be empty
    with pytest.raises(ValueError):
        Portfolio({}, "FMP")
    # Negative weights not allowed
    with pytest.raises(ValueError):
        Portfolio({"AAPL": -1}, "FMP")
    # Weights sum to 0
    with pytest.raises(ValueError):
        Portfolio({"AAPL": 0}, "FMP")
    # Exceed max position size
    with pytest.raises(ValueError):
        Portfolio({"AAPL": 60, "GOOGL": 40}, "FMP", max_position_size=20)
    # Fail on missing API key
    with pytest.raises(ValueError):
        Portfolio({"AAPL": 0.2, "GOOGL": 0.8}, "FMP")


def test_portfolio_weight_adjustments() -> None:
    # Test normalized weights
    with patch.dict(os.environ, {"FMP_API_KEY": "abc"}):
        portfolio = Portfolio({"AAPL": 40, "GOOGL": 20}, backend="FMP")
        assert sum([position.weight for position in portfolio.positions]) == 1
        assert portfolio["AAPL"].weight == 2 / 3
        assert "GOOGL" in portfolio
