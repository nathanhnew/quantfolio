from asyncio import Semaphore
from datetime import datetime
from logging import getLogger
from typing import List, Literal, Optional, Tuple, get_args

import httpx
import numpy as np
import pandas as pd

from .backend import FMP


class Asset:
    AssetTypeLiteral = Literal["etf", "stock", "bond", "mutualfund"]
    ALLOWED_ASSETS: Tuple[str] = get_args(AssetTypeLiteral)

    BackendLiteral = Literal["FMP"]
    ALLOWED_BACKENDS: Tuple[str] = get_args(BackendLiteral)

    def __init__(
        self,
        ticker: str,
        backend: BackendLiteral,
        backend_api_key: Optional[str] = None,
    ):
        if not isinstance(ticker, str):
            raise ValueError("String ticker must be provided")
        if backend not in self.ALLOWED_BACKENDS:
            raise ValueError(f"Backend not allowed. Must be in {self.ALLOWED_BACKENDS}")

        self.ticker = ticker.upper()

        self.history: Optional[pd.DataFrame] = None

        if backend == "FMP":
            self.backend = FMP(backend_api_key)

        self.volatility_metrics: Optional[pd.DataFrame] = None
        self.momentum_metrics: Optional[pd.DataFrame] = None

        self.logger = getLogger(__name__)

    def fetch_historical_data(
        self,
        start: datetime = datetime(1900, 1, 1),
        end: datetime = datetime.now(),
        refetch: bool = False,
    ) -> pd.DataFrame:
        if self.history is None or refetch:
            self.history = self.backend.fetch_asset_history(self.ticker, start_date=start, end_date=end)

        return self.history

    async def async_fetch_history(
        self,
        start_date: datetime = datetime(1900, 1, 1),
        end_date: datetime = datetime.now(),
        client: Optional[httpx.AsyncClient] = None,
        retries: int = 3,
        backoff_factor: float = 1.25,
        rate_limiter: Optional[Semaphore] = None,
        refetch: bool = False,
    ) -> pd.DataFrame:
        if self.history is None or refetch:
            self.logger.info(f"[{self.ticker}] Fetching historical data")
            start = datetime.now()
            self.history = await self.backend.async_fetch_history(
                ticker=self.ticker,
                start_date=start_date,
                end_date=end_date,
                client=client,
                retries=retries,
                backoff_factor=backoff_factor,
                rate_limiter=rate_limiter,
            )
            self.logger.debug(f"[{self.ticker}] Fetching historical data. Done {datetime.now() - start}s elapsed")
        return self.history

    @property
    def fetched(self) -> bool:
        return self.history is not None

    def calculate_volatility_metrics(self, windows: List[int] = [21, 63, 252]) -> pd.DataFrame:
        self.logger.debug("Beginning volatility calculations")
        if self.history is None:
            self.logger.warning("Historical data not fetched. Using full historical availability...")
            self.fetch_historical_data()
        assert self.history is not None

        if self.volatility_metrics is None:
            prices = self.history["adjClose"]
            returns = prices.pct_change()

            metrics = pd.DataFrame(index=self.history.index)

            # Get the rolling standard deviation of each window
            for window in windows:
                metrics[f"volatility_{window}d"] = returns.rolling(window=window).std() * np.sqrt(252)  # annualized

            rolling_max = prices.expanding().max()
            drawdown = prices / rolling_max - 1
            metrics["max_drawdown"] = drawdown

            self.volatility_metrics = metrics

        return self.volatility_metrics
