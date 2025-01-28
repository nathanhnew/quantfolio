import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

from .asset import Asset


@dataclass
class Position:
    asset: Asset
    weight: float


class Portfolio:

    def __init__(
        self,
        positions: Dict[str, float],
        backend: Asset.BackendLiteral,
        rebalance_threshold: Optional[float] = None,
        max_position_size: Optional[float] = None,
        inception_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        self.logger = logging.getLogger(__name__)

        self._validate_weights(positions, max_position_size)
        self.backend = backend
        self._positions = self._transform_positions(positions)
        self._history: Optional[pd.DataFrame] = None

        self.rebalance_threshold = rebalance_threshold or -1
        self.inception_date = inception_date or datetime(1900, 1, 1)
        self.end_date = end_date or datetime.now()
        self.historical_fetched = False

    def __getitem__(self, ticker: str) -> Position:
        try:
            return self._positions[ticker]
        except KeyError:
            raise KeyError(f"Asset {ticker} not in portfolio")

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._positions

    def __iter__(self):
        return iter(self._positions.values())

    def __repr__(self) -> str:
        return str({pos.asset.ticker: pos.weight for pos in self.positions})

    @staticmethod
    def _validate_weights(asset_dict: Dict[str, float], max_position_size: Optional[float] = None) -> None:
        if not asset_dict:
            raise ValueError("Portfolio cannot be empty")

        if any(w < 0 for w in asset_dict.values()):
            raise ValueError("Negative weights not allowed")

        if sum(asset_dict.values()) == 0:
            raise ValueError("Weights cannot sum to 0")

        total_weights = sum(asset_dict.values())
        if max_position_size and max_position_size >= 0:
            max_position_size = max_position_size if max_position_size < 1 else max_position_size / 100
            for asset, weight in asset_dict.items():
                if (weight / total_weights) > max_position_size:
                    raise ValueError(
                        f"Asset {asset} weighted {weight / total_weights * 100}% exceeds max position weight of {max_position_size}"
                    )

    def _transform_positions(self, position_dict: Dict[str, float]) -> Dict[str, Position]:
        self.logger.debug("Transforming assets to Position dict")
        total_weights = sum(position_dict.values())
        return {
            ticker: Position(asset=Asset(ticker=ticker, backend=self.backend), weight=weight / total_weights)
            for ticker, weight in position_dict.items()
        }

    @property
    def positions(self) -> List[Position]:
        return list(self._positions.values())

    async def fetch_historical_data(
        self,
        refetch: bool = False,
        max_concurrent: int = 10,
    ) -> None:
        self.logger.debug(f"Fetching historical data. Forcing refetch: {refetch}")
        needfetch = refetch or not all(position.asset.fetched for position in self.positions)
        print([position.asset.fetched for position in self.positions])
        if needfetch:
            self.logger.debug(f"Fetching historical data for {len(self.positions)} positions")
            concurrency_limiter = asyncio.Semaphore(max_concurrent)

            limits = httpx.Limits(max_keepalive_connections=max_concurrent, max_connections=max_concurrent)
            async with httpx.AsyncClient(limits=limits, timeout=30) as client:
                tasks = [
                    position.asset.async_fetch_history(
                        client=client,
                        start_date=self.inception_date,
                        end_date=self.end_date,
                        rate_limiter=concurrency_limiter,
                        refetch=refetch,
                    )
                    for position in self.positions
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                errors = [
                    (position.asset.ticker, result)
                    for position, result in zip(self.positions, results)
                    if isinstance(result, Exception)
                ]
                if errors:
                    error_msg = "\n".join(f"[{ticker}] Error: {str(error)}" for ticker, error in errors)
                    raise RuntimeError(f"Failed to initialize assets:\n{error_msg}")

    @property
    def history(self) -> pd.DataFrame:
        if self._history is None:
            if any(pos.asset.history is None for pos in self.positions):
                raise RuntimeError(
                    "Cannot update portfolio historical data without asset data. Call fetch_historical_data first"
                )
            assert self.history is not None
            dfs = []
            for position in self.positions:
                assert position.asset.history is not None
                df = position.asset.history.copy()
                df["weight"] = position.weight
                dfs.append(df)
            portfolio_df = pd.concat(dfs)
            portfolio_df = portfolio_df.reset_index()
            portfolio_df = portfolio_df.set_index(["date", "ticker"])

            self._history = portfolio_df
            self.flag_history()

        return self._history

    def flag_history(self) -> None:
        weighted_tickers = self.history[self.history["weight"] > 0].index.get_level_values(1).unique()

        wide_df = self.history.reset_index().pivot(index="date", columns="ticker")
        complete_mask = pd.Series(True, index=wide_df.index)
        for col in self.history.columns:
            for ticker in weighted_tickers:
                complete_mask &= wide_df[col, ticker].notna()

        complete_dates = set(complete_mask[complete_mask].index)
        self.history["process"] = self.history.index.get_level_values(0).isin(complete_dates)
