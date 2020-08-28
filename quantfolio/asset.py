from __future__ import annotations
from .web import TiingoWebReader, QuantfolioWebInterface
from typing import Dict, Union
import pandas as pd
import numpy as np
import aiohttp
from datetime import datetime
from functools import wraps
from pytz import UTC

class Asset:
    def __init__(self, ticker: str, weight: float = 1):
        self.ticker = ticker.upper()
        self.weight = weight if weight < 1 else weight / 100
        self.historical = None
        self._pct_change = None
        self._pct_change_raw = None
        self._std = None
    
    @property
    def min_date(self) -> datetime:
        return self.historical.index.min().to_pydatetime() if isinstance(self.historical, pd.DataFrame) else None
    
    @property
    def max_date(self) -> datetime:
        return self.historical.index.max().to_pydatetime() if isinstance(self.historical, pd.DataFrame) else None
    
    def get_historical_data(self, web_reader: QuantfolioWebInterface, start_date: str = '1970-01-01', end_date: str = datetime.now().strftime('%Y-%m-%d')) -> None:
        closing_prices = web_reader.get_historical_close(self.ticker, start_date, end_date)
        df = pd.DataFrame.from_dict(closing_prices, orient='index')
        df.index = pd.to_datetime(df.index)
        self.historical = df
        return
    
    async def get_historical_data_async(self, web_reader: QuantfolioWebInterface, session: aiohttp.ClientSession, start_date: str = '1970-01-01', end_date: str = datetime.now().strftime('%Y-%m-%d')) -> None:
        closing_prices = await web_reader.get_historical_close_async(self.ticker, session, start_date, end_date)
        df = pd.DataFrame.from_dict(closing_prices, orient='index')
        df.index = pd.to_datetime(df.index)
        self.historical = df
        return
    
    def assert_historical(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if args[0].historical is None or args[0].historical.empty:
                raise ValueError(f"Historical data must be defined for asset {args[0].ticker} before calling this function. Please call .get_historical_data() or .get_historical_data_async() first")
            value = func(*args, **kwargs)
            return value
        return wrapper
    
    @property
    @assert_historical
    def pct_change(self) -> float:
        if self._pct_change is None:
            self._pct_change = np.log(self.historical.close).diff()
        return self._pct_change
    
    @property
    @assert_historical
    def pct_change_raw(self) -> float:
        if self._pct_change_raw is None:
            self._pct_change_raw = self.historical.close.pct_change()
        return self._pct_change_raw
    
    @property
    @assert_historical
    def std(self):
        if not self._std:
            self._std = self.historical.close.std()
        return self._std
    
    @assert_historical
    def std_slice(self, period_start: datetime, period_end: datetime):
        period_start, period_end = self.cleanse_incoming_datetime(period_start, period_end)
        return self.historical.close[period_start:period_end].std()
    
    def cleanse_incoming_datetime(self, start_date: Union[datetime, str], end_date: Union[datetime, str]):
        if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = start_date.replace(tzinfo=UTC)
        end_date = end_date.replace(tzinfo=UTC)
        return start_date, end_date
    
    @assert_historical
    def time_overlap(self, other: Asset, override_start: Union[datetime, str] = None, override_end: Union[datetime, str] = None):
        if not override_start or not override_end:
            date_range = [max(self.historical.index.min(), other.historical.index.min()), 
                          min(self.historical.index.max(), other.historical.index.max())]
            date_range = list(map(lambda x: x.strftime("%Y-%m-%d"), date_range))
        else:
            override_start, override_end = self.cleanse_incoming_datetime(override_start, override_end)
        start = override_start or date_range[0]
        end = override_end or date_range[1]
        return start, end
    
    @assert_historical
    def correlation(self, other: Asset, period_start: datetime = None, period_end: datetime = None) -> float:
        start, end = self.time_overlap(other, period_start, period_end)
        left = self.historical[start:end]
        right = other.historical[start:end]
        return left.close.corr(right.close)
    
    @assert_historical
    def alpha(self, benchmark: Asset, risk_free_return: float = 4.43, period_start: datetime = None, period_end: datetime = None):
        start, end = self.time_overlap(benchmark, period_start, period_end)
        portfolio_return = (self.historical.close[end] - self.historical.close[start]) / self.historical.close[start]
        benchmark_return = (benchmark.historical.close[end] - benchmark.historical.close[start]) / benchmark.historical.close[start]
        return portfolio_return - risk_free_return - self.beta(benchmark, period_start, period_end) * (benchmark_return - risk_free_return)

    @assert_historical
    def beta(self, benchmark: Asset, period_start: datetime = None, period_end: datetime = None) -> float:
        start, end = self.time_overlap(benchmark, period_start, period_end)
        return self.std_slice(start, end) / benchmark.std_slice(start, end) * self.correlation(benchmark, period_start, period_end)


    def __eq__(self, other: Asset):
        if not isinstance(other, Asset):
            raise TypeError("Can only compare with type Asset")
        return self.ticker == other.ticker
    
    def __repr__(self):
        return self.ticker
    
    def __hash__(self):
        return hash(self.ticker)

class AssetList:
    def __init__(self, assets: Dict[str, float]):
        self.tickers = list(assets.keys())
        for ticker, weight in assets.items():
            self.__dict__[ticker] = Asset(ticker, weight)
    
    def get(self, ticker):
        return self.__dict__[ticker]
    
    def __iter__(self):
        return iter([self.get(ticker) for ticker in self.tickers])
    
    def __len__(self):
        return len(self.tickers)
    
    def __repr__(self):
        return str({self.get(tkr): self.get(tkr).weight for tkr in self.tickers})