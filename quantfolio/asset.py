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
        self.weight = weight if weight <= 1 else weight / 100
        self.historical = pd.DataFrame()
        self._historical_return = pd.Series()
        self._std = pd.Series()
        self._arithmetic_historical_return = pd.Series()
    
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
        self.historical = df.asfreq('B').tz_convert(None).dropna()
        return
    
    async def get_historical_data_async(self, web_reader: QuantfolioWebInterface, session: aiohttp.ClientSession, start_date: str = '1970-01-01', end_date: str = datetime.now().strftime('%Y-%m-%d')) -> None:
        closing_prices = await web_reader.get_historical_close_async(self.ticker, session, start_date, end_date)
        df = pd.DataFrame.from_dict(closing_prices, orient='index')
        df.index = pd.to_datetime(df.index)
        self.historical = df.asfreq('B').tz_convert(None).dropna()
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
    def geometric_historical_return(self) -> float:
        if self._historical_return.empty:
            self._historical_return = np.log(self.historical.close).diff()
            self._historical_return.iat[0] = 0
        return self._historical_return
    
    @property
    @assert_historical
    def arithmetic_historical_return(self) -> float:
        if self._arithmetic_historical_return.empty:
            self._arithmetic_historical_return = self.historical.close.pct_change()
            self._arithmetic_historical_return.iat[0] = 0
        return self._arithmetic_historical_return
    
    @property
    @assert_historical
    def total_return(self):
        return (self.arithmetic_historical_return + 1).cumprod()[-1]
    
    @assert_historical
    def return_over_period(self, start: Union[datetime, str], end: Union[datetime, str]):
        start, end = self._cleanse_incoming_datetime(start, end)
        period_return = self.arithmetic_historical_return[start: end].copy()
        period_return.iat[0] = 0
        return (period_return + 1).cumprod()[-1]

    
    @property
    @assert_historical
    def std(self):
        if not self._std:
            self._std = self.historical.close.std()
        return self._std
    
    def _cleanse_incoming_datetime(self, start_date: Union[datetime, str], end_date: Union[datetime, str]):
        if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
        if end_date < start_date:
            raise ValueError(f"Period end must be after period beginning")
        if start_date < self.min_date or start_date > self.max_date:
            raise ValueError(f"Starting point out of range\n{start_date} must be between {self.min_date} and {self.max_date}")
        if end_date < self.min_date or end_date > self.max_date:
            raise ValueError(f"Ending point out of range\n{end_date} must be between {self.min_date} and {self.max_date}")
        return start_date, end_date
    
    @assert_historical
    def time_overlap(self, other: Asset, override_start: Union[datetime, str] = None, override_end: Union[datetime, str] = None):
        if not override_start or not override_end:
            date_range = [max(self.historical.index.min(), other.historical.index.min()), 
                          min(self.historical.index.max(), other.historical.index.max())]
            date_range = list(map(lambda x: x.strftime("%Y-%m-%d"), date_range))
        else:
            override_start, override_end = self._cleanse_incoming_datetime(override_start, override_end)
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
        asset_time_slice = self.geometric_historical_return[start: end].copy()
        benchmark_time_slice = benchmark.geometric_historical_return[start: end].copy()
        asset_time_slice.iat[0] = 0
        benchmark_time_slice.iat[0] = 0
        asset_return = (asset_time_slice + 1).cumprod()[-1]
        benchmark_return = (benchmark_time_slice + 1).cumprod()[-1]
        return asset_return - risk_free_return - self.beta(benchmark, period_start, period_end) * (benchmark_return - risk_free_return)

    @assert_historical
    def beta(self, benchmark: Asset, period_start: datetime = None, period_end: datetime = None) -> float:
        start, end = self.time_overlap(benchmark, period_start, period_end)
        return self.geometric_historical_return[start: end].std() / benchmark.geometric_historical_return[start: end].std() * self.correlation(benchmark, period_start, period_end)


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