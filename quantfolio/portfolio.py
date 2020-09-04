from __future__ import annotations
from .web import QuantfolioWebInterface
import asyncio
import aiohttp
from datetime import datetime
from .asset import Asset, AssetList
import pandas as pd
import numpy as np
from math import sqrt
from functools import wraps
from typing import Dict, Union, List, Optional


class Portfolio:   
    def __init__(self, assets: Union[str, List[str], Dict[str, float]], initial_value=10000):
        self._asset_weights: Dict[str, float] = self.validate_assets(assets)
        self._assets = AssetList(self._asset_weights)
        self.initial_value = initial_value
        self._tickers: List[str] = self._assets.tickers
        self._historical = pd.DataFrame()
        self._correlation = pd.DataFrame()
        self._historical_return = pd.Series()
        self._arithmetic_historical_return = pd.Series()
    
    def validate_assets(self, incoming_assets):
        """
        Function to verify the integrity of incoming asset definitions.

        Assets can be defined in three ways. A single ticker, a list of tickers, or a dictionary of tickers with corresponding weights.
        assets = 'spy'
        assets = ['fb', 'aapl', 'amzn', 'nflx', 'googl']
        assets = {'msft': 0.8, 'tsla': 0.2}

        If no weights are provided, equal weights will be assigned.

        :param assets: Input asset parameter to verify
        :return: dictionary of input assets and associated weights
        """
        if not isinstance(incoming_assets, (str, list, dict)) or \
            not all(isinstance(tkr, str) for tkr in incoming_assets) or \
            (isinstance(incoming_assets, dict) and not all(isinstance(val, float) for val in incoming_assets.values())):
            raise ValueError("Portfolio asset values must be a string, list of strings, or dictionary of type string: float")
        if isinstance(incoming_assets, dict):
            incoming_assets = self.validate_weights(incoming_assets)
        else:
            incoming_assets = self.equal_weights(incoming_assets) if isinstance(incoming_assets, list) else self.equal_weights([incoming_assets])
        return incoming_assets
    
    def validate_weights(self, weights) -> Dict[Asset, float]:
        """
        Function to verify the integrity of the weights provided.

        Weights must be dictionary with key-value pairs corresponding to the ticker and weight
        """
        if not isinstance(weights, dict) or not all(isinstance(tkr, str) & isinstance(val, float) for tkr, val in weights.items()):
            raise ValueError("Weights values must be a {ticker: weight} dict of type with string type keys and float type values")
        for ticker, weight in weights.items():
            if weight > 1:
                weights[ticker] = weight / 100
        if not sum(weights.values()) == 1:
            raise ValueError(f"Unbalanced weights. Weights must equal 100% of the portfolio. Provided weights {sum(weights.values()) * 100}%")
        return {ticker: weight for ticker, weight in weights.items()}
    
    @property
    def assets(self):
        return self._assets

    @property
    def tickers(self):
        return list(self._tickers)
    
    def equal_weights(self, assets) -> Dict[str, float]:
        """
        Function to equally weigh securities in the portfolio. Used as default weight scheme
        :return: dictionary with key-value pairs of {ticker: weight}
        """
        return {ticker: 1 / len(assets) for ticker in assets}
    
    def with_weights(self, weights: Dict[str, float]) -> Portfolio:
        """
        Function to return a new Portfolio object with specific weights defined
        :param weights: dictionary with key-value pairs of 
        """
        return Portfolio(weights)
    
    def get_historical_data(self, reader: QuantfolioWebInterface):
        for asset in self.assets:
            asset.get_historical_data(reader)
        return
    
    async def get_historical_data_async(self, reader: QuantfolioWebInterface, start_date: str = '1970-01-01', end_date: str = datetime.now().strftime('%Y-%m-%d'), progress_bar=False):
        async with aiohttp.ClientSession() as session:
            asset_iterator = asyncio.as_completed([asset.get_historical_data_async(reader, session, start_date, end_date) for asset in self.assets])
            if progress_bar:
                import tqdm
                asset_iterator = tqdm.tqdm(asset_iterator, total=len(self.assets))
            for ret in asset_iterator:
                await ret
    
    def assert_historical(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if args[0].historical is None or args[0].historical.empty:
                raise ValueError("Historical data must be defined for portfolio before calling this function. Please call .get_historical_data() or .get_historical_data_async() first")
            value = func(*args, **kwargs)
            return value
        return wrapper
    
    @property
    def historical(self):
        if self._historical.empty:
            self._historical = pd.concat({asset.ticker: asset.historical for asset in self.assets}).dropna()
        return self._historical
    
    @property
    @assert_historical
    def min_date(self):
        return self.historical.close.unstack(level=0).dropna().index.min()
    
    @property
    @assert_historical
    def max_date(self):
        return self.historical.close.unstack(level=0).dropna().index.max()
    
    @property
    @assert_historical
    def historical_asset_returns(self):
        historical_close = self.historical.close.unstack(level=0).dropna()
        historical_close = np.log(historical_close).diff()
        return historical_close
    
    @property
    @assert_historical
    def arithmetic_historical_asset_returns(self):
        historical_close = self.historical.close.unstack(level=0).dropna()
        return historical_close.pct_change()
    
    @property
    @assert_historical
    def geometric_historical_return(self):
        if self._historical_return.empty:
            portfolio_close = self.historical_asset_returns * self.weights
            self._historical_return = portfolio_close.sum(axis=1)
            self._historical_return.iat[0] = 0
        return self._historical_return
    
    @property
    @assert_historical
    def arithmetic_historical_return(self):
        if self._arithmetic_historical_return.empty:
            portfolio_close = self.arithmetic_historical_asset_returns * self.weights
            self._arithmetic_historical_return = portfolio_close.sum(axis=1)
            self._arithmetic_historical_return.iat[0] = 0
        return self._arithmetic_historical_return
    
    @property
    @assert_historical
    def historical_value(self):
        return (((self.arithmetic_historical_return + 1).cumprod()) * self.initial_value).round(2)
    
    @property
    @assert_historical
    def total_return(self):
        return (self.arithmetic_historical_return + 1).cumprod()[-1]
    
    @assert_historical
    def historical_value_with_contributions(self, contribution_amount: float, contribution_frequency: str):
        if contribution_frequency not in ['d', 'w', 'm', 'q', 'y']:
            raise ValueError(f'Contribution frequency must be in one of the following [d, w, m, q, y]. Value: {contribution_frequency}')
        historical_return = self.geometric_historical_return
        historical_value = historical_return.to_frame(name='daily_return').reset_index()
        for ind, row in historical_value.iterrows():
            if ind == 0:
                dollar_value = self.initial_value
            else:
                dollar_value = historical_value.loc[ind - 1, 'daily_value'] * (1 + row['daily_return'])
                if contribution_frequency == 'd' or (contribution_frequency == 'w' and row['index'].dayofweek == 0) or \
                    (contribution_frequency == 'm' and row['index'].is_month_start) or (contribution_frequency == 'q' and row['index'].is_quarter_start) or \
                    (contribution_frequency == 'y' and row['index'].is_year_start):
                    dollar_value += contribution_amount
            historical_value.loc[ind, 'daily_value'] = dollar_value
        return historical_value.set_index('index').rename_axis(index=None).daily_value.round(2)
    
    @property
    @assert_historical
    def weights(self):
        return np.array([asset.weight for asset in self.assets])
    
    @property
    @assert_historical
    def autocorrelation_matrix(self):
        return self.historical_asset_returns.corr()
    
    @property
    @assert_historical
    def autocorrelation(self):
        return self.autocorrelation_matrix.values[np.triu_indices_from(self.autocorrelation_matrix.values, 1)].mean()
    
    @assert_historical
    def return_over_period(self, start: Union[datetime, str], end: Union[datetime, str]):
        start, end = self._date_string_to_datetime(start, end)
        if end < start:
            raise ValueError(f"Period end must be after period beginning")
        if start < self.min_date or start > self.max_date:
            raise ValueError(f"Starting point out of range\n{start} must be between {self.min_date} and {self.max_date}")
        if end < self.min_date or end > self.max_date:
            raise ValueError(f"Ending point out of range\n{end} must be between {self.min_date} and {self.max_date}")
        period_return = self.arithmetic_historical_return[start: end].copy()
        period_return.iat[0] = 0
        return (period_return + 1).cumprod()[-1]
    
    @assert_historical
    def correlation(self, other: Portfolio, range_start: datetime = None, range_end: datetime = None) -> float:
        range_start, range_end = self._time_overlap(other, range_start, range_end)
        left = self.geometric_historical_return[range_start: range_end]
        right = other.geometric_historical_return[range_start: range_end]
        return left.corr(right)
    
    @assert_historical
    def portfolio_time_overlap(self, other: Portfolio):
        return [max(self.min_date, other.min_date), min(self.max_date, other.max_date)]
    
    def _date_string_to_datetime(self, range_start: Union[datetime, str] = None, range_end: Union[datetime, str] = None):
        if range_start and isinstance(range_start, str):
            range_start = datetime.strptime(range_start, "%Y-%m-%d")
        if range_end and isinstance(range_end, str):
            range_end = datetime.strptime(range_end, "%Y-%m-%d")        
        return range_start, range_end
    
    def _time_overlap(self, other, period_start: Union[datetime, str] = None, period_end: Union[datetime, str] = None) -> [datetime, datetime]:
        period_start, period_end = self._date_string_to_datetime(period_start, period_end)
        period_start = period_start or self.portfolio_time_overlap(other)[0]
        period_end = period_end or self.portfolio_time_overlap(other)[1]
        if period_start < self.min_date or period_start < other.min_date:
            raise ValueError(f"Starting period out of range\n{period_start} provided must be greater than {self.min_date} and {other.min_date}")
        if period_end > self.max_date or period_start > other.max_date:
            raise ValueError(f"Ending period out of range\n{period_end} provided must be less than {self.max_date} and {other.max_date}")
        return period_start, period_end

    @assert_historical
    def alpha(self, benchmark: Portfolio, risk_free_rate=0.0245, period_start: Union[datetime, str] = None, period_end: Union[datetime, str] = None):
        period_start, period_end = self._time_overlap(benchmark, period_start, period_end)
        portfolio_time_slice = self.geometric_historical_return[period_start: period_end].copy()
        benchmark_time_slice = benchmark.geometric_historical_return[period_start:period_end].copy()
        portfolio_time_slice.iat[0] = 0
        benchmark_time_slice.iat[0] = 0
        portfolio_return = (portfolio_time_slice + 1).cumprod()[-1]
        benchmark_return = (benchmark_time_slice + 1).cumprod()[-1]
        return portfolio_return - risk_free_rate - self.beta(benchmark, period_start, period_end) * (benchmark_return - risk_free_rate)
    
    @assert_historical
    def beta(self, benchmark: Portfolio, period_start: Union[datetime, str] = None, period_end: Union[datetime, str] = None):
        period_start, period_end = self._time_overlap(benchmark, period_start, period_end)
        return self.geometric_historical_return[period_start:period_end].std() / benchmark.geometric_historical_return[period_start: period_end].std() * self.correlation(benchmark, period_start, period_end)
        
    @assert_historical
    def sharpe_ratio(self, risk_free_rate=0.0245, trading_days_per_year=252):
        avg_daily_returns = self.geometric_historical_return.mean()
        daily_rfr = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
        portfolio_standard_deviation = self.geometric_historical_return.std()
        return (avg_daily_returns - daily_rfr) / portfolio_standard_deviation * sqrt(trading_days_per_year)
    
    @assert_historical
    def sortino_ratio(self, risk_free_rate=0.0245, trading_days_per_year=252):
        avg_daily_returns = self.geometric_historical_return.mean()
        daily_rfr = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
        downside_deviation = self.geometric_historical_return.add(risk_free_rate * -1).where(lambda ret: ret < 0).dropna().std()
        return (avg_daily_returns - daily_rfr) / downside_deviation * sqrt(trading_days_per_year)
    
    def __repr__(self):
        return str(self._assets)
    
    def __iter__(self):
        for asset, weight in self._assets.items():
            yield asset, weight
    
    def __eq__(self, other: Portfolio) -> bool:
        for ticker, weight in other:
            if ticker not in self._assets or not self._assets[ticker] == weight:
                return False
        return True