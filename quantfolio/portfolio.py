from __future__ import annotations
from .web import QuantfolioWebInterface
import asyncio
import aiohttp
from datetime import datetime
from .asset import Asset, AssetList
import pandas as pd
import numpy as np
from math import sqrt
from typing import Dict, Union, List, Optional


class Portfolio:   
    def __init__(self, assets: Union[str, List[str], Dict[str, float]], initial_value=10000):
        self._asset_weights: Dict[str, float] = self.validate_assets(assets)
        self._assets = AssetList(self._asset_weights)
        self.initial_value = initial_value
        self._tickers: List[str] = self._assets.tickers
        self._historical = pd.DataFrame()
        self._correlation = pd.DataFrame()
    
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
    
    @property
    def historical(self):
        if self._historical.empty:
            self._historical = pd.concat({asset.ticker: asset.historical for asset in self.assets})
        return self._historical
    
    @property
    def min_date(self):
        return self.historical.close.dropna().index.min()
    
    @property
    def max_date(self):
        return self.historical.close.dropna().index.max()
    
    @property
    def historical_asset_returns(self):
        historical_close = self.historical.close.unstack(level=0).dropna().asfreq('B')
        return np.log(historical_close).diff()
    
    @property
    def arithmetic_historical_asset_returns(self):
        historical_close = self.historical.close.unstack(level=0).dropna().asfreq('B')
        return historical_close.pct_change()
    
    @property
    def historical_returns(self):
        portfolio_close = self.historical_asset_returns * self.weights
        return portfolio_close.sum(axis=1)
    
    @property
    def arithmetic_historical_returns(self):
        portfolio_close = self.arithmetic_historical_asset_returns * self.weights
        return portfolio_close.sum(axis=1)
    
    @property
    def historical_value(self):
        historical_returns = self.historical_returns
        historical_value = historical_returns.to_frame(name='daily_return').reset_index()
        for ind, row in historical_value.iterrows():
            if ind == 0:
                dollar_value = self.initial_value
            else:
                dollar_value = historical_value.loc[ind - 1, 'daily_value'] * (1 + row['daily_return'])
            historical_value.loc[ind, 'daily_value'] = round(dollar_value, 2)
        return historical_value.set_index('index').rename_axis(index=None).asfreq('B').daily_value
    
    def historical_value_with_contributions(self, contribution_amount: float, contribution_frequency: str):
        if contribution_frequency not in ['d', 'w', 'm', 'q', 'y']:
            raise ValueError(f'Contribution frequency must be in one of the following [d, w, m, q, y]. Value: {contribution_frequency}')
        historical_returns = self.historical_returns
        historical_value = historical_returns.to_frame(name='daily_return').reset_index()
        for ind, row in historical_value.iterrows():
            if ind == 0:
                dollar_value = self.initial_value
            else:
                dollar_value = historical_value.loc[ind - 1, 'daily_value'] * (1 + row['daily_return'])
                if contribution_frequency == 'd' or (contribution_frequency == 'w' and row['index'].dayofweek == 0) or \
                    (contribution_frequency == 'm' and row['index'].is_month_start) or (contribution_frequency == 'q' and row['index'].is_quarter_start) or \
                    (contribution_frequency == 'y' and row['index'].is_year_start):
                    dollar_value += contribution_amount
            historical_value.loc[ind, 'daily_value'] = round(dollar_value, 2)
        return historical_value.set_index('index').rename_axis(index=None).asfreq('B').daily_value

    
    @property
    def covariance(self):
        return self.historical_returns.cov()
    
    @property
    def arithmetic_covariance(self):
        return self.arithmetic_historical_returns.cov()
    
    @property
    def weights(self):
        return np.array([asset.weight for asset in self.assets])
    
    @property
    def autocorrelation_matrix(self):
        return self.historical_asset_returns.corr()
    
    @property
    def autocorrelation(self):
        return self.autocorrelation_matrix.values[np.triu_indices_from(self.autocorrelation_matrix.values, 1)].mean()

    def sharpe_ratio(self, risk_free_rate=0.02, trading_days_per_year=252):
        avg_daily_returns = self.historical_returns.mean()
        portfolio_annualized_return = np.sum(avg_daily_returns * self.weights) * trading_days_per_year
        portfolio_standard_deviation = sqrt(np.dot(self.weights.T, np.dot(self.covariance, self.weights))) * sqrt(trading_days_per_year)
        return (portfolio_annualized_return - risk_free_rate) / portfolio_standard_deviation
    
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