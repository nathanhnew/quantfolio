from __future__ import annotations
import asyncio
from .asset import Asset, AssetList
from typing import Dict, Union, List, Optional


class Portfolio:   
    def __init__(self, assets: Union[str, List[str], Dict[str, float]]):
        self._asset_weights: Dict[str, float] = self.validate_assets(assets)
        self._assets = AssetList(self._asset_weights)
        self._tickers: List[str] = self._assets.tickers
        self._benchmark = None
    
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
            if ticker not in self._tickers:
                raise ValueError(f"Unassigned ticker {ticker} provided in weights. Portfolio tickers: {self.tickers}")
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
    
    @property
    def benchmark(self):
        if not self._benchmark:
            self._benchmark = Portfolio('spy')
        return self._benchmark

    @benchmark.setter
    def benchmark(self, new_benchmark: Portfolio):
        """
        Set the benchmark portfolio for evaluation of relative success
        :param benchmark: new portfolio to use as a benchmark when evaluating current portfolio
        :return: Portfolio with equal weightings of benchmark tickers
        """
        if new_benchmark != self.benchmark:
            self._benchmark = new_benchmark

    
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