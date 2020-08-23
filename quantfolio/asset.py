from __future__ import annotations
from typing import Dict

class Asset:
    def __init__(self, ticker: str, weight: float = 0):
        self.ticker = ticker
        self.weight = weight
    
    def __eq__(self, other: Asset):
        if not isinstance(other, Asset):
            raise ValueError("Can only compare with type Asset")
        return self.ticker == other.ticker
    
    def __repr__(self):
        return str({self.ticker: self.weight})
    
    def __hash__(self):
        return hash(self.ticker)

class AssetList:
    def __init__(self, assets: Dict[str, float]):
        self.tickers = assets.keys()
        for ticker, weight in assets.items():
            self.__dict__[ticker] = Asset(ticker, weight)
    
    def get(self, ticker):
        return self.__dict__[ticker]
    
    def __repr__(self):
        return str([self.get(tkr) for tkr in self.tickers])