import requests
from abc import ABC, abstractmethod
from datetime import datetime
import aiohttp

class QuantfolioWebInterface(ABC):
    @abstractmethod
    def get_historical_close(self, ticker, start_time, end_time):
        raise NotImplementedError

    @abstractmethod
    def get_historical_close_async(self, ticker, session, start_time, end_time):
        raise NotImplementedError

    @staticmethod
    def generate_async_session():
        return aiohttp.ClientSession()

class TiingoWebReader(QuantfolioWebInterface):
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.tiingo.com/"
        if not self.valid_key:
            raise ValueError("Invalid key provided")
    
    @property
    def valid_key(self):
        url = self.base_url + "api/test"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}"
        }
        with requests.get(url, headers=headers) as response:
            return response.json().get("message") == "You successfully sent a request"
    
    async def get_historical_close_async(self, ticker, session, start_date='1970-01-01', end_date=datetime.now().strftime('%Y-%m-%d')):
        url = self.base_url + f"tiingo/daily/{ticker}/prices"
        daily_close = {}
        
        parameters = {
            "token": self.api_key,
            "startDate": start_date,
            "endDate": end_date
        }

        async with session.get(url, params=parameters, raise_for_status=True) as response:
            daily_prices = await response.json()
            for day in daily_prices:
                daily_close[day['date']] = {'high': day['adjHigh'], 'low': day['adjLow'], 
                                            'open': day['adjOpen'], 'close': day['adjClose'],
                                            'volume': day['adjVolume'], 'dividend': day['divCash']}
        
        return daily_close
    
    def get_historical_close(self, ticker, start_date='1970-01-01', end_date=datetime.now().strftime('%Y-%m-%d')):
        session = requests.Session()
        url = self.base_url + f"tiingo/daily/{ticker}/prices"
        daily_close = {}

        parameters = {
            "token": self.api_key,
            "startDate": start_date,
            "endDate": end_date
        }

        with session.get(url, params=parameters) as response:
            response.raise_for_status()
            daily_prices = response.json()
            for day in daily_prices:
                daily_close[day['date']] = {'high': day['adjHigh'], 'low': day['adjLow'], 
                                            'open': day['adjOpen'], 'close': day['adjClose'],
                                            'volume': day['adjVolume'], 'dividend': day['divCash']}
        
        return daily_close