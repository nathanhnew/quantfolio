import aiohttp

class Tiingo:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.tiingo.com/"
    
    async def close_history_async(self, ticker, session=None):
        url = self.base_url + f"/tiingo/daily/{ticker}/prices?start_date=1970-01-01"
        daily_close = {}

        need_close_session = False
        if not session:
            session = aiohttp.ClientSession()
            need_close_session = True
        
        parameters = {
            "token": self.api_key
        }
        async with session.get(url, params=parameters) as response:
            if response.status != 200:
                raise ValueError(f"Unable to recieve a valid response for ticker {ticker}")
            daily_prices = await response.json()
            for day in daily_prices:
                daily_close[day['date']] = day['adjClose']
        
        if need_close_session:
            await session.close()
        
        return daily_close