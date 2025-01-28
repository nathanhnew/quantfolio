import asyncio
import logging
import os
from abc import ABC, abstractmethod
from asyncio import Semaphore
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd


class WebBackend(ABC):
    @abstractmethod
    def fetch_asset_history(
        self, ticker: str, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> pd.DataFrame: ...


class FMP(WebBackend):
    URL = "https://financialmodelingprep.com"
    HISTORICAL_ENDPOINT = "api/v3/historical-price-full"

    def __init__(self, api_key: Optional[str]):
        self.api_key = self._api_key_or_env(api_key)
        self.logger = logging.getLogger(__name__)

    def _api_key_or_env(self, key: Optional[str]) -> str:
        if key:
            return key
        elif env_key := os.getenv("FMP_API_KEY"):
            return env_key
        else:
            raise ValueError(
                "FMP API Key undefined. "
                "Either pass key with 'api_key' parameter "
                "or set FMP_API_KEY environment variable."
            )

    def fetch_asset_history(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        endpoint = "/".join([self.URL, self.HISTORICAL_ENDPOINT, ticker])
        params = {"apikey": self.api_key}
        if start_date is not None:
            params["from"] = start_date.strftime("%Y-%m-%d")

        if end_date is not None:
            params["to"] = end_date.strftime("%Y-%m-%d")

        resp = httpx.get(endpoint, params=params)

        df = pd.DataFrame.from_records(resp.json().get("historical", []))
        df["ticker"] = ticker

        df.set_index("date", inplace=True)
        return df

    async def async_fetch_history(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        client: Optional[httpx.AsyncClient] = None,
        retries: int = 3,
        backoff_factor: float = 1.25,
        rate_limiter: Optional[Semaphore] = None,
    ) -> pd.DataFrame:
        if client is None:
            client = httpx.AsyncClient(timeout=30)

        endpoint = "/".join([self.URL, self.HISTORICAL_ENDPOINT, ticker])
        params = {"apikey": self.api_key}
        if start_date is not None:
            params["from"] = start_date.strftime("%Y-%m-%d")

        if end_date is not None:
            params["to"] = end_date.strftime("%Y-%m-%d")

        for attempt in range(retries):
            self.logger.debug(f"[FMP Backend|{ticker}|{attempt}] Fetching...")
            try:
                if rate_limiter:
                    async with rate_limiter:
                        self.logger.debug(f"[FMP Backend|{ticker}|{attempt}] Set rate limiter")
                        response = await client.get(endpoint, params=params)
                else:
                    response = await client.get(endpoint, params=params)
                self.logger.debug(f"[FMP Backend|{ticker}|{attempt}] Fetching complete. Evaluating response")
                response.raise_for_status()
                df = pd.DataFrame.from_records(response.json().get("historical", []))
                df["ticker"] = ticker
                df.set_index("date", inplace=True)
                return df

            except httpx.HTTPError as e:
                self.logger.warning(f"[{ticker}] Error fetching: {e}")
                if attempt == retries - 1:
                    logging.error(f"[{ticker}] Failed to fetch from FMP: {e}")
                    raise

                await asyncio.sleep(backoff_factor**attempt)

        raise RuntimeError("Failed to fetch data. Reached all attempts without erroring out")
