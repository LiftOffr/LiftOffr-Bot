import requests
import json
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any
from utils import get_nonce, get_kraken_signature
from config import API_KEY, API_SECRET, API_URL, API_VERSION, USE_SANDBOX

logger = logging.getLogger(__name__)

class KrakenAPI:
    """
    Class to interact with Kraken's REST API
    """
    def __init__(self, api_key: str = API_KEY, api_secret: str = API_SECRET):
        """
        Initialize the KrakenAPI class
        
        Args:
            api_key (str): Kraken API key
            api_secret (str): Kraken API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = API_URL
        self.api_version = API_VERSION
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Kraken Trading Bot'
        })
        
        # Using sandbox if specified
        self.using_sandbox = USE_SANDBOX
        if self.using_sandbox:
            logger.warning("Using Kraken in sandbox/test mode")
    
    def _request(self, method: str, endpoint: str, data: Dict = None, private: bool = False) -> Dict:
        """
        Make a request to the Kraken API
        
        Args:
            method (str): HTTP method (GET, POST)
            endpoint (str): API endpoint
            data (dict, optional): Request data
            private (bool, optional): Whether this is a private API call requiring authentication
        
        Returns:
            dict: API response
        """
        url = f"{self.api_url}/{self.api_version}/{endpoint}"
        
        if private:
            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret required for private endpoints")
            
            if data is None:
                data = {}
            
            data['nonce'] = get_nonce()
            
            # Add validate flag for test mode if sandbox is enabled
            if self.using_sandbox and endpoint in ['AddOrder', 'CancelOrder']:
                data['validate'] = True
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': get_kraken_signature(f"/{self.api_version}/{endpoint}", data, self.api_secret)
            }
            
            response = self.session.post(url, headers=headers, data=data)
        else:
            if method.upper() == 'GET':
                response = self.session.get(url, params=data)
            else:
                response = self.session.post(url, data=data)
        
        # Handle HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        
        # Check for API errors
        if result.get('error') and len(result['error']) > 0:
            error_msg = ", ".join(result['error'])
            logger.error(f"Kraken API error: {error_msg}")
            raise Exception(f"Kraken API error: {error_msg}")
        
        return result.get('result', {})
    
    def get_server_time(self) -> Dict:
        """
        Get Kraken server time
        
        Returns:
            dict: Server time information
        """
        return self._request('GET', 'public/Time')
    
    def get_asset_info(self, assets: List[str] = None) -> Dict:
        """
        Get asset information
        
        Args:
            assets (list, optional): List of assets to get information for
        
        Returns:
            dict: Asset information
        """
        data = {}
        if assets:
            data['asset'] = ','.join(assets)
        
        return self._request('GET', 'public/Assets', data)
    
    def get_tradable_asset_pairs(self, pairs: List[str] = None) -> Dict:
        """
        Get tradable asset pairs
        
        Args:
            pairs (list, optional): List of asset pairs to get information for
        
        Returns:
            dict: Asset pair information
        """
        data = {}
        if pairs:
            data['pair'] = ','.join(pairs)
        
        return self._request('GET', 'public/AssetPairs', data)
    
    def get_ticker(self, pairs: List[str]) -> Dict:
        """
        Get ticker information
        
        Args:
            pairs (list): List of asset pairs to get ticker for
        
        Returns:
            dict: Ticker information
        """
        data = {'pair': ','.join(pairs)}
        return self._request('GET', 'public/Ticker', data)
    
    def get_ohlc(self, pair: str, interval: int = 1, since: int = None) -> Dict:
        """
        Get OHLC data
        
        Args:
            pair (str): Asset pair
            interval (int, optional): Time frame interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since (int, optional): Return committed OHLC data since given ID
        
        Returns:
            dict: OHLC data
        """
        data = {'pair': pair, 'interval': interval}
        if since:
            data['since'] = since
        
        return self._request('GET', 'public/OHLC', data)
    
    def get_order_book(self, pair: str, count: int = 100) -> Dict:
        """
        Get order book
        
        Args:
            pair (str): Asset pair
            count (int, optional): Maximum number of asks/bids
        
        Returns:
            dict: Order book
        """
        data = {'pair': pair, 'count': count}
        return self._request('GET', 'public/Depth', data)
    
    def get_recent_trades(self, pair: str, since: int = None) -> Dict:
        """
        Get recent trades
        
        Args:
            pair (str): Asset pair
            since (int, optional): Return trade data since given ID
        
        Returns:
            dict: Recent trades
        """
        data = {'pair': pair}
        if since:
            data['since'] = since
        
        return self._request('GET', 'public/Trades', data)
    
    def get_account_balance(self) -> Dict:
        """
        Get account balance
        
        Returns:
            dict: Account balance
        """
        return self._request('POST', 'Balance', private=True)
    
    def get_trade_balance(self, asset: str = 'ZUSD') -> Dict:
        """
        Get trade balance
        
        Args:
            asset (str, optional): Base asset for balance
        
        Returns:
            dict: Trade balance
        """
        data = {'asset': asset}
        return self._request('POST', 'TradeBalance', data, private=True)
    
    def get_open_orders(self, trades: bool = False, userref: int = None) -> Dict:
        """
        Get open orders
        
        Args:
            trades (bool, optional): Whether to include trades
            userref (int, optional): Restrict results to given user reference ID
        
        Returns:
            dict: Open orders
        """
        data = {'trades': trades}
        if userref:
            data['userref'] = userref
        
        return self._request('POST', 'OpenOrders', data, private=True)
    
    def get_closed_orders(self, trades: bool = False, userref: int = None, 
                         start: int = None, end: int = None, ofs: int = None) -> Dict:
        """
        Get closed orders
        
        Args:
            trades (bool, optional): Whether to include trades
            userref (int, optional): Restrict results to given user reference ID
            start (int, optional): Starting unix timestamp or order tx ID
            end (int, optional): Ending unix timestamp or order tx ID
            ofs (int, optional): Result offset for pagination
        
        Returns:
            dict: Closed orders
        """
        data = {'trades': trades}
        if userref:
            data['userref'] = userref
        if start:
            data['start'] = start
        if end:
            data['end'] = end
        if ofs:
            data['ofs'] = ofs
        
        return self._request('POST', 'ClosedOrders', data, private=True)
    
    def get_trades_history(self, type_: str = 'all', trades: bool = False, 
                          start: int = None, end: int = None, ofs: int = None) -> Dict:
        """
        Get trades history
        
        Args:
            type_ (str, optional): Type of trade (all, any position, closed position, closing position, no position)
            trades (bool, optional): Whether to include trades
            start (int, optional): Starting unix timestamp or order tx ID
            end (int, optional): Ending unix timestamp or order tx ID
            ofs (int, optional): Result offset for pagination
        
        Returns:
            dict: Trades history
        """
        data = {'type': type_, 'trades': trades}
        if start:
            data['start'] = start
        if end:
            data['end'] = end
        if ofs:
            data['ofs'] = ofs
        
        return self._request('POST', 'TradesHistory', data, private=True)
    
    def place_order(self, pair: str, type_: str, ordertype: str, volume: Union[str, float, Decimal], 
                  price: Union[str, float, Decimal] = None, price2: Union[str, float, Decimal] = None, 
                  leverage: Union[str, int] = None, oflags: List[str] = None, 
                  starttm: int = 0, expiretm: int = 0, userref: int = None, 
                  validate: bool = None) -> Dict:
        """
        Place a new order
        
        Args:
            pair (str): Asset pair
            type_ (str): Type of order (buy/sell)
            ordertype (str): Order type (market/limit/stop-loss/take-profit/stop-loss-limit/take-profit-limit/settle-position)
            volume (str/float/Decimal): Order volume in base currency
            price (str/float/Decimal, optional): Price (dependent on ordertype)
            price2 (str/float/Decimal, optional): Secondary price (dependent on ordertype)
            leverage (str/int, optional): Amount of leverage desired
            oflags (list, optional): List of order flags (post, fcib, fciq, nompp, viqc)
            starttm (int, optional): Scheduled start time
            expiretm (int, optional): Expiration time
            userref (int, optional): User reference ID
            validate (bool, optional): Validate inputs only (does not submit order)
        
        Returns:
            dict: Order information
        """
        data = {
            'pair': pair,
            'type': type_,
            'ordertype': ordertype,
            'volume': str(volume)
        }
        
        if price:
            data['price'] = str(price)
        if price2:
            data['price2'] = str(price2)
        if leverage:
            data['leverage'] = str(leverage)
        if oflags:
            data['oflags'] = ','.join(oflags)
        if starttm:
            data['starttm'] = starttm
        if expiretm:
            data['expiretm'] = expiretm
        if userref:
            data['userref'] = userref
        
        # Use validate parameter or fallback to config setting
        if validate is not None:
            data['validate'] = validate
        elif self.using_sandbox:
            data['validate'] = True
        
        return self._request('POST', 'AddOrder', data, private=True)
    
    def cancel_order(self, txid: str) -> Dict:
        """
        Cancel open order
        
        Args:
            txid (str): Transaction ID
        
        Returns:
            dict: Cancellation result
        """
        data = {'txid': txid}
        return self._request('POST', 'CancelOrder', data, private=True)
    
    def cancel_all_orders(self) -> Dict:
        """
        Cancel all open orders
        
        Returns:
            dict: Cancellation result
        """
        return self._request('POST', 'CancelAll', private=True)
