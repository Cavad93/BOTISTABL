import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, api_key: str, api_secret: str):
        """Инициализация менеджера данных Binance"""
        self.client = Client(api_key, api_secret)
        self.symbol_info = {}
        self.update_symbol_info()
    
    def update_symbol_info(self):
        """Обновление информации о торговых парах"""
        try:
            exchange_info = self.client.get_exchange_info()
            for symbol in exchange_info['symbols']:
                if symbol['status'] == 'TRADING':
                    self.symbol_info[symbol['symbol']] = symbol
        except Exception as e:
            logger.error(f"Ошибка обновления информации о символах: {e}")
    
    def get_top_coins(self, count: int = 200, min_volume: float = 1000000) -> List[str]:
        """
        Получение топовых криптовалют по капитализации
        
        Args:
            count: Количество монет
            min_volume: Минимальный объем торгов за 24ч
        
        Returns:
            Список символов торговых пар
        """
        try:
            tickers = self.client.get_ticker()
            usdt_pairs = []
            
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol.endswith('USDT') and symbol != 'USDT':
                    volume = float(ticker['quoteVolume'])
                    if volume >= min_volume:
                        usdt_pairs.append({
                            'symbol': symbol,
                            'volume': volume,
                            'price': float(ticker['lastPrice']),
                            'priceChangePercent': float(ticker['priceChangePercent'])
                        })
            
            # Сортировка по объему (как прокси для капитализации)
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            
            return [pair['symbol'] for pair in usdt_pairs[:count]]
        
        except Exception as e:
            logger.error(f"Ошибка получения топовых монет: {e}")
            return []
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Получение исторических данных свечей
        
        Args:
            symbol: Торговая пара
            interval: Временной интервал
            limit: Количество свечей
        
        Returns:
            DataFrame с OHLCV данными
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Конвертация типов
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)
            
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Расчет индикатора RSI
        
        Args:
            prices: Серия цен закрытия
            period: Период расчета
        
        Returns:
            Серия значений RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Расчет экспоненциальной скользящей средней
        
        Args:
            prices: Серия цен
            period: Период EMA
        
        Returns:
            Серия значений EMA
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return 0.0
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Получение стакана заявок"""
        try:
            return self.client.get_order_book(symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"Ошибка получения стакана для {symbol}: {e}")
            return {'bids': [], 'asks': []}
