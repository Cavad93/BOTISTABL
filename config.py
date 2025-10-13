import os
from typing import Dict, List

class Config:
    # Binance API настройки
    API_KEY = os.getenv('BINANCE_API_KEY', '')
    API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    
    # Торговые параметры
    TOP_COINS_COUNT = 200  # Количество топовых монет для анализа
    MIN_VOLUME_24H = 1000000  # Минимальный объем торгов за 24ч в USDT
    
    # Таймфреймы для анализа
    TIMEFRAMES = ['15m', '1h', '4h', '1d']
    
    # Параметры уровней сопротивления
    MIN_TOUCHES = 2  # Минимальное количество касаний для валидного уровня
    LEVEL_TOLERANCE = 0.002  # Допустимое отклонение от уровня (0.2%)
    
    # Критерии пробоя
    BREAKOUT_PERCENT = 0.005  # Процент пробоя уровня (0.5%)
    VOLUME_SURGE_MULTIPLIER = 1.5  # Увеличение объема в 1.5 раза
    BREAKOUT_CONFIRMATION_CANDLES = 2  # Количество свечей для подтверждения
    RSI_THRESHOLD = 60  # Минимальный RSI для входа
    
    # Скользящие средние
    EMA_PERIODS = [50, 100, 200]
    
    # Уровни Фибоначчи
    FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
    
    # Параметры риск-менеджмента
    POSITION_SIZE_PERCENT = 0.02  # 2% от депозита на позицию
    MAX_CONCURRENT_POSITIONS = 5
    STOP_LOSS_PERCENT = 0.02  # 2% стоп-лосс
    TAKE_PROFIT_PERCENT = 0.05  # 5% тейк-профит
    
    # Интервалы обновления (в секундах)
    SCAN_INTERVAL = 60  # Сканирование рынка каждую минуту
    LEVEL_UPDATE_INTERVAL = 3600  # Обновление уровней каждый час
