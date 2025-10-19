import os
from dotenv import load_dotenv

class Config:
    """Глобальные настройки бота"""

    # ===== Binance API =====
    API_KEY = os.getenv('BINANCE_API_KEY', '')
    API_SECRET = os.getenv('BINANCE_API_SECRET', '')

    # Режим торговли: 'PAPER' (по умолчанию) или 'LIVE'
    TRADING_MODE = os.getenv('TRADING_MODE', 'PAPER')

    # Депозит по умолчанию для PAPER-режима (в USDT)
    DEFAULT_DEPOSIT_USDT = float(os.getenv('DEFAULT_DEPOSIT_USDT', '1000'))

    # ===== Путь к состоянию (для переживания рестарта) =====
    STATE_DIR = os.getenv('STATE_DIR', 'state')
    POSITIONS_STATE_PATH = os.getenv('POSITIONS_STATE_PATH', os.path.join(STATE_DIR, 'positions.json'))
    PAPER_STATE_PATH = os.getenv('PAPER_STATE_PATH', os.path.join(STATE_DIR, 'paper_state.json'))


    # ===== Рынок / Сканирование =====
# ===== Рынок / Сканирование =====
    TOP_COINS_COUNT = int(os.getenv('TOP_COINS_COUNT', '200'))
    MIN_VOLUME_24H = float(os.getenv('MIN_VOLUME_24H', '1000000'))

    # Исключение стейблкоинов (динамически + по списку базовых активов)
    EXCLUDE_STABLES = os.getenv('EXCLUDE_STABLES', 'true').lower() == 'true'
    STABLE_BASES = os.getenv(
        'STABLE_BASES',
        'USDT,USDC,BUSD,DAI,TUSD,USDP,USDJ,FDUSD,UST,USTC,SUSD,USDD,USDX'
    ).split(',')
    STABLE_PEG_LOW = float(os.getenv('STABLE_PEG_LOW', '0.9'))
    STABLE_PEG_HIGH = float(os.getenv('STABLE_PEG_HIGH', '1.1'))
    STABLE_MAX_CV = float(os.getenv('STABLE_MAX_CV', '0.02'))  # коэф. вариации цены
    STABLE_LOOKBACK_DAYS = int(os.getenv('STABLE_LOOKBACK_DAYS', '7'))
    STABLE_24H_PCT_MAX = float(os.getenv('STABLE_24H_PCT_MAX', '3.0'))

    # Таймфреймы для анализа
    TIMEFRAMES = os.getenv('TIMEFRAMES', '15m,1h,4h,1d').split(',')


    # Минимальная глубина истории на каждом таймфрейме (дней)
    MIN_HISTORY_DAYS = int(os.getenv('MIN_HISTORY_DAYS', '30'))

    # Интервалы обновления (секунды)
    SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL', '60'))            # каждые X секунд
    LEVEL_UPDATE_INTERVAL = int(os.getenv('LEVEL_UPDATE_INTERVAL', '3600'))
    MANAGE_POSITIONS_INTERVAL = int(os.getenv('MANAGE_POSITIONS_INTERVAL', '30'))

    # ===== Уровни сопротивления =====
    MIN_TOUCHES = int(os.getenv('MIN_TOUCHES', '2'))                 # сила уровня: сколько касаний минимум
    LEVEL_TOLERANCE = float(os.getenv('LEVEL_TOLERANCE', '0.005'))   # 0.5% слияние близких уровней
    EMA_PERIODS = [int(x) for x in os.getenv('EMA_PERIODS', '50,100,200').split(',')]
    FIBONACCI_LEVELS = [float(x) for x in os.getenv('FIBONACCI_LEVELS', '0.236,0.382,0.5,0.618,0.786').split(',')]

    # ===== Критерии пробоя =====
    BREAKOUT_PERCENT = float(os.getenv('BREAKOUT_PERCENT', '0.001'))  # 0.2% буфер
    VOLUME_SURGE_MULTIPLIER = float(os.getenv('VOLUME_SURGE_MULTIPLIER', '1.2'))  # 120% среднего
    BREAKOUT_CONFIRMATION_CANDLES = int(os.getenv('BREAKOUT_CONFIRMATION_CANDLES', '1'))  # достаточно 1
    RSI_THRESHOLD = int(os.getenv('RSI_THRESHOLD', '55'))


    # ===== Риск-менеджмент =====
# СТАЛО: config.py (фрагмент внутри class Config)
    # ===== Риск-менеджмент =====
    POSITION_SIZE_PERCENT = float(os.getenv('POSITION_SIZE_PERCENT', '0.02'))  # аллокация на сделку (legacy)
    RISK_PER_TRADE_PERCENT = float(os.getenv('RISK_PER_TRADE_PERCENT', '0.0035'))  # риск на сделку 0.35%
    MAX_CONCURRENT_POSITIONS = int(os.getenv('MAX_CONCURRENT_POSITIONS', '5'))
    STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', '0.02'))  # базовый SL 2%
    TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', '0.05'))  # базовый TP 5%
    PORTFOLIO_MAX_NOTIONAL = float(os.getenv('PORTFOLIO_MAX_NOTIONAL', '0.80'))  # совокупная НОМИНАЛЬНАЯ аллокация ≤80%


    # Трейлинг-стоп и частичная фиксация
    TRAILING_START = float(os.getenv('TRAILING_START', '0.03'))  # начать подтягивать SL после +3%
    TRAILING_STEP = float(os.getenv('TRAILING_STEP', '0.01'))    # шаг подтягивания 1%
    PARTIAL_TPS = [float(x) for x in os.getenv('PARTIAL_TPS', '0.03,0.05').split(',')]
    PARTIAL_TP_SIZES = [float(x) for x in os.getenv('PARTIAL_TP_SIZES', '0.5,0.25').split(',')]  # 50% и 25%

    # ===== Вебсокеты =====
    WEBSOCKETS_ENABLED = os.getenv('WEBSOCKETS_ENABLED', 'true').lower() == 'true'
    WS_RETRY_SEC = int(os.getenv('WS_RETRY_SEC', '5'))

    # ===== Уведомления (Telegram) =====
    TG_BOT_TOKEN = os.getenv('TG_BOT_TOKEN', '')
    TG_CHAT_ID = os.getenv('TG_CHAT_ID', '')

    # ===== Фильтры =====
    EXCLUDE_BELOW_EMA200 = os.getenv('EXCLUDE_BELOW_EMA200', 'false').lower() == 'false'
    CORRELATION_WINDOW = int(os.getenv('CORRELATION_WINDOW', '100'))
    MAX_MEAN_CORRELATION = float(os.getenv('MAX_MEAN_CORRELATION', '0.8'))

    # ADX фильтр трендовости
    ADX_FILTER_ENABLED = os.getenv('ADX_FILTER_ENABLED', 'true').lower() == 'true'
    ADX_MIN_THRESHOLD = float(os.getenv('ADX_MIN_THRESHOLD', '25.0'))  # Минимальное значение для сильного тренда
    ADX_PERIOD = int(os.getenv('ADX_PERIOD', '14'))  # Стандартный период расчета ADX

    # ===== ML (ARF) =====
# BOTISTABL-main/config.py (фрагмент: блок ML)
    # ===== ML (ARF) =====
    ARF_STATE_PATH   = os.getenv("ARF_STATE_PATH", "ml_state/arf_model.pkl")
    ARF_SAVE_EVERY   = int(os.getenv("ARF_SAVE_EVERY", "50"))

    # Выбор ансамбля: auto | arf | srp | bagging
    ARF_ENSEMBLE     = os.getenv("ARF_ENSEMBLE", "auto")
    ARF_N_MODELS     = int(os.getenv("ARF_N_MODELS", "15"))
    ARF_LAMBDA       = int(os.getenv("ARF_LAMBDA", "6"))
    ARF_SEED         = int(os.getenv("ARF_SEED", "42"))

    # Порог входа по ML
    ARF_ENTRY_PROBA = float(os.getenv("ARF_ENTRY_PROBA", "0.55"))

    # WARMUP: сколько финальных исходов (закрытых сделок) накопить, прежде чем
    # включать ML-гейтинг p>=ARF_ENTRY_PROBA. Пока меньше — входим по базовому правилу.
    ARF_WARMUP_LABELS = int(os.getenv("ARF_WARMUP_LABELS", "50"))

    # Калибровка вероятностей ARF (онлайн Platt scaling)
    ARF_CALIBRATION_ENABLED    = os.getenv("ARF_CALIBRATION_ENABLED", "true").lower() == "true"
    ARF_CALIBRATION_MIN_LABELS = int(os.getenv("ARF_CALIBRATION_MIN_LABELS", "50"))
    ARF_CALIBRATION_LR         = float(os.getenv("ARF_CALIBRATION_LR", "0.05"))
    ARF_CALIBRATION_CLIP       = float(os.getenv("ARF_CALIBRATION_CLIP", "10.0"))
    ARF_CALIBRATION_USE_LOGIT  = os.getenv("ARF_CALIBRATION_USE_LOGIT", "true").lower() == "true"

    
    # Путь для CSV-лога закрытых сделок
    TRADES_LOG_PATH = os.getenv('TRADES_LOG_PATH', os.path.join('state', 'trades.csv'))
    # Количество графиков с уровнями в сводке
# Количество графиков с уровнями в сводке
    SUMMARY_CHART_TOPN = int(os.getenv('SUMMARY_CHART_TOPN', '5'))
    # Сколько баров показывать на картинке
    SUMMARY_CHART_MAX_BARS = int(os.getenv('SUMMARY_CHART_MAX_BARS', '220'))
    
    # ===== SHORT торговля =====
    # Включить/выключить SHORT торговлю
    SHORT_TRADING_ENABLED = os.getenv('SHORT_TRADING_ENABLED', 'true').lower() == 'true'
    
    # RSI порог для SHORT (должен быть низким, например 45)
    RSI_THRESHOLD_SHORT = int(os.getenv('RSI_THRESHOLD_SHORT', '45'))
    
    # Путь к ARF модели для SHORT
    ARF_SHORT_STATE_PATH = os.getenv("ARF_SHORT_STATE_PATH", "ml_state/arf_model_short.pkl")
    
    # Порог входа по ML для SHORT
    ARF_ENTRY_PROBA_SHORT = float(os.getenv("ARF_ENTRY_PROBA_SHORT", "0.55"))
    
    # WARMUP для SHORT ARF
    ARF_WARMUP_LABELS_SHORT = int(os.getenv("ARF_WARMUP_LABELS_SHORT", "50"))
