# -*- coding: utf-8 -*-
from __future__ import annotations

import os
# фикс на уровне процесса: если что-то импортирует matplotlib до charting.py
os.environ.setdefault("MPLBACKEND", "Agg")

import hashlib
import logging
import signal
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
from charting import render_level_chart


import schedule
from dotenv import load_dotenv

# .env нужно загрузить ДО создания Config
load_dotenv()

from config import Config
from data_manager import DataManager
from resistance_analyzer import ResistanceAnalyzer
from breakout_detector import BreakoutDetector
from market_scanner import MarketScanner
from trading_engine import TradingEngine
from risk_manager import RiskManager
from notifier import Notifier
from websocket_streams import WebsocketStreams

# === ML (онлайн-ARF) ===
from ml_arf import ARFModel, extract_numeric_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="-%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BreakoutTradingBot:
    def __init__(self):
        self.config = Config()
        self.data_manager = DataManager(self.config.API_KEY, self.config.API_SECRET, self.config)
        
        # анализаторы уровней
        self.resistance_analyzer = ResistanceAnalyzer(self.config)
        self.support_analyzer = None  # создадим позже, если SHORT включён
        
        self.notifier = Notifier(
            self.config,
            data_manager=self.data_manager,
            resistance_analyzer=self.resistance_analyzer
        )
        
        # ML модели: LONG и SHORT
        from ml_arf import create_arf_long, create_arf_short
        self.arf_long = create_arf_long(self.config)
        self.arf_short = None
        
        # детекторы пробоя
        self.breakout_detector = BreakoutDetector(
            self.config,
            self.data_manager,
            notifier=self.notifier,
            arf_model=self.arf_long,
        )
        self.breakdown_detector = None
        
        # если SHORT включён
        if self.config.SHORT_TRADING_ENABLED:
            from support_analyzer import SupportAnalyzer
            from breakdown_detector import BreakdownDetector
            
            self.support_analyzer = SupportAnalyzer(self.config)
            self.arf_short = create_arf_short(self.config)
            self.breakdown_detector = BreakdownDetector(
                self.config,
                self.data_manager,
                notifier=self.notifier,
                arf_model_short=self.arf_short,
            )
            logger.info("✅ SHORT торговля включена")
        
        self.market_scanner = MarketScanner(
            self.config, self.data_manager, self.resistance_analyzer, self.breakout_detector
        )
        self.risk_manager = RiskManager(self.config, self.data_manager)
        self.engine = TradingEngine(self.config, self.data_manager, self.risk_manager, self.notifier)
        self.ws = WebsocketStreams(self.config, self.data_manager)
        self.engine.on_position_closed = self._on_position_closed

        self.is_running = False
        self._sched_thread: Optional[threading.Thread] = None
        self._last_top_hash: Optional[str] = None
        self._last_top_hash_short: Optional[str] = None

        logger.info("Инициализация завершена")

    # ====== Жизненный цикл ======

    def start(self):
        logger.info("Запуск бота...")
        self.is_running = True

        # уведомление о старте
        try:
            if hasattr(self.notifier, "startup"):
                self.notifier.startup()
            elif hasattr(self.notifier, "_send_tg"):
                self.notifier._send_tg("✅ Бот запущен")
        except Exception:
            logger.exception("Ошибка при отправке уведомления о старте")

        # вебсокеты
        try:
            self.ws.start()
        except Exception:
            logger.exception("Ошибка запуска вебсокетов")

        # первичный прогон
        self.scan_and_analyze()

        # планировщик задач
        schedule.every(self.config.SCAN_INTERVAL).seconds.do(self.scan_and_analyze)
        schedule.every(self.config.LEVEL_UPDATE_INTERVAL).seconds.do(self.update_resistance_levels)
        schedule.every(self.config.MANAGE_POSITIONS_INTERVAL).seconds.do(self.engine.manage_positions)

        # НЕ daemon — держим процесс живым
        self._sched_thread = threading.Thread(target=self._run_schedule, name="SchedulerLoop", daemon=False)
        self._sched_thread.start()

        logger.info("Бот успешно запущен")

        # удерживаем главный поток
        try:
            while self.is_running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки (Ctrl+C)")
        finally:
            self.stop()
            if self._sched_thread and self._sched_thread.is_alive():
                self._sched_thread.join(timeout=5)

    def stop(self):
        logger.info("Остановка бота...")
        self.is_running = False
        try:
            schedule.clear()
        except Exception:
            pass
        try:
            self.ws.stop()
        except Exception:
            logger.exception("Ошибка остановки вебсокетов")

        # сохранить обе ML-модели
        try:
            if self.arf_long:
                self.arf_long.save_now()
                logger.info("✅ ARF LONG модель сохранена")
        except Exception:
            logger.exception("Не удалось сохранить ARF LONG модель при остановке")
        
        try:
            if self.arf_short:
                self.arf_short.save_now()
                logger.info("✅ ARF SHORT модель сохранена")
        except Exception:
            logger.exception("Не удалось сохранить ARF SHORT модель при остановке")

        # уведомление об остановке
        try:
            if hasattr(self.notifier, "shutdown"):
                self.notifier.shutdown()
            elif hasattr(self.notifier, "_send_tg"):
                self.notifier._send_tg("🛑 Бот остановлен")
        except Exception:
            logger.exception("Ошибка при отправке уведомления об остановке")

        logger.info("Бот остановлен")

    # ====== Планировщик ======

    def _run_schedule(self):
        while self.is_running:
            try:
                schedule.run_pending()
            except Exception:
                logger.exception("Ошибка в планировщике")
            time.sleep(1)

    # ====== Основная логика ======

    def scan_and_analyze(self):
        """Сканирование для LONG позиций"""
        try:
            logger.info("=" * 50)
            logger.info("Начинаю сканирование рынка (LONG)...")
            opportunities = self.market_scanner.scan_market()

            if opportunities:
                logger.info(f"Найдено {len(opportunities)} потенциальных LONG возможностей")
                for opp in opportunities[:5]:
                    self._process_opportunity(opp, direction="LONG")
            else:
                logger.info("Подходящих LONG возможностей не найдено")

            summary = self.market_scanner.get_market_summary()
            self._log_market_summary(summary, direction="LONG")
            self._maybe_notify_summary(summary, direction="LONG")

        except Exception:
            logger.exception("Ошибка при сканировании LONG")
        
        # сканирование SHORT
        if self.config.SHORT_TRADING_ENABLED and self.breakdown_detector:
            self.scan_and_analyze_short()


    def scan_and_analyze_short(self):
        """Сканирование для SHORT позиций"""
        try:
            logger.info("=" * 50)
            logger.info("Начинаю сканирование рынка (SHORT)...")
            
            top_coins = self.data_manager.get_top_coins(self.config.TOP_COINS_COUNT, self.config.MIN_VOLUME_24H)
            if not top_coins:
                logger.error("Не удалось получить список монет для SHORT")
                return

            logger.info(f"Анализирую {len(top_coins)} монет для SHORT...")
            opportunities_short: List[Dict] = []
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_symbol_short, symbol): symbol
                    for symbol in top_coins
                }
                for future in as_completed(future_to_symbol):
                    res = future.result()
                    if res:
                        opportunities_short.append(res)

            opportunities_short.sort(key=lambda x: x['score'], reverse=True)
            
            if opportunities_short:
                logger.info(f"Найдено {len(opportunities_short)} потенциальных SHORT возможностей")
                for opp in opportunities_short[:5]:
                    self._process_opportunity(opp, direction="SHORT")
            else:
                logger.info("Подходящих SHORT возможностей не найдено")

            # сводка для SHORT
            summary_short = self._get_short_summary(opportunities_short)
            self._log_market_summary(summary_short, direction="SHORT")
            self._maybe_notify_summary(summary_short, direction="SHORT")

        except Exception:
            logger.exception("Ошибка при сканировании SHORT")


    def _analyze_symbol_short(self, symbol: str) -> Optional[Dict]:
        """Анализ одного символа для SHORT"""
        try:
            analysis_result: Dict = {'symbol': symbol, 'timestamp': datetime.now(), 'timeframes': {}}
            best_opportunity = None
            max_score = 0

            for timeframe in self.config.TIMEFRAMES:
                df = self.data_manager.fetch_klines_full(symbol, timeframe, self.config.MIN_HISTORY_DAYS)
                if df.empty:
                    continue

                # фильтр восходящего тренда (выше EMA200 - кандидат на разворот вниз)
                if self.config.EXCLUDE_BELOW_EMA200 and len(df) >= 210:
                    ema200 = self.data_manager.calculate_ema(df['close'], 200)
                    if df['close'].iloc[-1] > ema200.iloc[-1]:
                        # для SHORT нам нужна цена ВЫШЕ EMA200, чтобы было куда падать
                        pass
                    else:
                        continue

                levels = self.support_analyzer.find_support_levels(df, symbol)
                combined = levels.get('combined', [])
                if not combined:
                    continue

                breakdown_info = self.breakdown_detector.check_breakdown(symbol, combined, df)

                score = self._score_symbol_short(df, combined, breakdown_info)
                analysis_result['timeframes'][timeframe] = {
                    'levels_count': len(combined),
                    'has_breakdown': bool(breakdown_info),
                    'score': score,
                    'breakdown_info': breakdown_info
                }

                if score > max_score:
                    max_score = score
                    best_opportunity = {
                        'symbol': symbol,
                        'score': score,
                        'timeframe': timeframe,
                        'breakdown_info': breakdown_info,
                        'levels': combined,
                        'direction': 'SHORT'
                    }

            return best_opportunity
        except Exception as e:
            logger.error(f"Ошибка анализа SHORT {symbol}: {e}")
            return None


    def _score_symbol_short(self, df: pd.DataFrame, levels: List[Dict], breakdown_info: Optional[Dict]) -> float:
        """Scoring для SHORT возможностей"""
        score = 0.0
        current_price = df['close'].iloc[-1]

        if breakdown_info:
            score += 50
            score += float(breakdown_info.get('confidence_score', 0.0)) * 20
            return score

        if levels:
            nearest = levels[0]
            distance_percent = abs(nearest['distance_percent'])
            if distance_percent < 1: score += 30
            elif distance_percent < 2: score += 20
            elif distance_percent < 3: score += 10
            score += float(nearest['strength']) * 15

        rsi = self.data_manager.calculate_rsi(df['close']).iloc[-1]
        # для SHORT: низкий RSI благоприятен
        if 30 < rsi < 50:
            score += 10
        elif rsi <= 30:
            score += 5

        return score


    def _get_short_summary(self, opportunities: List[Dict]) -> Dict:
        """Формирование сводки для SHORT"""
        total = len(opportunities)
        with_breakdowns = sum(1 for o in opportunities if o.get('breakdown_info'))
        
        top_opps = opportunities[:5] if opportunities else []
        
        return {
            'timestamp': datetime.now(),
            'total_analyzed': total,
            'coins_with_levels': total,
            'active_breakouts': with_breakdowns,
            'top_opportunities': [
                {
                    'symbol': o['symbol'],
                    'timeframe': o.get('timeframe', ''),
                    'score': o['score'],
                    'action': 'SHORT' if o.get('breakdown_info') else 'WATCH',
                    'arf_proba': (o.get('breakdown_info') or {}).get('arf_proba')
                } for o in top_opps
            ],
            'market_strength': 'SHORT_MARKET'
        }

    def _build_ml_features(self, opportunity: Dict) -> Dict[str, float]:
        """
        Забираем ВСЕ числовые фичи из тех же структур, на которые опирается score:
        - сам score
        - breakout_info (все числовые поля, рекурсивно)
        - levels[0] (все числовые поля, рекурсивно)
        """
        feats: Dict[str, float] = {}

        # score как признак
        try:
            feats["score"] = float(opportunity.get("score", 0.0))
        except Exception:
            feats["score"] = 0.0

        # breakout_info
        bi = opportunity.get("breakout_info") or {}
        feats.update(extract_numeric_features(bi, "brk_"))

        # верхний уровень из levels
        levels = opportunity.get("levels") or []
        if isinstance(levels, list) and levels:
            top = levels[0]
            if isinstance(top, dict):
                feats.update(extract_numeric_features(top, "lvl_"))

        # чистим NaN/inf
        for k, v in list(feats.items()):
            if v != v or v in (float("inf"), float("-inf")):
                feats[k] = 0.0

        return feats

    def _process_opportunity(self, opportunity: Dict, direction: str = "LONG"):
        """Обработка возможности для LONG или SHORT"""
        symbol = opportunity["symbol"]
        tf = opportunity.get("timeframe")
        score = float(opportunity.get("score", 0.0))

        # выбираем правильную ARF модель
        if direction == "SHORT":
            arf_model = self.arf_short
            info_key = "breakdown_info"
            thr = float(getattr(self.config, "ARF_ENTRY_PROBA_SHORT", 0.55))
            warm_min = int(getattr(self.config, "ARF_WARMUP_LABELS_SHORT", 50))
        else:
            arf_model = self.arf_long
            info_key = "breakout_info"
            thr = float(getattr(self.config, "ARF_ENTRY_PROBA", 0.62))
            warm_min = int(getattr(self.config, "ARF_WARMUP_LABELS", 50))

        feats = self._build_ml_features(opportunity)
        p_ml = arf_model.predict_proba(feats) if arf_model else 0.5
        
        logger.info(f"\n🎯 Возможность ({direction}): {symbol} ({tf}) | score={score:.1f} | ARF p_success={p_ml:.3f}")

        signal_info = opportunity.get(info_key)
        use_ml_gate = arf_model.is_warm(warm_min) if arf_model else False

        if signal_info and (not use_ml_gate or p_ml >= thr):
            ok, reason = self._validate_and_open(signal_info, ml_features=feats, ml_pred=p_ml)
            if ok:
                logger.info(f"  ✅ Пробой валиден, {direction} позиция открыта")
                try:
                    self.ws.subscribe_symbol(symbol)
                except Exception:
                    logger.exception(f"WebSocket подписка на {symbol} не удалась")
            else:
                logger.warning(f"  ❌ Пробой отклонен: {reason}")
        else:
            if signal_info and use_ml_gate:
                logger.info(f"  ⏸ Пропущено ARF-гейтом ({direction}) (p={p_ml:.3f} < {thr:.3f})")
            else:
                levels = opportunity.get("levels") or []
                if levels and isinstance(levels[0], dict):
                    level = levels[0]
                    dist = float(level.get("distance_percent", 0.0))
                    strength = float(level.get("strength", 0.0))
                    try:
                        lp = float(level["price"])
                    except Exception:
                        lp = 0.0
                    logger.info(
                        f"  Ближайший уровень: {lp:.4f} ({dist:.2f}% от цены) | сила={strength:.2f}"
                    )

        if arf_model:
            logger.info(
                f"  ARF gate: {'ON' if use_ml_gate else 'WARMUP-OFF'} | "
                f"labels={arf_model.labels_seen()} | thr={thr:.3f} | p={p_ml:.3f}"
            )


    def _validate_and_open(
        self,
        breakout: Dict,
        ml_features: Optional[Dict[str, float]] = None,
        ml_pred: Optional[float] = None,
    ):
        symbol = breakout.get("symbol")

        # портфельные ограничения / корреляция
        try:
            active_symbols = [p.symbol for p in self.engine.positions if getattr(p, "status", "") == "ACTIVE"]
        except Exception:
            active_symbols = []

        try:
            if not self.risk_manager.correlation_check(symbol, active_symbols):
                return False, "Высокая корреляция с текущим портфелем"
        except Exception:
            logger.exception("Ошибка проверки корреляции")
            return False, "Ошибка проверки корреляции"

        # вход
        try:
            pos = self.engine.execute_entry(breakout, ml_features=ml_features, ml_pred=ml_pred)
        except Exception:
            logger.exception("Ошибка открытия позиции")
            return False, "Ошибка при открытии позиции"

        if pos is None:
            return False, "Ограничение риска или ошибка при открытии"
        return True, "OK"

    def update_resistance_levels(self):
        logger.info("Обновление уровней сопротивления...")
        try:
            symbols = {p.symbol for p in self.engine.positions if getattr(p, "status", "") == "ACTIVE"}
        except Exception:
            symbols = set()

        for symbol in symbols:
            for timeframe in self.config.TIMEFRAMES:
                try:
                    df = self.data_manager.fetch_klines_full(symbol, timeframe, self.config.MIN_HISTORY_DAYS)
                    if df.empty:
                        continue
                    levels = self.resistance_analyzer.find_resistance_levels(df, symbol)
                    self.resistance_analyzer.levels_cache[f"{symbol}_{timeframe}"] = levels
                except Exception:
                    logger.exception(f"Ошибка обновления уровней для {symbol} {timeframe}")

    # ====== Логи/уведомления ======

    def _log_market_summary(self, summary: Dict, direction: str = "LONG"):
        prefix = "📊" if direction == "LONG" else "📉"
        logger.info(f"\n{prefix} СВОДКА ПО РЫНКУ ({direction}):")
        logger.info(f"Проанализировано монет: {summary.get('total_analyzed')}")
        logger.info(f"С уровнями: {summary.get('coins_with_levels')}")
        logger.info(f"Активных пробоев: {summary.get('active_breakouts')}")
        logger.info(f"Сила рынка: {summary.get('market_strength')}")
        top = summary.get("top_opportunities") or []
        if top:
            logger.info(f"\n🎯 ТОП ВОЗМОЖНОСТИ ({direction}):")
            lines: List[str] = []
            for i, opp in enumerate(top[:5], 1):
                bi = opp.get('breakdown_info' if direction == 'SHORT' else 'breakout_info') or {}
                p = bi.get('arf_proba')
                if p is not None:
                    lines.append(
                        f"{i}. {opp['symbol']} ({opp.get('timeframe','')}) - "
                        f"ARF p={float(p):.3f}, Action: {opp.get('action','')}"
                    )
                else:
                    lines.append(
                        f"{i}. {opp['symbol']} ({opp.get('timeframe','')}) - "
                        f"Score: {opp.get('score', float('nan')):.1f}, Action: {opp.get('action','')}"
                    )
            if lines:
                logger.info("\n".join(lines))


    def _maybe_notify_summary(self, summary: Dict, direction: str = "LONG"):
        """Анти-спам: отправляем сводку в TG только если топ изменился"""
        try:
            payload = summary.get("top_opportunities") or []

            # выбираем правильный хэш атрибут
            hash_attr = "_last_top_hash_short" if direction == "SHORT" else "_last_top_hash"
            
            key_items = []
            for o in payload[:10]:
                p = o.get('arf_proba')
                metric = f"p={float(p):.2f}" if p is not None else f"s={float(o.get('score', 0)):.1f}"
                key_items.append(f"{o.get('symbol')}|{o.get('timeframe')}|{o.get('action')}|{metric}")
            key_str = ";".join(key_items)
            curr_hash = hashlib.sha1(key_str.encode("utf-8")).hexdigest()

            if curr_hash == getattr(self, hash_attr, None):
                return

            setattr(self, hash_attr, curr_hash)

            # текст сводки
            prefix = "📊" if direction == "LONG" else "📉"
            lines: List[str] = [
                f"{prefix} СВОДКА ПО РЫНКУ ({direction})",
                f"Проанализировано монет: {summary.get('total_analyzed')}",
                f"С уровнями: {summary.get('coins_with_levels')}",
                f"Активных пробоев: {summary.get('active_breakouts')}",
                f"Сила рынка: {summary.get('market_strength')}",
                f"🎯 ТОП ВОЗМОЖНОСТИ ({direction}):",
            ]
            for i, opp in enumerate(payload[:5], 1):
                lines.append(
                    f"{i}. {opp['symbol']} ({opp.get('timeframe','15m')}) - "
                    f"Score: {float(opp.get('score',0.0)):.1f}, Action: {opp.get('action','WATCH')}"
                )
            summary_text = "\n".join(lines)

            # картинки (используем правильный анализатор)
            chart_images: List[bytes] = []
            try:
                from charting import render_level_chart
                top_n = int(getattr(self.config, "SUMMARY_CHART_TOPN", 5))
                max_bars = int(getattr(self.config, "SUMMARY_CHART_MAX_BARS", 220))
                
                analyzer = self.support_analyzer if direction == "SHORT" else self.resistance_analyzer

                for opp in payload[:top_n]:
                    symbol = opp.get('symbol')
                    timeframe = opp.get('timeframe', '15m')
                    if not symbol:
                        continue

                    df = self.data_manager.fetch_klines_full(symbol, timeframe, self.config.MIN_HISTORY_DAYS)
                    if df is None or df.empty:
                        continue

                    if direction == "SHORT":
                        levels_info = analyzer.find_support_levels(df, symbol)
                    else:
                        levels_info = analyzer.find_resistance_levels(df, symbol)
                        
                    if isinstance(levels_info, dict):
                        levels = (
                            levels_info.get('combined')
                            or levels_info.get('historical_peaks')
                            or levels_info.get('historical_lows')
                            or levels_info.get('horizontal_zones')
                            or []
                        )
                    else:
                        levels = []

                    signal_info = opp.get('breakdown_info' if direction == 'SHORT' else 'breakout_info')

                    img = render_level_chart(
                        df=df,
                        levels=levels,
                        symbol=symbol,
                        timeframe=timeframe,
                        breakout=signal_info,
                        max_bars=max_bars,
                    )
                    chart_images.append(img)
            except Exception:
                logger.exception(f"Ошибка подготовки графиков для сводки ({direction})")

            # отправка
            if hasattr(self.notifier, "market_summary_with_charts") and chart_images:
                self.notifier.market_summary_with_charts(summary_text, chart_images)
            elif hasattr(self.notifier, "market_summary"):
                self.notifier.market_summary(summary)
            elif hasattr(self.notifier, "_send_tg"):
                self.notifier._send_tg(summary_text)

        except Exception:
            logger.exception(f"Ошибка при отправке сводки в Telegram ({direction})")


    # ====== ML online-learning callback ======

    def _on_position_closed(self, pos, success: bool):
        """
        Вызывается TradingEngine на финальном закрытии.
        Выбирает правильную ARF модель по направлению позиции.
        """
        try:
            direction = getattr(pos, "direction", "LONG").upper()
            arf_model = self.arf_short if direction == "SHORT" else self.arf_long
            
            if not arf_model:
                return
            
            x = getattr(pos, "ml_features", None) or {}
            if x:
                y = 1 if success else 0
                arf_model.learn(x, y)
                logger.info(f"[ARF {direction}] online-learn: y={y} | features={len(x)}")
        except Exception:
            logger.exception(f"Ошибка онлайн-обучения ARF ({direction})")


def _setup_signals(bot: BreakoutTradingBot):
    def handler(signum, frame):
        logger.info(f"Получен сигнал {signum}, останавливаемся...")
        bot.stop()

    try:
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
    except Exception:
        # Windows может не поддерживать SIGTERM
        pass


def main():
    print(
        """    ╔══════════════════════════════════════════╗
    ║     BREAKOUT TRADING BOT v2.0           ║
    ║     Binance Resistance Scanner           ║
    ╚══════════════════════════════════════════╝
    """
    )
    bot = BreakoutTradingBot()
    _setup_signals(bot)
    try:
        bot.start()
    except Exception:
        logger.exception("Критическая ошибка в главном цикле")
        try:
            bot.stop()
        except Exception:
            from error_logger import log_exception
            log_exception("Error in main")


if __name__ == "__main__":
    main()
