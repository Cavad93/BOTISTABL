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

        # Core services
        self.data_manager = DataManager(self.config.API_KEY, self.config.API_SECRET, self.config)

        # сначала анализатор уровней — на него ссылается Notifier
        self.resistance_analyzer = ResistanceAnalyzer(self.config)

        # теперь Notifier может безопасно принять зависимости
        self.notifier = Notifier(
            self.config,
            data_manager=self.data_manager,
            resistance_analyzer=self.resistance_analyzer
        )

        # ML: онлайн-лес, переживающий рестарт
        self.arf = ARFModel(self.config)

        self.breakout_detector = BreakoutDetector(
            self.config,
            self.data_manager,
            notifier=self.notifier,
            arf_model=self.arf,
        )
        self.market_scanner = MarketScanner(
            self.config, self.data_manager, self.resistance_analyzer, self.breakout_detector
        )
        self.risk_manager = RiskManager(self.config, self.data_manager)
        self.engine = TradingEngine(self.config, self.data_manager, self.risk_manager, self.notifier)
        self.ws = WebsocketStreams(self.config, self.data_manager)
        self.engine.on_position_closed = self._on_position_closed  # callback(success: bool)

        self.is_running = False
        self._sched_thread: Optional[threading.Thread] = None
        self._last_top_hash: Optional[str] = None  # анти-спам в Telegram

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
            from error_logger import log_exception
            log_exception("Error in stop")
        try:
            self.ws.stop()
        except Exception:
            logger.exception("Ошибка остановки вебсокетов")

        # сохранить ML-модель
        try:
            self.arf.save_now()
        except Exception:
            logger.exception("Не удалось сохранить ARF-модель при остановке")

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
        try:
            logger.info("=" * 50)
            logger.info("Начинаю сканирование рынка...")
            opportunities = self.market_scanner.scan_market()

            if opportunities:
                logger.info(f"Найдено {len(opportunities)} потенциальных возможностей")
                for opp in opportunities[:5]:
                    self._process_opportunity(opp)
            else:
                logger.info("Подходящих возможностей не найдено")

            summary = self.market_scanner.get_market_summary()
            self._log_market_summary(summary)
            self._maybe_notify_summary(summary)

        except Exception:
            logger.exception("Ошибка при сканировании")

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

    def _process_opportunity(self, opportunity: Dict):
        symbol = opportunity["symbol"]
        tf = opportunity.get("timeframe")
        score = float(opportunity.get("score", 0.0))

        # ML-гейтинг с прогревом
        feats = self._build_ml_features(opportunity)
        p_ml = self.arf.predict_proba(feats)
        logger.info(f"\n🎯 Возможность: {symbol} ({tf}) | score={score:.1f} | ARF p_success={p_ml:.3f}")

        breakout = opportunity.get("breakout_info")
        thr = float(getattr(self.config, "ARF_ENTRY_PROBA", 0.62))

        warm_min = int(getattr(self.config, "ARF_WARMUP_LABELS", 50))
        use_ml_gate = self.arf.is_warm(warm_min)

        if breakout and (not use_ml_gate or p_ml >= thr):
            ok, reason = self._validate_and_open(breakout, ml_features=feats, ml_pred=p_ml)
            if ok:
                logger.info("  ✅ Пробой валиден, позиция открыта")
                try:
                    self.ws.subscribe_symbol(symbol)
                except Exception:
                    logger.exception(f"WebSocket подписка на {symbol} не удалась")
            else:
                logger.warning(f"  ❌ Пробой отклонен: {reason}")
        else:
            if breakout and use_ml_gate:
                logger.info(f"  ⏸ Пропущено ARF-гейтом (p={p_ml:.3f} < {thr:.3f})")
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

        logger.info(
            f"  ARF gate: {'ON' if use_ml_gate else 'WARMUP-OFF'} | "
            f"labels={self.arf.labels_seen()} | thr={thr:.3f} | p={p_ml:.3f}"
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

    def _log_market_summary(self, summary: Dict):
            logger.info("\n📊 СВОДКА ПО РЫНКУ:")
            logger.info(f"Проанализировано монет: {summary.get('total_analyzed')}")
            logger.info(f"С уровнями: {summary.get('coins_with_levels')}")
            logger.info(f"Активных пробоев: {summary.get('active_breakouts')}")
            logger.info(f"Сила рынка: {summary.get('market_strength')}")
            top = summary.get("top_opportunities") or []
            if top:
                logger.info("\n🎯 ТОП ВОЗМОЖНОСТИ:")
                lines: List[str] = []
                for i, opp in enumerate(top[:5], 1):
                    bi = opp.get('breakout_info') or {}
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


    def _maybe_notify_summary(self, summary: Dict):
        """Анти-спам: отправляем сводку в TG только если топ изменился; + альбом графиков уровней."""
        try:
            payload = summary.get("top_opportunities") or []

            # ключ стабильности (анти-спам) по топ-10
            key_items = []
            for o in payload[:10]:
                p = o.get('arf_proba')
                metric = f"p={float(p):.2f}" if p is not None else f"s={float(o.get('score', 0)):.1f}"
                key_items.append(f"{o.get('symbol')}|{o.get('timeframe')}|{o.get('action')}|{metric}")
            key_str = ";".join(key_items)
            curr_hash = hashlib.sha1(key_str.encode("utf-8")).hexdigest()

            if curr_hash == getattr(self, "_last_top_hash", None):
                return

            self._last_top_hash = curr_hash

            # текст сводки
            lines: List[str] = [
                "📊 СВОДКА ПО РЫНКУ",
                f"Проанализировано монет: {summary.get('total_analyzed')}",
                f"С уровнями: {summary.get('coins_with_levels')}",
                f"Активных пробоев: {summary.get('active_breakouts')}",
                f"Сила рынка: {summary.get('market_strength')}",
                "🎯 ТОП ВОЗМОЖНОСТИ:",
            ]
            for i, opp in enumerate(payload[:5], 1):
                lines.append(
                    f"{i}. {opp['symbol']} ({opp.get('timeframe','15m')}) - "
                    f"Score: {float(opp.get('score',0.0)):.1f}, Action: {opp.get('action','WATCH')}"
                )
            summary_text = "\n".join(lines)

            # картинки для ТОП-N возможностей (альбом)
            chart_images: List[bytes] = []
            try:
                from charting import render_level_chart
                top_n = int(getattr(self.config, "SUMMARY_CHART_TOPN", 5))
                max_bars = int(getattr(self.config, "SUMMARY_CHART_MAX_BARS", 220))

                for opp in payload[:top_n]:
                    symbol = opp.get('symbol')
                    timeframe = opp.get('timeframe', '15m')
                    if not symbol:
                        continue

                    df = self.data_manager.fetch_klines_full(symbol, timeframe, self.config.MIN_HISTORY_DAYS)
                    if df is None or df.empty:
                        continue

                    levels_info = self.resistance_analyzer.find_resistance_levels(df, symbol)
                    if isinstance(levels_info, dict):
                        levels = (
                            levels_info.get('combined')
                            or levels_info.get('historical_peaks')
                            or levels_info.get('horizontal_zones')
                            or []
                        )
                    else:
                        levels = []

                    breakout_info = opp.get('breakout_info')

                    img = render_level_chart(
                        df=df,
                        levels=levels,
                        symbol=symbol,
                        timeframe=timeframe,
                        breakout=breakout_info,
                        max_bars=max_bars,
                    )
                    chart_images.append(img)
            except Exception:
                logger.exception("Ошибка подготовки графиков для сводки")

            # отправка: если есть графики — альбом, иначе как раньше
            if hasattr(self.notifier, "market_summary_with_charts") and chart_images:
                self.notifier.market_summary_with_charts(summary_text, chart_images)
            elif hasattr(self.notifier, "market_summary"):
                self.notifier.market_summary(summary)
            elif hasattr(self.notifier, "_send_tg"):
                self.notifier._send_tg(summary_text)

        except Exception:
            logger.exception("Ошибка при отправке сводки в Telegram")


    # ====== ML online-learning callback ======

    def _on_position_closed(self, pos, success: bool):
        """
        Вызывается TradingEngine на финальном закрытии.
        success=True для TP, False для SL. Учим ARF онлайн.
        """
        try:
            x = getattr(pos, "ml_features", None) or {}
            if x:
                y = 1 if success else 0
                self.arf.learn(x, y)
                logger.info(f"[ARF] online-learn: y={y} | features={len(x)}")
        except Exception:
            logger.exception("Ошибка онлайн-обучения ARF")


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
