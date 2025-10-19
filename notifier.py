import logging
import json
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class Notifier:
    def __init__(self, config, data_manager=None, resistance_analyzer=None):
        self.cfg = config
        self.enabled = bool(self.cfg.TG_BOT_TOKEN and self.cfg.TG_CHAT_ID)
        self._base = f"https://api.telegram.org/bot{self.cfg.TG_BOT_TOKEN}"
        self._timeout = 10
        # зависимости для рендера графиков/чек-листов
        self.data_manager = data_manager
        self.resistance_analyzer = resistance_analyzer

    def _send_tg(self, text: str):
        if not self.enabled:
            return
        try:
            resp = requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self.cfg.TG_CHAT_ID, "text": text},
                timeout=self._timeout
            )
            if not resp.ok:
                logger.warning(f"Telegram sendMessage error: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Не удалось отправить уведомление в Telegram: {e}")

    # новые удобные события
    def startup(self):
        self._send_tg("✅ Бот запущен")

    def shutdown(self):
        self._send_tg("🛑 Бот остановлен")

    def market_summary(self, summary: dict):
        lines = [
            "📊 СВОДКА ПО РЫНКУ",
            f"Проанализировано монет: {summary.get('total_analyzed')}",
            f"С уровнями: {summary.get('coins_with_levels')}",
            f"Активных пробоев: {summary.get('active_breakouts')}",
            f"Сила рынка: {summary.get('market_strength')}",
        ]
        top = summary.get('top_opportunities') or []
        if top:
            lines.append("🎯 ТОП ВОЗМОЖНОСТИ:")
            for i, opp in enumerate(top[:5], 1):
                p = opp.get('arf_proba')
                # на всякий случай поддержим старый формат, если он ещё где-то прокатится
                if p is None:
                    p = (opp.get('breakout_info') or {}).get('arf_proba')

                if p is not None:
                    lines.append(f"{i}. {opp['symbol']} ({opp['timeframe']}) - ARF p={float(p):.3f}, Action: {opp['action']}")
                else:
                    lines.append(f"{i}. {opp['symbol']} ({opp['timeframe']}) - Score: {opp['score']:.1f}, Action: {opp['action']}")
        self._send_tg("\n".join(lines))

    # существующие трейд-события
    def notify_trade_opened(self, pos):
        self._send_tg(f"📈 Открыта позиция {pos.symbol} @ {pos.entry_price:.6f}, qty={pos.qty}")

    def notify_partial_tp(self, pos, threshold: float, qty: float, price: float):
        self._send_tg(f"⚖️ Частичный TP {pos.symbol}: {qty} @ +{threshold*100:.1f}% (цена {price:.6f})")

    def notify_trailing(self, pos):
        self._send_tg(f"🔧 Трейлинг-стоп подтянут {pos.symbol}: SL={pos.stop_loss:.6f}")

    def _send_tg_photo(self, caption: str, photo_bytes: bytes, filename: str = "chart.png"):
        if not self.enabled:
            return
        try:
            resp = requests.post(
                f"{self._base}/sendPhoto",
                data={"chat_id": self.cfg.TG_CHAT_ID, "caption": caption},
                files={"photo": (filename, photo_bytes, "image/png")},
                timeout=self._timeout
            )
            resp.raise_for_status()
        except Exception:
            logger.exception("Ошибка отправки фото в Telegram")

    def _send_tg_media_group(self, images: list[bytes], caption_for_first: str = ""):
        """
        Отправка альбома: первая картинка содержит подпись (сводка).
        Реализация через sendMediaGroup + attach://file_name (оф. протокол).
        """
        if not self.enabled or not images:
            return
        try:
            media = []
            files = {}
            for i, img in enumerate(images):
                name = f"photo{i}.png"
                media.append({
                    "type": "photo",
                    "media": f"attach://{name}",
                    **({"caption": caption_for_first} if i == 0 and caption_for_first else {})
                })
                files[name] = (name, img, "image/png")

            resp = requests.post(
                f"{self._base}/sendMediaGroup",
                data={
                    "chat_id": self.cfg.TG_CHAT_ID,
                    "media": json.dumps(media, ensure_ascii=False)
                },
                files=files,
                timeout=self._timeout
            )
            resp.raise_for_status()
        except Exception:
            logger.exception("Ошибка отправки альбома в Telegram")

    def market_summary_with_charts(self, summary_text: str, chart_images: list[bytes]):
        """
        summary_text — готовая текстовая сводка (как у вас в main.py формируется сейчас)
        chart_images — список PNG-байтов (до 10 штук; в Telegram лимит альбома).
        Если список пуст — просто шлём текст.
        """
        if not chart_images:
            self._send_tg(summary_text)
        else:
            # первая картинка — с подписью; остальные — без
            self._send_tg_media_group(chart_images, caption_for_first=summary_text)

# СТАЛО (добавили уведомление о пробое с чек-листом и ARF)
    def notify_closed(self, pos, reason: str):
        self._send_tg(f"🔔 Закрыта позиция {pos.symbol}: {reason}")

    def notify_breakout_conditions(self, info: dict):
        if not self.enabled:
            return

        def _fmt_bool(ok: bool) -> str:
            return "✅" if ok else "❌"

        conds = info.get("conditions", {})
        cc = conds.get("close_above_level", {})
        vc = conds.get("volume_surge", {})
        rc = conds.get("rsi", {})
        kc = conds.get("confirmation_candles", {})

        lines: List[str] = [
            f"🚨 ПРОБОЙ: {info.get('symbol')} {info.get('timeframe','')}".strip(),
            f"Уровень: {info.get('level_price'):.6f}  →  Пробой @ {info.get('breakout_price'):.6f}",
            f"Сила уровня: {info.get('level_strength')}",
        ]

        p = info.get("arf_proba")
        if p is not None:
            try:
                lines.append(f"🤖 ARF p = {float(p):.3f}")
            except Exception:
                from error_logger import log_exception
                log_exception("Unhandled exception")

        lines += [
            "— Условия:",
            f"{_fmt_bool(cc.get('passed', False))} Закрытие > уровня+Δ   "
            f"(close={cc.get('close'):.6f}; Δ={cc.get('delta_pct'):.4f}; порог>{cc.get('threshold_pct'):.4f})",
            f"{_fmt_bool(vc.get('passed', False))} Объём > средн×mult    "
            f"({vc.get('current'):.2f}/{vc.get('average'):.2f}; mult×{vc.get('multiplier'):.2f})",
            f"{_fmt_bool(rc.get('passed', False))} RSI ≥ порога          "
            f"({rc.get('value'):.2f} ≥ {rc.get('threshold')})",
            f"{_fmt_bool(kc.get('passed', False))} Подтверждающих свечей "
            f"{kc.get('count_ok')}/{kc.get('required')}",
        ]

        # (опционально) картинка уровня при чек-листе — РЕАЛИЗОВАНО
        try:
            from charting import render_level_chart
            symbol = info.get("symbol")
            timeframe = info.get("timeframe", "15m")

            if symbol and self.data_manager is not None and self.resistance_analyzer is not None:
                # тянем историю
                min_days = getattr(self.cfg, "MIN_HISTORY_DAYS", 7)
                df = self.data_manager.fetch_klines_full(symbol, timeframe, min_days)
                if df is not None and not df.empty:
                    # вычисляем уровни
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

                    breakout_info = {
                        "breakout_price": (
                            info.get("breakout_price") or
                            info.get("breakout_price_at") or
                            info.get("price")
                        ),
                        "ts": info.get("ts") or info.get("time")
                    }
                    img_bytes = render_level_chart(
                        df=df,
                        levels=levels,
                        symbol=symbol,
                        timeframe=timeframe,
                        breakout=breakout_info,
                        max_bars=max_bars,
                    )

                    # короткая подпись (лимит подписи к фото у Telegram ~1024 символа)
                    caption = f"📌 {symbol} ({timeframe}) — чек-лист пробоя"
                    self._send_tg_photo(caption=caption, photo_bytes=img_bytes, filename=f"{symbol}_{timeframe}_breakout.png")
        except Exception:
            logger.exception("Не удалось подготовить/отправить график для чек-листа")

        self._send_tg("\n".join(lines))

    def notify_breakdown_conditions(self, info: dict):
        """Уведомление о пробое поддержки ВНИЗ (для SHORT)"""
        if not self.enabled:
            return

        def _fmt_bool(ok: bool) -> str:
            return "✅" if ok else "❌"

        conds = info.get("conditions", {})
        cc = conds.get("close_below_level", {})
        vc = conds.get("volume_surge", {})
        rc = conds.get("rsi", {})
        kc = conds.get("confirmation_candles", {})

        lines: List[str] = [
            f"🚨 ПРОБОЙ ВНИЗ (SHORT): {info.get('symbol')} {info.get('timeframe','')}".strip(),
            f"Уровень: {info.get('level_price'):.6f}  →  Пробой @ {info.get('breakdown_price'):.6f}",
            f"Сила уровня: {info.get('level_strength')}",
        ]

        p = info.get("arf_proba")
        if p is not None:
            try:
                lines.append(f"🤖 ARF SHORT p = {float(p):.3f}")
            except Exception:
                pass

        lines += [
            "— Условия:",
            f"{_fmt_bool(cc.get('passed', False))} Закрытие < уровня-Δ   "
            f"(close={cc.get('close'):.6f}; Δ={cc.get('delta_pct'):.4f}; порог>{cc.get('threshold_pct'):.4f})",
            f"{_fmt_bool(vc.get('passed', False))} Объём > средн×mult    "
            f"({vc.get('current'):.2f}/{vc.get('average'):.2f}; mult×{vc.get('multiplier'):.2f})",
            f"{_fmt_bool(rc.get('passed', False))} RSI ≤ порога          "
            f"({rc.get('value'):.2f} ≤ {rc.get('threshold')})",
            f"{_fmt_bool(kc.get('passed', False))} Подтверждающих свечей "
            f"{kc.get('count_ok')}/{kc.get('required')}",
        ]

        # картинка уровня
        try:
            from charting import render_level_chart
            symbol = info.get("symbol")
            timeframe = info.get("timeframe", "15m")

            if symbol and self.data_manager is not None and self.resistance_analyzer is not None:
                min_days = getattr(self.cfg, "MIN_HISTORY_DAYS", 7)
                df = self.data_manager.fetch_klines_full(symbol, timeframe, min_days)
                if df is not None and not df.empty:
                    # для SHORT используем поддержки, но resistance_analyzer может быть заменён на support_analyzer
                    # временно используем resistance_analyzer для уровней
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

                    breakdown_info = {
                        "breakout_price": (
                            info.get("breakdown_price") or
                            info.get("price")
                        ),
                        "ts": info.get("ts") or info.get("time"),
                        "direction": "down"
                    }
                    max_bars = int(getattr(self.cfg, "SUMMARY_CHART_MAX_BARS", 220))
                    img_bytes = render_level_chart(
                        df=df,
                        levels=levels,
                        symbol=symbol,
                        timeframe=timeframe,
                        breakout=breakdown_info,
                        max_bars=max_bars,
                    )

                    caption = f"📌 {symbol} ({timeframe}) — чек-лист пробоя ВНИЗ (SHORT)"
                    self._send_tg_photo(caption=caption, photo_bytes=img_bytes, filename=f"{symbol}_{timeframe}_breakdown.png")
        except Exception:
            logger.exception("Не удалось подготовить/отправить график для чек-листа SHORT")

        self._send_tg("\n".join(lines))

