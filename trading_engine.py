from __future__ import annotations
import csv
import json
import logging
import os
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    partial_tps: List[float]
    partial_sizes: List[float]
    filled_partials: List[bool]
    status: str = "ACTIVE"
    ml_features: Optional[Dict[str, float]] = None  # признаки для ML (переживают рестарт)
    ml_pred: Optional[float] = None                 # предсказание ML на входе (переживает рестарт)


class TradingEngine:
    """
    Держит список позиций, управляет частичными TP, трейлингом и финальными выходами.
    Персистит состояние позиций в JSON, чтобы переживать рестарты (PAPER/LIVE).
    Вызывает on_position_closed(pos, success: bool), где success=True для TP и False для SL —
    это используется для онлайн-обучения ARF с меткой успеха/неуспеха.
    """

    def __init__(self, config, data_manager, risk_manager, notifier=None):
        self.cfg = config
        self.dm = data_manager
        self.rm = risk_manager
        self.notifier = notifier

        self.positions: List[Position] = []
        self.on_position_closed = None  # колбэк: Callable[[Position, bool], None]

        self._load_positions_state()

    # ===== Параметры риска/размера позиции =====

# СТАЛО: trading_engine.py (метод внутри TradingEngine)

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """
        Размер позиции по риску: RISK_PER_TRADE_PERCENT от депозита при SL=STOP_LOSS_PERCENT.
        qty = (deposit * RISK_PER_TRADE_PERCENT) / (price * STOP_LOSS_PERCENT)
        """
        try:
            deposit = float(self.dm.get_account_balance("USDT")) or float(self.cfg.DEFAULT_DEPOSIT_USDT)
        except Exception:
            deposit = float(getattr(self.cfg, "DEFAULT_DEPOSIT_USDT", 0.0))

        risk_frac = float(getattr(self.cfg, "RISK_PER_TRADE_PERCENT", 0.0035))
        sl_frac = float(getattr(self.cfg, "STOP_LOSS_PERCENT", 0.02))
        if price <= 0 or sl_frac <= 0 or risk_frac <= 0 or deposit <= 0:
            return 0.0

        risk_usdt = deposit * risk_frac
        qty = risk_usdt / (price * sl_frac)
        return round(qty, 6)


    def can_open_new_position(self) -> bool:
        max_positions = int(getattr(self.cfg, "MAX_CONCURRENT_POSITIONS", 3))
        active = sum(1 for p in self.positions if p.status == "ACTIVE")
        return active < max_positions

    def portfolio_risk_check(self) -> bool:
        """
        Формальный контроль совокупной аллокации.
        """
        try:
            deposit = float(self.dm.get_account_balance("USDT")) or float(self.cfg.DEFAULT_DEPOSIT_USDT)
        except Exception:
            deposit = float(getattr(self.cfg, "DEFAULT_DEPOSIT_USDT", 0.0))

        if deposit <= 0:
            return True

        total_alloc = sum(p.qty * p.entry_price for p in self.positions if p.status == "ACTIVE")
        max_frac = float(getattr(self.cfg, "PORTFOLIO_MAX_NOTIONAL", 0.8))
        return (total_alloc / deposit) <= max_frac

    # ===== Вход в позицию =====

    def execute_entry(
        self,
        breakout_info: Dict,
        ml_features: Optional[Dict[str, float]] = None,
        ml_pred: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Открывает позицию, сохраняет её в state, возвращает Position или None при ошибке.
        """
        if not self.can_open_new_position() or not self.portfolio_risk_check():
            logger.warning("Нельзя открыть новую позицию: ограничение риска/кол-ва позиций")
            return None

        symbol = breakout_info["symbol"]
        price = float(self.dm.get_current_price(symbol))
        qty = self.calculate_position_size(symbol, price)
        if qty <= 0:
            logger.warning("Размер позиции получился 0 — вход отменён")
            return None

        paper = str(getattr(self.cfg, "TRADING_MODE", "PAPER")).upper() != "LIVE"
        order = self.dm.place_market_order(symbol, "BUY", qty, paper=paper)
        if not order:
            logger.warning("Не удалось разместить ордер на вход")
            return None

        pos = Position(
            symbol=symbol,
            qty=qty,
            entry_price=price,
            entry_time=datetime.now(),
            stop_loss=float(breakout_info["stop_loss"]),
            take_profit=float(breakout_info["take_profit"]),
            partial_tps=list(getattr(self.cfg, "PARTIAL_TPS", [])),
            partial_sizes=list(getattr(self.cfg, "PARTIAL_TP_SIZES", [])),
            filled_partials=[False] * len(getattr(self.cfg, "PARTIAL_TP_SIZES", [])),
            ml_features=ml_features,
            ml_pred=ml_pred,
        )
        self.positions.append(pos)
        self._save_positions_state()

        logger.info(f"📈 Открыта позиция: {symbol} qty={qty} @ {price:.6f}")
        if self.notifier:
            try:
                self.notifier.notify_trade_opened(pos)
            except Exception:
                logger.exception("notify_trade_opened failed")

        return pos

    # ===== Управление открытыми позициями =====

    def manage_positions(self):
        """
        Вызывается периодически: частичные TP, трейлинг, SL/TP закрытия.
        На финальном закрытии вызывает on_position_closed(pos, success: bool).
        """
        paper = str(getattr(self.cfg, "TRADING_MODE", "PAPER")).upper() != "LIVE"

        for pos in list(self.positions):
            if pos.status != "ACTIVE":
                continue

            current = float(self.dm.get_current_price(pos.symbol))

            # Частичные TP
            for i, (thr, share) in enumerate(zip(pos.partial_tps, pos.partial_sizes)):
                if not pos.filled_partials[i] and current >= pos.entry_price * (1.0 + float(thr)):
                    sell_qty = round(pos.qty * float(share), 6)
                    if sell_qty > 0:
                        self.dm.place_market_order(pos.symbol, "SELL", sell_qty, paper=paper)
                        pos.qty = max(0.0, round(pos.qty - sell_qty, 6))
                        pos.filled_partials[i] = True
                        self._save_positions_state()
                        logger.info(f"⚖️ Частичная фиксация {pos.symbol}: {share*100:.0f}% @ +{thr*100:.1f}% (rem={pos.qty:.6f})")
                        if self.notifier:
                            try:
                                self.notifier.notify_partial_tp(pos, float(thr), sell_qty, current)
                            except Exception:
                                logger.exception("notify_partial_tp failed")

            # Трейлинг-стоп
            trailing_start = float(getattr(self.cfg, "TRAILING_START", 0.0))
            trailing_step = float(getattr(self.cfg, "TRAILING_STEP", 0.0))
            if trailing_start > 0 and trailing_step > 0:
                gain = (current - pos.entry_price) / pos.entry_price
                if gain >= trailing_start:
                    new_sl = max(pos.stop_loss, current * (1.0 - trailing_step))
                    if new_sl > pos.stop_loss:
                        pos.stop_loss = float(new_sl)
                        self._save_positions_state()
                        logger.info(f"🔧 Трейлинг-стоп подтянут: {pos.symbol} SL={pos.stop_loss:.6f}")
                        if self.notifier:
                            try:
                                self.notifier.notify_trailing(pos)
                            except Exception:
                                logger.exception("notify_trailing failed")

            # SL / TP — финальные выходы
            if current <= pos.stop_loss:
                self._final_close(pos, reason="STOP", paper=paper)
            elif current >= pos.take_profit:
                self._final_close(pos, reason="TP", paper=paper)

    def _final_close(self, pos: Position, reason: str, paper: bool):
        """
        Единая точка финального закрытия (TP/SL). Вызывает ML-колбэк, персистит состояние
        и дописывает результат сделки в CSV-лог.
        """
        exit_price: Optional[float] = None
        closed_qty: float = float(pos.qty)

        try:
            if pos.qty > 0:
                order = self.dm.place_market_order(pos.symbol, "SELL", pos.qty, paper=paper)
                # Пытаемся взять цену выхода из ответа биржи/симулятора
                try:
                    if isinstance(order, dict):
                        fills = order.get("fills") or []
                        if fills and "price" in fills[0]:
                            exit_price = float(fills[0]["price"])
                except Exception:
                    exit_price = None
                # Фолбэк: берём текущую рыночную цену
                if not exit_price or exit_price <= 0:
                    exit_price = float(self.dm.get_current_price(pos.symbol))
                pos.qty = 0.0
        except Exception:
            logger.exception("Не удалось отправить SELL на финальном закрытии")

        if reason == "TP":
            pos.status = "PROFIT"
            success = True
            log_fn = logger.info
            log_msg = f"✅ Take-Profit достигнут для {pos.symbol}"
        else:
            pos.status = "STOPPED"
            success = False
            log_fn = logger.warning
            log_msg = f"❌ Stop-Loss сработал для {pos.symbol}"

        # ✍️ Запишем сделку в CSV-лог (даже если цена фолбэкнулась с текущего тикера)
        try:
            self._append_trade_result(pos, reason, exit_price, closed_qty)
        except Exception:
            logger.exception("Не удалось записать результат сделки в CSV")

        self._save_positions_state()
        log_fn(log_msg)

        if self.notifier:
            try:
                self.notifier.notify_closed(pos, reason)
            except Exception:
                logger.exception("notify_closed failed")

        # Колбэк для онлайн-обучения ARF: success=True (TP) / False (SL)
        cb = getattr(self, "on_position_closed", None)
        if callable(cb):
            try:
                cb(pos, success)
            except Exception:
                logger.exception("on_position_closed callback failed")


    # ===== Персист состояния позиций =====
    def _trades_log_path(self) -> str:
        # Настраиваемый путь из конфигурации, по умолчанию state/trades.csv
        return str(getattr(self.cfg, "TRADES_LOG_PATH", os.path.join("state", "trades.csv")))

    def _append_trade_result(self, pos: Position, reason: str, exit_price: Optional[float], closed_qty: float):
        """
        Добавляет строку в CSV-лог закрытых сделок.
        """
        path = self._trades_log_path()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            logger.exception("Не удалось создать каталог для лога сделок")

        exit_px = float(exit_price) if exit_price else float(self.dm.get_current_price(pos.symbol))
        entry_px = float(pos.entry_price)
        qty = float(closed_qty)
        pnl_usdt = (exit_px - entry_px) * qty
        pnl_pct = (exit_px / entry_px - 1.0) * 100.0

        duration_sec = None
        try:
            if isinstance(pos.entry_time, datetime):
                duration_sec = int((datetime.now() - pos.entry_time).total_seconds())
        except Exception:
            duration_sec = None

        row = {
            "close_time": datetime.now().isoformat(timespec="seconds"),
            "symbol": pos.symbol,
            "reason": reason,                         # "TP" или "STOP"
            "qty": f"{qty:.6f}",
            "entry_price": f"{entry_px:.6f}",
            "exit_price": f"{exit_px:.6f}",
            "pnl_usdt": f"{pnl_usdt:.6f}",
            "pnl_pct": f"{pnl_pct:.4f}",
            "duration_sec": duration_sec if duration_sec is not None else "",
            "ml_pred": f"{float(pos.ml_pred):.6f}" if getattr(pos, "ml_pred", None) is not None else "",
        }

        write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "close_time","symbol","reason","qty",
                        "entry_price","exit_price","pnl_usdt","pnl_pct",
                        "duration_sec","ml_pred",
                    ],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            logger.info(f"💾 Результат сделки сохранён: {row}")
        except Exception:
            logger.exception("Не удалось дописать CSV-лог сделок")


    def _positions_state_path(self) -> str:
        return str(getattr(self.cfg, "POSITIONS_STATE_PATH", os.path.join("state", "positions.json")))

    def _ensure_state_dir(self):
        try:
            directory = os.path.dirname(self._positions_state_path()) or "."
            os.makedirs(directory, exist_ok=True)
        except Exception:
            logger.exception("Не удалось создать каталог для state")

    def _serialize_pos(self, pos: Position) -> Dict:
        d = {f.name: getattr(pos, f.name) for f in fields(Position)}
        # datetime → ISO
        et = d.get("entry_time")
        if isinstance(et, datetime):
            d["entry_time"] = et.isoformat()
        return d

    def _deserialize_pos(self, d: Dict) -> Position:
        """
        Аккуратно восстанавливаем dataclass Position.
        Лишние поля игнорируем (чтобы не падать на старых state).
        """
        fnames = {f.name for f in fields(Position)}
        dd = dict(d)

        # ISO → datetime
        et = dd.get("entry_time")
        if isinstance(et, str):
            try:
                dd["entry_time"] = datetime.fromisoformat(et)
            except Exception:
                dd["entry_time"] = datetime.now()

        # Сконструировать только известными полями
        kwargs = {k: dd[k] for k in dd.keys() if k in fnames}
        # Бэкап по умолчаниям
        kwargs.setdefault("status", "ACTIVE")
        kwargs.setdefault("ml_features", None)
        kwargs.setdefault("ml_pred", None)

        # Списки защитим от None
        kwargs["partial_tps"] = list(kwargs.get("partial_tps") or [])
        kwargs["partial_sizes"] = list(kwargs.get("partial_sizes") or [])
        fps = kwargs.get("filled_partials")
        if fps is None:
            kwargs["filled_partials"] = [False] * len(kwargs["partial_sizes"])
        else:
            kwargs["filled_partials"] = list(fps)

        return Position(**kwargs)

    def _save_positions_state(self):
        try:
            self._ensure_state_dir()
            payload = {"positions": [self._serialize_pos(p) for p in self.positions]}
            tmp = self._positions_state_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._positions_state_path())
        except Exception as e:
            logger.warning(f"Не удалось сохранить state позиций: {e}")

    def _load_positions_state(self):
        try:
            path = self._positions_state_path()
            if not os.path.isfile(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded: List[Position] = []
            for item in data.get("positions", []):
                try:
                    loaded.append(self._deserialize_pos(item))
                except Exception:
                    logger.exception("Не удалось десериализовать позицию из state")
            if loaded:
                self.positions = loaded
                logger.info(f"Загружено позиций из state: {len(self.positions)}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить state позиций: {e}")
