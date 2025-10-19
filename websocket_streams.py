import logging
from binance import ThreadedWebsocketManager

logger = logging.getLogger(__name__)

class WebsocketStreams:
    def __init__(self, config, data_manager):
        self.cfg = config
        self.dm = data_manager
        self.twm = None
        self.symbol_handlers = {}

    def start(self):
        if not self.cfg.WEBSOCKETS_ENABLED:
            return
        try:
            self.twm = ThreadedWebsocketManager(api_key=self.dm.client.API_KEY, api_secret=self.dm.client.API_SECRET)
            self.twm.start()
        except Exception as e:
            logger.warning(f"Не удалось запустить WebSocket: {e}")

    def subscribe_symbol(self, symbol: str):
        if not self.twm: return
        sym_l = symbol.lower()
        if sym_l in self.symbol_handlers: return
        def _cb(msg):
            try:
                if msg.get('e') == '24hrMiniTicker':
                    self.dm.last_price[symbol] = float(msg['c'])
            except Exception:
                from error_logger import log_exception
                log_exception("Error in subscribe_symbol")
        try:
            self.symbol_handlers[sym_l] = self.twm.start_symbol_miniticker_socket(callback=_cb, symbol=symbol)
        except Exception as e:
            logger.warning(f"WebSocket подписка на {symbol} не удалась: {e}")

    def stop(self):
        try:
            if self.twm:
                self.twm.stop()
        except Exception:
            from error_logger import log_exception
            log_exception("Error in stop")
