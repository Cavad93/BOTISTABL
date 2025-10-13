import logging
import time
import json
from datetime import datetime
from typing import Dict, List
import schedule
import threading

from config import Config
from data_manager import DataManager
from resistance_analyzer import ResistanceAnalyzer
from breakout_detector import BreakoutDetector
from market_scanner import MarketScanner

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BreakoutTradingBot:
    def __init__(self):
        """Инициализация торгового бота"""
        logger.info("Инициализация Breakout Trading Bot...")
        
        self.config = Config()
        
        # Инициализация компонентов
        self.data_manager = DataManager(
            self.config.API_KEY,
            self.config.API_SECRET
        )
        
        self.resistance_analyzer = ResistanceAnalyzer(self.config)
        self.breakout_detector = BreakoutDetector(self.config, self.data_manager)
        self.market_scanner = MarketScanner(
            self.config,
            self.data_manager,
            self.resistance_analyzer,
            self.breakout_detector
        )
        
        self.active_positions = []
        self.trading_history = []
        self.is_running = False
        
        logger.info("Инициализация завершена")
    
    def start(self):
        """Запуск бота"""
        logger.info("Запуск бота...")
        self.is_running = True
        
        # Первоначальное сканирование
        self.scan_and_analyze()
        
        # Настройка расписания
        schedule.every(self.config.SCAN_INTERVAL).seconds.do(self.scan_and_analyze)
        schedule.every(self.config.LEVEL_UPDATE_INTERVAL).seconds.do(self.update_resistance_levels)
        schedule.every(300).seconds.do(self.check_active_positions)  # Каждые 5 минут
        schedule.every(3600).seconds.do(self.generate_report)  # Каждый час
        
        # Запуск в отдельном потоке
        bot_thread = threading.Thread(target=self._run_schedule)
        bot_thread.daemon = True
        bot_thread.start()
        
        logger.info("Бот успешно запущен")
        
        # Главный цикл
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def _run_schedule(self):
        """Выполнение запланированных задач"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def scan_and_analyze(self):
        """Основная функция сканирования и анализа"""
        try:
            logger.info("=" * 50)
            logger.info("Начинаю сканирование рынка...")
            
            # Сканирование
            opportunities = self.market_scanner.scan_market()
            
            if opportunities:
                logger.info(f"Найдено {len(opportunities)} потенциальных возможностей")
                
                # Анализ каждой возможности
                for opp in opportunities[:5]:  # Топ-5
                    self._process_opportunity(opp)
            else:
                logger.info("Подходящих возможностей не найдено")
            
            # Вывод сводки
            summary = self.market_scanner.get_market_summary()
            self._log_market_summary(summary)
            
        except Exception as e:
            logger.error(f"Ошибка при сканировании: {e}")
    
    def _process_opportunity(self, opportunity: Dict):
        """Обработка торговой возможности"""
        symbol = opportunity['symbol']
        score = opportunity['score']
        
        logger.info(f"\n{symbol}:")
        logger.info(f"  Оценка: {score:.2f}/100")
        logger.info(f"  Таймфрейм: {opportunity.get('timeframe', 'N/A')}")
        
        if opportunity.get('breakout_info'):
            breakout = opportunity['breakout_info']
            logger.info(f"  ⚡ ПРОБОЙ ОБНАРУЖЕН!")
            logger.info(f"    Уровень: {breakout['level_price']:.4f}")
            logger.info(f"    Текущая цена: {breakout['breakout_price']:.4f}")
            logger.info(f"    Объем: {breakout['volume_surge']:.2f}x")
            logger.info(f"    RSI: {breakout['rsi']:.2f}")
            logger.info(f"    Уверенность: {breakout['confidence_score']:.2f}")
            
            # Валидация качества пробоя
            is_valid, reason = self.breakout_detector.validate_breakout_quality(breakout)
            
            if is_valid:
                logger.info(f"  ✅ Пробой валиден, готов к входу")
                self._execute_trade(breakout)
            else:
                logger.warning(f"  ❌ Пробой отклонен: {reason}")
        else:
            if opportunity.get('levels'):
                level = opportunity['levels'][0]
                logger.info(f"  Ближайший уровень: {level['price']:.4f} "
                          f"({level['distance_percent']:.2f}% от цены)")
                logger.info(f"  Сила уровня: {level['strength']:.2f}")
                logger.info(f"  Типы: {', '.join(level.get('types', []))}")
    
    def _execute_trade(self, breakout_info: Dict):
        """Исполнение сделки (симуляция)"""
        # В реальной версии здесь будет размещение ордера через API
        
        if len(self.active_positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            logger.warning("Достигнут максимум открытых позиций")
            return
        
        position = {
            'symbol': breakout_info['symbol'],
            'entry_price': breakout_info['entry_price'],
            'entry_time': datetime.now(),
            'stop_loss': breakout_info['stop_loss'],
            'take_profit': breakout_info['take_profit'],
            'confidence': breakout_info['confidence_score'],
            'status': 'ACTIVE'
        }
        
        self.active_positions.append(position)
        logger.info(f"📈 Открыта позиция: {position['symbol']} @ {position['entry_price']:.4f}")
        
        # Сохранение в историю
        self.trading_history.append(position.copy())
    
    def check_active_positions(self):
        """Проверка активных позиций"""
        if not self.active_positions:
            return
        
        logger.info(f"Проверка {len(self.active_positions)} активных позиций...")
        
        for position in self.active_positions[:]:
            current_price = self.data_manager.get_current_price(position['symbol'])
            
            if current_price <= position['stop_loss']:
                logger.warning(f"❌ Stop-Loss сработал для {position['symbol']}")
                position['status'] = 'STOPPED'
                position['exit_price'] = current_price
                position['exit_time'] = datetime.now()
                self.active_positions.remove(position)
                
            elif current_price >= position['take_profit']:
                logger.info(f"✅ Take-Profit достигнут для {position['symbol']}")
                position['status'] = 'PROFIT'
                position['exit_price'] = current_price
                position['exit_time'] = datetime.now()
                self.active_positions.remove(position)
            else:
                pnl_percent = ((current_price - position['entry_price']) / 
                              position['entry_price']) * 100
                logger.info(f"  {position['symbol']}: {pnl_percent:+.2f}%")
    
    def update_resistance_levels(self):
        """Обновление уровней сопротивления для активных символов"""
        logger.info("Обновление уровней сопротивления...")
        
        # Обновляем уровни для символов с активными позициями
        for position in self.active_positions:
            symbol = position['symbol']
            
            for timeframe in self.config.TIMEFRAMES:
                df = self.data_manager.get_klines(symbol, timeframe, limit=500)
                if not df.empty:
                    levels = self.resistance_analyzer.find_resistance_levels(df, symbol)
                    # Кэширование результатов
                    self.resistance_analyzer.levels_cache[f"{symbol}_{timeframe}"] = levels
    
    def generate_report(self):
        """Генерация отчета о работе бота"""
        logger.info("\n" + "=" * 60)
        logger.info("ОТЧЕТ О РАБОТЕ БОТА")
        logger.info("=" * 60)
        
        # Статистика по позициям
        total_trades = len(self.trading_history)
        profitable_trades = sum(1 for t in self.trading_history 
                               if t.get('status') == 'PROFIT')
        
        if total_trades > 0:
            win_rate = (profitable_trades / total_trades) * 100
            logger.info(f"Всего сделок: {total_trades}")
            logger.info(f"Прибыльных: {profitable_trades}")
            logger.info(f"Win Rate: {win_rate:.2f}%")
        
        logger.info(f"Активных позиций: {len(self.active_positions)}")
        
        # Текущие позиции
        if self.active_positions:
            logger.info("\nАктивные позиции:")
            for pos in self.active_positions:
                current_price = self.data_manager.get_current_price(pos['symbol'])
                pnl = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                logger.info(f"  {pos['symbol']}: {pnl:+.2f}%")
        
        # Рыночная сводка
        summary = self.market_scanner.get_market_summary()
        logger.info(f"\nСила рынка: {summary['market_strength']}")
        logger.info(f"Активных пробоев: {summary['active_breakouts']}")
        
        logger.info("=" * 60 + "\n")
    
    def _log_market_summary(self, summary: Dict):
        """Вывод сводки по рынку"""
        logger.info("\n📊 СВОДКА ПО РЫНКУ:")
        logger.info(f"Проанализировано монет: {summary['total_analyzed']}")
        logger.info(f"С уровнями: {summary['coins_with_levels']}")
        logger.info(f"Активных пробоев: {summary['active_breakouts']}")
        logger.info(f"Сила рынка: {summary['market_strength']}")
        
        if summary['top_opportunities']:
            logger.info("\n🎯 ТОП ВОЗМОЖНОСТИ:")
            for i, opp in enumerate(summary['top_opportunities'], 1):
                logger.info(f"{i}. {opp['symbol']} ({opp['timeframe']}) - "
                          f"Score: {opp['score']:.1f}, Action: {opp['action']}")
    
    def stop(self):
        """Остановка бота"""
        logger.info("Остановка бота...")
        self.is_running = False
        
        # Закрытие всех позиций
        if self.active_positions:
            logger.info(f"Закрытие {len(self.active_positions)} активных позиций...")
            for position in self.active_positions:
                position['status'] = 'CLOSED'
                position['exit_time'] = datetime.now()
        
        # Финальный отчет
        self.generate_report()
        
        logger.info("Бот остановлен")

def main():
    """Точка входа"""
    print("""
    ╔══════════════════════════════════════════╗
    ║     BREAKOUT TRADING BOT v1.0           ║
    ║     Binance Resistance Scanner           ║
    ╚══════════════════════════════════════════╝
    """)
    
    bot = BreakoutTradingBot()
    
    try:
        bot.start()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        bot.stop()

if __name__ == "__main__":
    main()
