# src/logger.py

import logging
from pathlib import Path

# Конфигурация Логгера
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "validator.log"

# Убедимся, что папка для логов существует
LOG_DIR.mkdir(exist_ok=True)

# Инициализация логгера
logger = logging.getLogger('DatasetValidator')
logger.setLevel(logging.INFO)

# Форматтер для консоли (минимальный)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')

# Обработчик для консоли
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(console_formatter)
logger.addHandler(ch)

# Обработчик для файла (для полной истории)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_formatter)
logger.addHandler(fh)

def get_logger():
    """Возвращает сконфигурированный логгер."""
    return logger