# src/runtime_logger.py

import logging
import sys
from pathlib import Path
# Добавь путь к корню, если питон не видит config_monitor
sys.path.append(str(Path(__file__).parent.parent)) 
import config_monitor

def get_runtime_logger(name: str):
    """Логгер для Node и App."""
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
        
###     logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Консоль
    ch = logging.StreamHandler(sys.stdout)
###     ch.setLevel(logging.INFO)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Файл (logs/runtime.log)
    log_dir = config_monitor.LOG_DIR
    log_file = log_dir / "runtime.log"
    
    try:
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
###         fh.setLevel(logging.INFO)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Ошибка создания лог-файла: {e}")

    return logger