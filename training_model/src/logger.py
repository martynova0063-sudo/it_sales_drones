# training_model/src/logger.py
import logging
import os

class Logger:
    """
    Класс для кастомного логгера, который выводит сообщения в консоль и в файл.
    """
    def __init__(self, name: str, log_file: str, level: int = logging.INFO) -> None:
        """
        Инициализирует логгер.

        Args:
            name: Имя логгера.
            log_file: Путь к файлу, в который будут записываться логи.
            level: Уровень логирования (например, logging.INFO, logging.DEBUG).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = [] # Очищаем существующие обработчики

        # Обработчик для консоли
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)

        # Обработчик для файла
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        # Устанавливаем уровень логирования по умолчанию для каждого обработчика
        # Консоль: только INFO и выше (без DEBUG)
        ch.setLevel(logging.INFO) 
        # Файл: все уровни, включая DEBUG
        fh.setLevel(logging.DEBUG)
        
        # Общий уровень логгера устанавливаем в DEBUG, чтобы все сообщения были доступны для обработчиков
        self.logger.setLevel(logging.DEBUG) 

    def info(self, message: str) -> None:
        """Логирует информационное сообщение."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Логирует предупреждение."""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """
        Логирует ошибку.

        Args:
            message: Сообщение об ошибке.
            exc_info: Если True, добавляет информацию об исключении.
        """
        self.logger.error(message, exc_info=exc_info)

    def debug(self, message: str) -> None:
        """Логирует отладочное сообщение."""
        self.logger.debug(message)

    def set_level(self, level: int) -> None:
        """Устанавливает уровень логирования для всех обработчиков."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)