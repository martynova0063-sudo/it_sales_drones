# src/telegram_notifier.py

import requests
import time
from typing import Optional, Dict, Any

import config_monitor
from src.runtime_logger import get_runtime_logger

class TelegramNotifier:
    """
    Класс для отправки уведомлений и системных сообщений в Telegram.
    Инкапсулирует всю логику взаимодействия с Telegram Bot API.
    """

    def __init__(self, bot_token: Optional[str], channel_id: Optional[str], chat_id: Optional[str]):
        """
        Инициализирует TelegramNotifier.

        Args:
            bot_token: Токен Telegram бота.
            channel_id: ID Telegram канала для системных сообщений (например, закрепление URL ПУ).
            chat_id: ID Telegram чата для отправки алертов.
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.chat_id = chat_id
        self.logger = get_runtime_logger("TelegramNotifier")

        if not self.bot_token:
            self.logger.warning("Telegram BOT_TOKEN не установлен. Отправка сообщений в Telegram невозможна.")
        if not self.channel_id:
            self.logger.warning("Telegram CHANNEL_ID не установлен. Закрепление системного URL в Telegram не будет работать.")
        if not self.chat_id:
            self.logger.warning("Telegram CHAT_ID не установлен. Отправка алертов в Telegram не будет работать.")

    def is_configured(self) -> bool:
        """Проверяет, достаточно ли настроек для работы с Telegram API."""
        return bool(self.bot_token and (self.channel_id or self.chat_id))

    def send_system_url_message(self, pu_alert_url: str, timestamp: int):
        """
        Отправляет и закрепляет системный URL ПУ в Telegram канале.

        Args:
            pu_alert_url: URL ПУ, куда ноды должны отправлять алерты.
            timestamp: Временная метка запуска ПУ.
        """
        if not (self.bot_token and self.channel_id):
            self.logger.warning("Недостаточно настроек Telegram (BOT_TOKEN или CHANNEL_ID) для отправки системного URL.")
            return

        conn_info_str = f"{timestamp},{pu_alert_url}"
        pinned_text = f"SYSTEM_URL: {conn_info_str}" 
        human_text = f"Ситуационный центр запущен!\nURL для Нод: {pu_alert_url}\nTimestamp: {timestamp}"

        try:
            # Отправляем сообщение для закрепления
            response = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage", 
                json={"chat_id": self.channel_id, "text": pinned_text},
                timeout=5
            )
            response.raise_for_status()
            message_id = response.json().get("result", {}).get("message_id")
            
            if message_id:
                # Закрепляем сообщение
                requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/pinChatMessage", 
                    json={"chat_id": self.channel_id, "message_id": message_id},
                    timeout=5
                ).raise_for_status()
                self.logger.info("URL сервера отправлен и закреплен в Telegram-канале.")
            
            # Отправляем более дружелюбное сообщение
            requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage", 
                json={"chat_id": self.channel_id, "text": human_text},
                timeout=5
            ).raise_for_status()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка работы с Telegram API при отправке/закреплении системного URL: {e}")
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при отправке системного URL в Telegram: {e}")

    def send_alert_photo(self, image_bytes: bytes, caption: str):
        """
        Отправляет фотографию с алертом в Telegram чат.

        Args:
            image_bytes: Байты изображения JPEG.
            caption: Текст подписи к изображению.
        """
        if not (self.bot_token and self.chat_id):
            self.logger.warning("Недостаточно настроек Telegram (BOT_TOKEN или CHAT_ID) для отправки алерта.")
            return

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            files = {"photo": ("alert.jpg", image_bytes, "image/jpeg")}
            response = requests.post(
                url, 
                data={"chat_id": self.chat_id, "caption": caption}, 
                files=files, 
                timeout=5
            )
            response.raise_for_status()
            self.logger.info("Алерт успешно отправлен в Telegram.")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка отправки алерта в Telegram: {e}")
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при отправке алерта в Telegram: {e}")