# src/public_address_manager.py

import asyncio
import socket
import subprocess
import time
import atexit
from typing import Optional, Literal
from pyngrok import ngrok, conf

import config_monitor
from src.runtime_logger import get_runtime_logger

class PublicAddressManager:
    """
    Управляет определением публичного или локального IP-адреса для сервиса
    (ПУ или Нода) на основе настроек config_monitor.
    Поддерживает Ngrok, Tailscale, статический IP и только локальный доступ.
    """

    def __init__(self, service_port: int, public_url_override: str = "", service_name: Literal["App", "Node"] = "App"):
        """
        Инициализирует менеджер публичных адресов.

        Args:
            service_port: Порт, на котором работает локальный сервис (например, 9000 для ПУ, 8001 для Ноды).
            public_url_override: Публичный URL, переданный из аргументов командной строки, если есть.
            service_name: Имя сервиса (для логирования), "App" для пульта, "Node" для ноды.
        """
        self.service_port = service_port
        self.public_url_override = public_url_override
        self.service_name = service_name
        self.logger = get_runtime_logger(f"PublicAddressManager_{service_name}")
        
        self._ngrok_tunnel: Optional[ngrok.NgrokTunnel] = None
        self._public_resolved_url: str = ""
        self._is_initialized: bool = False

    def _get_local_ip(self) -> str:
        """
        Возвращает локальный IP-адрес устройства (в приоритете - Tailscale, затем LAN).

        Returns:
            Локальный IP-адрес как строка.
        """
        # 1. Попытка определить Tailscale IP через команду `tailscale ip -4`
        try:
            tailscale_ip_output = subprocess.check_output(["tailscale", "ip", "-4"], text=True).strip()
            if tailscale_ip_output.startswith("100."):
                self.logger.info(f"Обнаружен Tailscale IP: {tailscale_ip_output}")
                return tailscale_ip_output
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.debug("Tailscale клиент не запущен или команда 'tailscale ip -4' не найдена.")
        except Exception as e:
            self.logger.warning(f"Ошибка при определении Tailscale IP: {e}")

        # 2. Попытка определить локальный IP (LAN IP) через сокет
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
            s.close()
            self.logger.info(f"Обнаружен LAN IP: {lan_ip}")
            return lan_ip
        except Exception as e:
            self.logger.error(f"Не удалось определить локальный IP: {e}")
            return "127.0.0.1"

    async def initialize_public_address(self) -> str:
        """
        Определяет публичный/виртуальный адрес сервиса в зависимости от PUBLIC_IP_PROVIDER.
        Если --public_url_override передан, он имеет приоритет.

        Returns:
            Разрешенный публичный URL сервиса.
        """
        if self._is_initialized:
            self.logger.info(f"{self.service_name}: PublicAddressManager уже инициализирован. Пропускаем повторный запуск.")
            return self._public_resolved_url

        if self.public_url_override:
            self._public_resolved_url = self.public_url_override
            self.logger.info(f"{self.service_name} использует публичный URL из аргументов: {self._public_resolved_url}")
            self._is_initialized = True
            return self._public_resolved_url

        local_ip = self._get_local_ip()
        base_url_local = f"http://{local_ip}:{self.service_port}"

        if config_monitor.PUBLIC_IP_PROVIDER == "NGROK":
            self.logger.info(f"{self.service_name}: Провайдер IP: Ngrok.")
            if not config_monitor.NGROK_AUTHTOKEN:
                self.logger.warning("NGROK_AUTHTOKEN не найден. Ngrok не будет запущен. Используем localhost.")
                self._public_resolved_url = f"http://localhost:{self.service_port}"
            else:
                ngrok.set_auth_token(config_monitor.NGROK_AUTHTOKEN)
                self.logger.info("Ngrok authtoken установлен.")
                max_retries = 5
                retry_delay = 3 # seconds
                for attempt in range(max_retries):
                    self.logger.info(f"Попытка запуска Ngrok туннеля для {self.service_name} (попытка {attempt + 1}/{max_retries})...")
                    # Небольшая задержка перед подключением, чтобы сервис успел запуститься
                    if attempt > 0: await asyncio.sleep(retry_delay)
                    try:
                        self._ngrok_tunnel = ngrok.connect(str(self.service_port), "http")

                        # Проверяем, что public_url не None перед присвоением
                        if self._ngrok_tunnel.public_url:
                            self._public_resolved_url = self._ngrok_tunnel.public_url
                            self.logger.info(f"Ngrok туннель для {self.service_name} успешно запущен: {self._public_resolved_url}")

                        else:
                            raise RuntimeError("Ngrok туннель не предоставил публичный URL.")
                        
                        break
                    except Exception as e:
                        self.logger.error(f"Ошибка запуска Ngrok для {self.service_name} (попытка {attempt + 1}): {e}")
                        if attempt == max_retries - 1:
                            self.logger.error(f"Все попытки запуска Ngrok для {self.service_name} исчерпаны. Используем localhost как резерв.")
                            self._public_resolved_url = f"http://localhost:{self.service_port}"
                            break
                if self._ngrok_tunnel:
                    atexit.register(self.cleanup_ngrok)

        elif config_monitor.PUBLIC_IP_PROVIDER == "STATIC_IP":
            self.logger.info(f"{self.service_name}: Провайдер IP: Статический IP.")
            if config_monitor.STATIC_IP_ADDRESS:
                self._public_resolved_url = f"http://{config_monitor.STATIC_IP_ADDRESS}:{self.service_port}"
                self.logger.info(f"{self.service_name}: Используется статический IP: {self._public_resolved_url}")
            else:
                self.logger.warning("STATIC_IP_ADDRESS не указан. Используем localhost.")
                self._public_resolved_url = f"http://localhost:{self.service_port}"
        elif config_monitor.PUBLIC_IP_PROVIDER == "TAILSCALE":
            self.logger.info(f"{self.service_name}: Провайдер IP: Tailscale.")
            if local_ip.startswith("100."):
                self._public_resolved_url = base_url_local
                self.logger.info(f"{self.service_name}: Используется Tailscale IP: {self._public_resolved_url}")
            else:
                self.logger.warning("Tailscale IP не обнаружен. Убедитесь, что Tailscale запущен и авторизован. Используем localhost.")
                self._public_resolved_url = f"http://localhost:{self.service_port}"
        elif config_monitor.PUBLIC_IP_PROVIDER == "LOCAL_ONLY":
            self.logger.info(f"{self.service_name}: Провайдер IP: Только локальный (LOCAL_ONLY).")
            self._public_resolved_url = f"http://localhost:{self.service_port}"
        else:
            self.logger.warning(f"{self.service_name}: Неизвестный PUBLIC_IP_PROVIDER: {config_monitor.PUBLIC_IP_PROVIDER}. Используем localhost.")
            self._public_resolved_url = f"http://localhost:{self.service_port}"
        
        self._is_initialized = True
        if self._public_resolved_url == f"http://localhost:{self.service_port}":
            self.logger.warning(f"{self.service_name}: Используется локальный адрес сервера. Удаленные ноды/ПУ не смогут подключиться.")

        return self._public_resolved_url

    def get_public_url(self) -> str:
        """
        Возвращает ранее разрешенный публичный URL. Должен быть вызван после initialize_public_address.
        """
        if not self._is_initialized:
            self.logger.error(f"{self.service_name}: PublicAddressManager не инициализирован. Вызовите initialize_public_address сначала.")
            return f"http://localhost:{self.service_port}" # Fallback
        return self._public_resolved_url

    def cleanup_ngrok(self):
        """Останавливает Ngrok туннель, если он был запущен."""
        if self._ngrok_tunnel:
            self.logger.info(f"{self.service_name}: Остановка Ngrok туннеля.")
            try:
                self._ngrok_tunnel.disconnect() # type: ignore
                # self._ngrok_tunnel.kill() # kill может быть слишком агрессивным
            except Exception as e:
                self.logger.error(f"{self.service_name}: Ошибка при остановке Ngrok туннеля: {e}")
            self._ngrok_tunnel = None