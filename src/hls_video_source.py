# src/hls_video_source.py

import subprocess
import numpy as np
import logging

from pathlib import Path
from typing import Optional, Tuple
import shutil
import signal
import time
from src.logger import get_logger
import config_monitor

logger = get_logger()

class HLSVideoSource:
    """
    Источник видеопотока для HLS-трансляций (`.m3u8`).
    
    Использует FFmpeg как внешний процесс для декодирования HLS-потока
    и предоставляет кадры в формате BGR (numpy array), совместимом с OpenCV.
    Поддерживает передачу HTTP-заголовков (например, Referer) для доступа
    к защищённым потокам.
    """

    def __init__(
        self,
        stream_url: str,
        referer: Optional[str] = None,
        user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        # target_fps: int = 5,
        frame_size: Tuple[int, int] = (1280, 720),
        page_url: Optional[str] = None
    ) -> None:
        """Инициализирует HLS-источник и запускает FFmpeg.

        Args:
            stream_url: Полный URL до `.m3u8` манифеста.
            referer: Значение HTTP-заголовка Referer (обязательно для многих защищённых потоков).
            user_agent: Значение HTTP User-Agent.
            # target_fps: Желаемая частота кадров на выходе (FFmpeg будет прореживать).
            frame_size: Целевой размер кадра (ширина, высота). По умолчанию 1280x720.

        Raises:
            RuntimeError: Если FFmpeg не установлен или процесс завершился с ошибкой.
        """
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg не найден в системе. Установите: sudo apt install ffmpeg")

        self.stream_url = stream_url
        self.referer = referer
        self.user_agent = user_agent

        self.target_fps_config = config_monitor.TARGET_FPS ## Получаем целевой FPS из конфига
        self.fps = self.target_fps_config ## FPS источника равен TARGET_FPS из конфига
        self.frame_width, self.frame_height = frame_size
        self.frame_bytes = self.frame_width * self.frame_height * 3  # BGR = 3 байта на пиксель
        self.page_url = page_url or referer

        # Формируем заголовки
        headers = ""
        if self.referer:
            headers += f"Referer: {self.referer}\r\n"
        headers += f"User-Agent: {self.user_agent}\r\n"
        headers += "\r\n" # обязательный завершающий CRLF

        # Команда FFmpeg: читает HLS, декодирует, приводит к нужному FPS и размеру, выводит raw BGR в stdout
        self.cmd = [
            "ffmpeg",
            "-headers", headers,
            "-i", self.stream_url,
            "-vf", f"fps={self.target_fps_config},scale={self.frame_width}:{self.frame_height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",  # отключить аудио
            "-v", "error",  # только ошибки в stderr
            "pipe:1"
        ]

        logger.info(f"Запуск FFmpeg для HLS-потока: {self.stream_url} с FPS: {self.target_fps_config}")
        logger.debug(f"Команда FFmpeg: {' '.join(self.cmd)}")


        self.process: Optional[subprocess.Popen] = None
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=10**8
            )
        except Exception as e:
            raise RuntimeError(f"Не удалось запустить FFmpeg: {e}")

        # Небольшая пауза для инициализации
        time.sleep(1)

        # Проверяем, не упал ли процесс сразу
        if self.process.poll() is not None:
            stderr = self.process.stderr.read().decode() if self.process.stderr else "Нет данных"
            raise RuntimeError(f"FFmpeg завершился сразу после запуска. stderr: {stderr}")

    def read_frame(self) -> Optional[np.ndarray]:
        """Читает следующий кадр из HLS-потока.

        Returns:
            Кадр в формате BGR (HxWxC, dtype=uint8) или None, если поток завершён или ошибка.
        """

        if self.process is None or self.process.poll() is not None:

            stderr_data = ""
            # Процесс уже завершён
            if self.process is not None and self.process.stderr:
                stderr_data = self.process.stderr.read().decode() if self.process.stderr else ""
            logger.error(f"FFmpeg завершился. stderr: {stderr_data}")
            return None
        
        if self.process.stdout is None:
            return None

        try:
            raw = self.process.stdout.read(self.frame_bytes)
            if len(raw) != self.frame_bytes:
                # Не хватает данных — поток, вероятно, оборвался
                return None

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.frame_height, self.frame_width, 3)
            )
            return frame

        except Exception as e:
            logger.error(f"Ошибка при чтении кадра: {e}")
            return None

    def is_alive(self) -> bool:
        """Проверяет, работает ли поток."""
        if self.process is None:
            return False
        if self.process.poll() is not None:
            return False
        return True

    def release(self) -> None:
        """Корректно завершает FFmpeg-процесс и освобождает ресурсы."""

        if self.process is not None and self.process.poll() is None:
            logger.info("Остановка FFmpeg-процесса...")

            self.process.send_signal(signal.SIGINT)  # вежливый запрос на завершение
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg не ответил — принудительное завершение")
                self.process.kill()
            finally:
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
        self.process = None

    def refresh(self, new_url: str | None = None) -> bool:
        """Обновляет поток с новым URL.
        
        Args:
            new_url: Новый URL. Если None — попробует получить через stream_refresher.
        
        Returns:
            True, если обновление прошло успешно.
        """
        old_process = self.process
        old_url = self.stream_url

        if new_url is None:
            from src.stream_refresher import get_fresh_hls_url
            if not self.page_url:
                logger.error("Нет page_url для обновления потока")
                return False
            new_url = get_fresh_hls_url(
                camera_page_url=self.page_url,
                referer=self.referer or self.page_url
            )
            if not new_url:
                logger.error("Не удалось получить новый URL")
                return False

        # Останавливаем текущий процесс
        self.release()

        # Пересоздаём с новым URL
        try:
            self.__init__(
                stream_url=new_url,
                referer=self.referer,
                user_agent=self.user_agent,
                frame_size=(self.frame_width, self.frame_height),
                page_url=self.page_url
            )
            logger.info(f"Поток обновлён: {new_url}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при обновлении потока: {e}")
            # Восстанавливаем старый URL
            self.stream_url = old_url
            self.process = old_process
            return False