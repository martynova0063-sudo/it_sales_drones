# src/rtsp_video_source.py

import subprocess
import numpy as np
import logging
import shutil
import signal
import time
import json
import cv2
import threading

from pathlib import Path
from typing import Optional, Tuple, Dict, Any 

from src.runtime_logger import get_runtime_logger
logger = get_runtime_logger("RTSPVideoSource")

def _log_stderr_thread(pipe, logger):
    """Функция для чтения stderr в отдельном потоке, чтобы избежать deadlock."""
    try:
        for line in iter(pipe.readline, b''):
            logger.info(f"FFmpeg STDERR: {line.decode('utf-8', errors='ignore').strip()}")
    except Exception as e:
        logger.error(f"Ошибка в потоке чтения stderr: {e}")
    finally:
        if pipe:
            pipe.close()

class RTSPVideoSource:
    """
    Источник видеопотока для RTSP-трансляций.
    
    Использует FFmpeg как внешний процесс для декодирования RTSP-потока
    и предоставляет кадры в формате BGR (numpy array), совместимом с OpenCV.
    """

    def __init__(
        self,
        stream_url: str,
        # target_fps: int = 30, # Можно сделать настраиваемым
        # frame_size: Tuple[int, int] = (1280, 720) # Дефолтный размер, можно сделать конфигурируемым
    ) -> None:
        """Инициализирует RTSP-источник и запускает FFmpeg.

        Args:
            stream_url: Полный URL до RTSP-попока (rtsp://...).
            # frame_size: Целевой размер кадра (ширина, высота). По умолчанию 1280x720.

        Raises:
            RuntimeError: Если FFmpeg не установлен или процесс завершился с ошибкой.
        """
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg не найден в системе. Установите: sudo apt install ffmpeg")
        if not shutil.which("ffprobe"): 
            raise RuntimeError("FFprobe не найден в системе. Установите: sudo apt install ffmpeg (обычно идет в комплекте)") 
        
        self.stream_url = stream_url
        self.process: Optional[subprocess.Popen] = None
        self.fps: float = 30.0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.frame_bytes: int = 0
###         self._debug_frame_counter = 0

        # 1. Автоматическое определение параметров потока с помощью ffprobe
        probe_cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            "-i", self.stream_url
        ]
        logger.info(f"Запуск FFprobe для анализа RTSP-потока: {' '.join(probe_cmd)}")
        try:
            probe_output = subprocess.check_output(probe_cmd, stderr=subprocess.PIPE, timeout=10, text=True)
            probe_data = json.loads(probe_output)
            
            # Проверяем, что стримы вообще есть и видеострим найден
            if not probe_data.get("streams"):
                raise ValueError("FFprobe не обнаружил видеостримов.")
            
            video_stream = probe_data["streams"][0]

            self.frame_width = video_stream.get("width", 1280)
            self.frame_height = video_stream.get("height", 720)
            
            # Исходный FPS может быть 60, но мы возьмем меньше.
            TARGET_PROCESSING_FPS = 15

            avg_frame_rate = video_stream.get("avg_frame_rate", "30/1")
            try:
                num, den = map(int, avg_frame_rate.split('/'))
                source_fps = num / den if den > 0 else 30.0
                ### self.fps = num / den if den > 0 else 30.0
            except (ValueError, IndexError):
                source_fps = 30.0

            self.fps = float(TARGET_PROCESSING_FPS)

            # Дополнительная диагностика: логируем pix_fmt из ffprobe
            detected_pix_fmt = video_stream.get("pix_fmt", "unknown")
            logger.info(f"FFprobe обнаружил поток: Разрешение {self.frame_width}x{self.frame_height}, Исходный FPS: {source_fps:.2f}, Pix_fmt: {detected_pix_fmt}")
            
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, IndexError, subprocess.TimeoutExpired, ValueError) as e:
            logger.error(f"Не удалось определить параметры RTSP-потока с помощью FFprobe: {e}. Используем дефолтные 1280x720@30FPS.")
            self.frame_width, self.frame_height = 1280, 720
            self.fps = 30.0
        
        self.frame_bytes = self.frame_width * self.frame_height * 3 # BGR = 3 байта на пиксель


        # 2. Основная команда FFmpeg для декодирования 
        self.cmd = [
            "ffmpeg"
        ]

        # Добавляем опцию транспорта только для RTSP
        if self.stream_url.startswith("rtsp://"): ## ДОБАВЛЕНО
            self.cmd.extend(["-rtsp_transport", "tcp"]) ## ДОБАВЛЕНО

        self.cmd.extend([
            "-i", self.stream_url,
            "-vf", f"fps={self.fps}", # Принудительно снижаем FPS до целевого (15)
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",  # отключить аудио
            "-v", "info", # Устанавливаем максимальное логирование для диагностики
            "pipe:1"
        ])


        logger.info(f"Запуск FFmpeg для RTSP-потока: {self.stream_url}. Ожидаемые параметры: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS") 
        logger.debug(f"Команда FFmpeg: {' '.join(self.cmd)}")

###        self.process: Optional[subprocess.Popen] = None

        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=10**8
            )
        except Exception as e:
            raise RuntimeError(f"Не удалось запустить FFmpeg для RTSP: {e}")


        # Запускаем поток для неблокирующего чтения stderr
        if self.process.stderr:
            stderr_thread = threading.Thread(target=_log_stderr_thread, args=(self.process.stderr, logger), daemon=True)
            stderr_thread.start()

        # Небольшая пауза для инициализации
        time.sleep(2)

        # Проверяем, не упал ли процесс сразу
        if self.process.poll() is not None:
            ### stderr_output = self.process.stderr.read().decode() if self.process.stderr else "Нет данных"
            ### raise RuntimeError(f"FFmpeg завершился сразу после запуска для RTSP. stderr: {stderr_output}")
            raise RuntimeError(f"FFmpeg завершился сразу после запуска для RTSP. Проверьте логи.")

        # Попытка определить реальные параметры потока после запуска ffmpeg, если не заданы жестко
        # Это более сложная доработка, пока используем заданные frame_size и fps
        # Для начала, попробуем читать первый кадр, чтобы получить его реальный размер
        if self.process.stdout is None:
            raise RuntimeError("FFmpeg stdout пайп не инициализирован.")

        first_raw = self.process.stdout.read(self.frame_bytes)
        if len(first_raw) != self.frame_bytes:
            ### stderr_output = self.process.stderr.read().decode() if self.process.stderr else "Нет данных"

            logger.warning(f"FFmpeg не смог отдать первый кадр полного размера ({len(first_raw)}/{self.frame_bytes} байт). Поток может быть некорректным или параметры неверны.")
            logger.error(f"FFmpeg stderr при попытке первого кадра: проверьте логи DEBUG уровня.")            
            raise RuntimeError("FFmpeg не смог инициализироваться с корректными кадрами.")

        # Если первый кадр успешно прочитан, можно обновить frame_width/height, если они еще не установлены
        # (или если мы хотим использовать реальные данные потока, а не дефолтные)
        # Пока оставляем заданные дефолтные для простоты.
        
        # Возвращаем первый прочитанный кадр в буфер, чтобы read_frame мог его использовать
        # Это не так просто с pipe. Лучше использовать временный буфер или просто выкинуть его,
        # так как основная логика read_frame будет читать дальше.
        # Для простоты, первый кадр будет считан и проигнорирован здесь.

    def read_frame(self) -> Optional[np.ndarray]:
        """Читает следующий кадр из RTSP-потока.

        Returns:
            Кадр в формате BGR (numpy array) или None, если поток завершён или ошибка.
        """
        
        if self.process is None:
            logger.error("FFmpeg-процесс не запущен.")
            return None
        
        # Читаем stderr асинхронно, чтобы не блокировать и не переполнять буфер stderr
        # Для debug уровня stderr может быть очень большим
        # Простая неблокирующая чтение stderr для диагностики
        if self.process.stderr:
            try:
#                 debug_stderr_output = self.process.stderr.read1(1024).decode(errors='ignore').strip()
#                 if debug_stderr_output:
#                     logger.debug(f"FFmpeg STDERR (live): {debug_stderr_output}")
                pass
            except Exception as e:
                logger.debug(f"Ошибка чтения live stderr: {e}")

        if self.process.poll() is not None:
            stderr_output = self.process.stderr.read().decode() if self.process.stderr else "Нет данных"
            logger.error(f"FFmpeg для RTSP завершился. STDERR: {stderr_output}")
            return None
        
        if self.process.stdout is None:
            logger.error("stdout пайп FFmpeg не доступен.")
            return None

        try:
            raw = self.process.stdout.read(self.frame_bytes)
            if len(raw) != self.frame_bytes:
                stderr_output = self.process.stderr.read().decode() if self.process.stderr else "Нет данных"
                logger.warning(f"Не хватает данных ({len(raw)}/{self.frame_bytes}) для RTSP-кадра. Поток, вероятно, оборвался. STDERR: {stderr_output}")
                return None

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.frame_height, self.frame_width, 3)
            )

###             if self._debug_frame_counter < 3: # Сохраняем первые 3 кадра
###                 debug_output_dir = Path("debug_frames")
###                 debug_output_dir.mkdir(exist_ok=True)
###                 
###                 raw_filename = debug_output_dir / f"frame_{self._debug_frame_counter:02d}_raw.bin"
###                 with open(raw_filename, 'wb') as f:
###                     f.write(raw)
###                 
###                 # Сохраняем как JPEG, если кадр успешно собран
###                 try:
###                     jpeg_filename = debug_output_dir / f"frame_{self._debug_frame_counter:02d}.jpg"
###                     cv2.imwrite(str(jpeg_filename), frame)
###                     logger.info(f"Отладочный кадр {self._debug_frame_counter} сохранен: {jpeg_filename}")
###                 except Exception as img_e:
###                     logger.error(f"Ошибка сохранения отладочного JPEG: {img_e}")
### 
###                 self._debug_frame_counter += 1

            return frame

        except Exception as e:
            logger.error(f"Ошибка при чтении RTSP-кадра: {e}")
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
            logger.info("Остановка FFmpeg-процесса для RTSP...")
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg для RTSP не ответил — принудительное завершение")
                self.process.kill()
            finally:
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
        self.process = None