# src/video_writer.py

import os
import cv2
from datetime import datetime
from pathlib import Path
from src.runtime_logger import get_runtime_logger
from typing import Optional, Any, Dict
import subprocess
import asyncio
import threading

logger = get_runtime_logger("VideoWriter")
class ChunkedVideoWriter:
    """
    Записывает обработанное видео чанками фиксированной длительности.

    Автоматически создаёт новый видеофайл каждые N секунд (по умолчанию 30).
    Использует временные метки в именах файлов для уникальности.
    """
    
    def __init__(self, output_dir: str | Path, fps: float, frame_size: tuple[int, int], chunk_sec: int = 30) -> None:
        """Инициализирует запись видео по чанкам.

        Args:
            output_dir: Директория для сохранения видеофайлов.
            fps: Кадровая частота исходного видео.
            frame_size: Размер кадра в формате (ширина, высота).
            chunk_sec: Длительность одного видеофайла в секундах.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self.chunk_frames = int(fps * chunk_sec)
        self.current_writer = None
        self.frame_count = 0
        self.current_chunk_path: Optional[Path] = None
        self.chunk_repacked_events: Dict[Path, asyncio.Event] = {}

    def write(self, frame) -> None:
        """
        Записывает кадр в текущий или новый видеофайл.
        При достижении границы чанка закрывает текущий файл и создаёт новый.

        Args:
            frame: Изображение в формате BGR (numpy array, HxWxC).
        """
        if self.frame_count % self.chunk_frames == 0:
            if self.current_writer:
                self.current_writer.release()
                logger.info(f"Закрыт видео-чанк: {self.current_chunk_path}")


                # Принудительная репаковка предыдущего чанка FFmpeg
                # Запускаем репаковку в отдельном потоке, чтобы не блокировать основной поток Node
                if self.current_chunk_path and self.current_chunk_path.exists():
                    old_path = self.current_chunk_path
                    # Создаем новый временный путь для перепакованного файла
                    new_path = old_path.parent / f"{old_path.stem}_repacked.mp4"
                    
                    repack_event_for_old_path = self.chunk_repacked_events.get(old_path)
                    if not repack_event_for_old_path:
                        repack_event_for_old_path = asyncio.Event()
                        self.chunk_repacked_events[old_path] = repack_event_for_old_path


                    def _repack_sync_task(input_path: Path, output_path: Path, original_path: Path, repack_event: asyncio.Event):
                        """Синхронная задача для репаковки файла с FFmpeg."""
                        logger.info(f"Начало репаковки чанка: {input_path.name} -> {output_path.name}")
                        cmd = [
                            "ffmpeg",
                            "-y", # Перезаписать файл, если существует
                            "-i", str(input_path),
                            "-c", "copy", # Просто скопировать потоки
                            "-movflags", "faststart", # Поместить moov atom в начало для быстрой отдачи
                            str(output_path)
                        ]
                        try:
                            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                            logger.info(f"Успешно перепакован чанк: {input_path.name} -> {output_path.name}")
                            # После успешной репаковки, заменяем оригинальный файл перепакованным
                            os.remove(input_path)
                            os.rename(output_path, original_path)
                            logger.info(f"Оригинальный чанк заменен перепакованным: {original_path.name}")
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Ошибка FFmpeg при репаковке чанка '{input_path.name}': {e.returncode}")
                            logger.error(f"STDOUT: {e.stdout}")
                            logger.error(f"STDERR: {e.stderr}")
                            if output_path.exists(): os.remove(output_path)
                        except Exception as e:
                            logger.error(f"Непредвиденная ошибка при репаковке чанка '{input_path.name}': {e}")
                        finally:
                            # Убедимся, что временный файл, если он остался, удален
                            if output_path.exists(): os.remove(output_path) # Удаляем временный репакованный файл, если он не был переименован
                            repack_event.set()
                            # (на всякий случай, если os.rename не сработал или был перехвачен)
                            # P.S. В текущей логике new_path переименовывается в original_path, так что этот remove не сработает
                            # если все ок, но это хорошая практика
                    
                    # Запускаем эту синхронную задачу в отдельном потоке
                    threading.Thread(target=_repack_sync_task, args=(old_path, new_path, old_path, repack_event_for_old_path), daemon=True).start()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"output_{timestamp}.mp4"
            self.current_chunk_path = path
            
            self.chunk_repacked_events[path] = asyncio.Event()
            fourcc_code = cv2.VideoWriter_fourcc(*"MJPG") # type: ignore

            try:
                self.current_writer = cv2.VideoWriter(
                    str(path), fourcc_code, self.fps, self.frame_size # Использование fourcc_code
                )
                if not self.current_writer.isOpened(): # Проверка успешности открытия VideoWriter
                    raise RuntimeError(f"Не удалось открыть VideoWriter для {path} с кодеком MJPG ({fourcc_code}). Проверьте FFmpeg и его кодеки.")

                logger.info(f"Начат новый видео-чанк: {path} с FPS {self.fps}, размером {self.frame_size}, кодеком MJPG.")
            except Exception as e: # Обработка ошибок при инициализации
                logger.error(f"Ошибка инициализации VideoWriter: {e}")
                self.current_writer = None # Убеждаемся, что writer None в случае ошибки

                # В случае ошибки при создании нового чанка, также устанавливаем событие для него.
                # Это предотвратит вечное ожидание, если prepare_archive_video попытается дождаться несуществующего файла.
                if self.current_chunk_path and self.current_chunk_path in self.chunk_repacked_events:
                    self.chunk_repacked_events[self.current_chunk_path].set()

                return #Выход из функции, если writer не создан
            
        if self.current_writer: # Проверка, что VideoWriter успешно инициализирован перед записью
            self.current_writer.write(frame)
            self.frame_count += 1
        else: # Логирование, если writer отсутствует
            logger.warning("VideoWriter не инициализирован, кадр не записан.")

    def release(self) -> None:
        """
        Закрывает текущий видеофайл, если он открыт.
        """
        if self.current_writer:
            self.current_writer.release()
            logger.info(f"Финальное закрытие видео-чанка: {self.current_chunk_path}") # Логирование финального закрытия

            # Принудительная репаковка финального чанка
            if self.current_chunk_path and self.current_chunk_path.exists():
                old_path = self.current_chunk_path
                new_path = old_path.parent / f"{old_path.stem}_repacked.mp4"


                repack_event_for_old_path = self.chunk_repacked_events.get(old_path)
                if not repack_event_for_old_path:
                    repack_event_for_old_path = asyncio.Event()
                    self.chunk_repacked_events[old_path] = repack_event_for_old_path

                def _repack_sync_task(input_path: Path, output_path: Path, original_path: Path, repack_event: asyncio.Event):
                    logger.info(f"Начало финальной репаковки чанка: {input_path.name} -> {output_path.name}")
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i", str(input_path),
                        "-c", "copy",
                        "-movflags", "faststart",
                        str(output_path)
                    ]
                    try:
                        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        logger.info(f"Успешно перепакован финальный чанк: {input_path.name} -> {output_path.name}")
                        os.remove(input_path)
                        os.rename(output_path, original_path)
                        logger.info(f"Оригинальный финальный чанк заменен перепакованным: {original_path.name}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Ошибка FFmpeg при финальной репаковке чанка '{input_path.name}': {e.returncode}")
                        logger.error(f"STDOUT: {e.stdout}")
                        logger.error(f"STDERR: {e.stderr}")
                        if output_path.exists(): os.remove(output_path)
                    except Exception as e:
                        logger.error(f"Непредвиденная ошибка при финальной репаковке чанка '{input_path.name}': {e}")
                    finally:
                        if output_path.exists(): os.remove(output_path)
                        repack_event.set()

                threading.Thread(target=_repack_sync_task, args=(old_path, new_path, old_path, repack_event_for_old_path), daemon=True).start()
            
        self.current_writer = None # Сброс writer'а
        self.current_chunk_path = None # Сброс пути к чанку