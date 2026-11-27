# node.py

import argparse
import json
import sys
import numpy as np
import cv2
import time
import base64
import requests
import uvicorn
import threading
import io
import asyncio
import re
import atexit
import socket
import subprocess
import tempfile
import os

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Generator, Tuple, List, Union, Deque, TypedDict
from collections import deque
from pyngrok import ngrok, conf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse

from src.video_source import VideoSource
from src.hls_video_source import HLSVideoSource
from src.rtsp_video_source import RTSPVideoSource
from src.stream_refresher import get_fresh_hls_url
from src.detector import YOLODetector
from src.detection_logger import DetectionLogger
from src.video_writer import ChunkedVideoWriter
import config_monitor
from config_monitor import (
    DEFAULT_MODEL_WEIGHTS,
    LOG_DIR,
    OUTPUT_VIDEO_DIR,
    CHUNK_DURATION_SEC,
    TARGET_CLASSES,
    STREAM_PORT_DEFAULT,
    HEARTBEAT_INTERVAL,
    ALERT_COOLDOWN,
    PUBLIC_IP_PROVIDER,
    STATIC_IP_ADDRESS,
    CONNECTION_FILE_PATH,
    USE_TELEGRAM_DISCOVERY,
    USE_FILE_DISCOVERY,
    DISCOVERY_CHECK_INTERVAL,
    TRACK_HISTORY_LENGTH,
    ARCHIVE_DOWNLOAD_PRE_EVENT_SEC,
    ARCHIVE_DOWNLOAD_POST_EVENT_SEC,
    DATA_COLLECTION_FRAME_INTERVAL,
    NODE_AUTH_KEY # Импортируем ключ аутентификации Ноды
)
from src.runtime_logger import get_runtime_logger
from src.public_address_manager import PublicAddressManager
from src.schemas import AlertPayload

logger = get_runtime_logger("Node")

def load_cameras() -> list[dict]:
    """Загружает камеры из configs/stream_sources.json."""
    with open("configs/stream_sources.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cameras"]

def select_camera(camera_id: str) -> Optional[dict]:
    cameras = load_cameras()
    for cam in cameras:
        if cam["id"] == camera_id:
            # Убеждаемся, что source_type существует, иначе по умолчанию HLS
            if "source_type" not in cam:
                cam["source_type"] = "HLS"
            return cam
    return None

def create_video_source(page_url: str, source_type: str, loop: bool = False, debug: bool = False):
    """Создаёт источник видео (HLS, RTSP или файл).

    Args:
        page_url: URL или путь к источнику.
        source_type: Тип источника ("HLS", "RTSP", "FILE", "CAMERA").
        loop: Зацикливать ли воспроизведение (для FILE).
        debug: Флаг отладки.

    Returns:
        Экземпляр VideoSource, HLSVideoSource или RTSPVideoSource.
    
    Raises:
        RuntimeError: Если не удалось получить HLS-поток или неизвестный тип источника.
    """
    if source_type == "FILE":
        # Для локальных файлов page_url должен быть file:///path/to/file.mp4
        if not page_url.startswith("file://"):
            page_url = f"file://{page_url}"
        src_path = page_url.replace("file://", "")
        return VideoSource("file", src_path, loop=loop)

    elif source_type == "RTSP" or source_type == "RTMP":
        # Для RTSP и RTMP используем наш FFmpeg-обертку, так как FFmpeg умеет читать оба протокола
        return RTSPVideoSource(stream_url=page_url.strip())
    
    elif source_type == "HLS":
        # Очистка URL от лишних пробелов
        page_url = page_url.strip()
        fresh_url = get_fresh_hls_url(camera_page_url=page_url, referer=page_url)
        if not fresh_url:
            raise RuntimeError("Не удалось получить HLS-поток")
        return HLSVideoSource(
            stream_url=fresh_url,
            referer=page_url,
            page_url=page_url
        )

    else:
        raise RuntimeError(f"Неизвестный тип источника '{source_type}' для {page_url}")
    
#############################
# --- НАСТРОЙКИ СТРИМИНГА ---
#############################
stream_app = FastAPI()
frame_lock = threading.Lock()
latest_frame_jpeg = None

# Глобальные переменные для Service Discovery Ноды
current_pu_notifier_url: Optional[str] = None # Актуальный адрес ПУ для отправки
public_node_stream_url: str = "" # Публичный URL видеопотока этой Ноды
last_discovery_check_time: float = 0.0 # Время последней проверки адреса ПУ

# Глобальные переменные для контроля потока MJPEG
ACTUAL_PROCESSING_FPS: float = 0.0 # Фактический FPS обработки в основном цикле
frame_id_counter: int = 0 # Уникальный ID для каждого обработанного кадра

def generate_stream() -> Generator[bytes, None, None]:
    """Генератор кадров для MJPEG."""
    logger.debug("Клиент подключился к видеопотоку")
    last_sent_frame_id = -1 # Отслеживает ID последнего отправленного кадра

    while True:
        current_frame_id = -1
        with frame_lock:
            if latest_frame_jpeg is None:
                time.sleep(0.05) # Небольшая начальная задержка, если кадр еще не готов
                continue
            frame_data = latest_frame_jpeg
            current_frame_id = frame_id_counter ## Получаем текущий уникальный ID кадра
        
        # Если текущий кадр уже был отправлен, ждем новый.
        # Это предотвратит отправку дубликатов и "прогонку" видео.
        if current_frame_id == last_sent_frame_id:
            time.sleep(0.01) # Очень короткая задержка, чтобы не забивать ЦПУ пустым циклом
            continue
        
        # Формат MJPEG (Boundary string обязательна)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        last_sent_frame_id = current_frame_id # Обновляем ID последнего отправленного кадра

        # Пауза основана на реальном FPS обработки, а не на фиксированном значении.
        # Это позволяет стриму адаптироваться к реальной производительности YOLO.
        if ACTUAL_PROCESSING_FPS > 0:
            time.sleep(1 / ACTUAL_PROCESSING_FPS)
        else:
            time.sleep(0.04) # Fallback на 25 FPS, если ACTUAL_PROCESSING_FPS еще не определен

@stream_app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

def start_stream_server(port):
    logger.info(f"[STREAM] Запуск MJPEG сервера на порту {port}...")
    # Запускаем uvicorn в тихом режиме
    uvicorn.run(stream_app, host="0.0.0.0", port=port, log_level="error")
# ---------------------------

################################
# --- ЛОГИКА АРХИВНОГО ВИДЕО ---
################################
def get_chunk_time_range(filepath: Path) -> Optional[Tuple[datetime, datetime]]:
    """
    Парсит имя файла видео-чанка (output_YYYYMMDD_HHMMSS.mp4) и
    возвращает его временной диапазон (начало, конец).
    """
    try:
        filename_stem = filepath.stem # 'output_YYYYMMDD_HHMMSS' or 'output_YYYYMMDD_HHMMSS_repacked'

        # Удаляем суффикс '_repacked', если он есть
        if filename_stem.endswith("_repacked"):
            filename_stem = filename_stem[:-len("_repacked")]
        

        # Теперь filename_stem должен быть в формате 'output_YYYYMMDD_HHMMSS'
        # Извлекаем только часть с датой и временем
        parts = filename_stem.split('_')
        if len(parts) >= 3:
            time_str = f"{parts[-2]}_{parts[-1]}"
        else:
            # Это будет поймано strptime как некорректный формат, если filename_stem не соответствует
            time_str = filename_stem

        start_time = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
        end_time = start_time + timedelta(seconds=CHUNK_DURATION_SEC)
        return start_time, end_time
    except (ValueError, IndexError) as e:
        logger.error(f"[ARCHIVE ERROR] Ошибка парсинга имени файла чанка '{filepath.name}' для извлечения времени. Raw error: {e}")
        return None

def find_relevant_chunks(
    node_output_dir: Path,
    requested_start: datetime,
    requested_end: datetime
) -> List[Path]:
    """
    Находит все видео-чанки в директории, которые перекрываются с
    запрошенным временным диапазоном.
    """
    relevant_chunks: List[Path] = []
    # Сортируем файлы, чтобы FFmpeg склеивал их в правильном порядке
    sorted_files = sorted(node_output_dir.glob("output_*.mp4"))


    logger.debug(f"[ARCHIVE DEBUG] Поиск чанков в {node_output_dir} для диапазона {requested_start} - {requested_end}")
    logger.debug(f"[ARCHIVE DEBUG] Найдено файлов: {len(sorted_files)}")

    for filepath in sorted_files:
        chunk_range = get_chunk_time_range(filepath)
        if chunk_range:
            chunk_start, chunk_end = chunk_range
            # Проверяем, перекрывается ли чанк с запрошенным диапазоном
            if max(requested_start, chunk_start) < min(requested_end, chunk_end):
                relevant_chunks.append(filepath)
    
                logger.debug(f"[ARCHIVE DEBUG] Чанк '{filepath.name}' ({chunk_start}-{chunk_end}) РЕЛЕВАНТЕН.")
            else:
                logger.debug(f"[ARCHIVE DEBUG] Чанк '{filepath.name}' ({chunk_start}-{chunk_end}) НЕ релевантен.")
    logger.debug(f"[ARCHIVE DEBUG] Итого релевантных чанков: {len(relevant_chunks)}")

    return relevant_chunks

async def prepare_archive_video(
    node_id: str,
    alert_timestamp_iso: str,
    pre_event_sec: int,
    post_event_sec: int,
    writer_events_map: Dict[Path, asyncio.Event]
) -> Optional[Path]:
    """
    Подготавливает архивный видеофайл, склеивая и обрезая необходимые чанки.

    Args:
        node_id: ID ноды, для которой запрашивается архив.
        alert_timestamp_iso: ISO-форматированная строка времени тревоги.
        pre_event_sec: Длительность видео до тревоги в секундах.
        post_event_sec: Длительность видео после тревоги в секундах.
        writer_events_map: Словарь, содержащий asyncio.Event для каждого чанка,
                           используемый для синхронизации с VideoWriter.

    Returns:
        Путь к временному файлу MP4, если успешно, иначе None.
    """
    node_output_dir = Path(OUTPUT_VIDEO_DIR) / node_id
    if not node_output_dir.is_dir():
        logger.error(f"[ARCHIVE ERROR] Директория архива для ноды {node_id} не найдена: {node_output_dir}")
        return None

    try:
        alert_time = datetime.fromisoformat(alert_timestamp_iso)
    except ValueError:
        logger.error(f"[ARCHIVE ERROR] Некорректный формат ISO-времени: {alert_timestamp_iso}")
        return None

    requested_start_time = alert_time - timedelta(seconds=pre_event_sec)
    requested_end_time = alert_time + timedelta(seconds=post_event_sec)

    logger.debug(f"[ARCHIVE DEBUG] Запрос архива для {node_id}, время тревоги: {alert_time}")
    logger.debug(f"[ARCHIVE DEBUG] Запрошенный диапазон: {requested_start_time} - {requested_end_time}")

    relevant_chunks = find_relevant_chunks(node_output_dir, requested_start_time, requested_end_time)

    if not relevant_chunks:
        logger.warning(f"[ARCHIVE WARNING] Не найдено релевантных видео-чанков для {node_id} в диапазоне {requested_start_time}-{requested_end_time}")
        return None

    # Глобальные переменные для доступа к текущему VideoWriter и его пути
    # Они должны быть доступны здесь. node_output_dir будет соответствовать output_dir из main.
    # self.current_chunk_path хранится в VideoWriter, но нам нужен его прямой доступ в prepare_archive_video.
    # Для этого передадим его как аргумент, или сделаем более явным.
    # Сейчас VideoWriter - это self.video_writer в main(), а prepare_archive_video - это функция в модуле.
    # ЛУЧШЕЕ РЕШЕНИЕ: current_video_writer_path сделать глобальной в node.py, обновляемой main().

    global video_writer # Доступ к глобальному объекту video_writer

    # Ждем завершения репаковки для всех релевантных чанков
    for chunk_path in relevant_chunks:
        if chunk_path in writer_events_map:
            logger.info(f"[ARCHIVE INFO] Ожидание готовности чанка: {chunk_path.name}")
            try:
                await asyncio.wait_for(writer_events_map[chunk_path].wait(), timeout=CHUNK_DURATION_SEC + 5) # Таймаут на случай проблем
            except asyncio.TimeoutError:
                logger.error(f"[ARCHIVE ERROR] Таймаут ожидания готовности чанка '{chunk_path.name}'. Продолжаем без него.")
            logger.info(f"[ARCHIVE INFO] Чанк {chunk_path.name} готов.")
        elif not chunk_path.exists():
            logger.warning(f"[ARCHIVE WARNING] Релевантный чанк '{chunk_path.name}' не существует на диске. Пропускаем его.")

    # Создаем временный файл для списка FFmpeg concat
    list_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.txt') as f:
            list_file_path = Path(f.name)
            for chunk_path in relevant_chunks:
                # Только добавляем чанки, которые реально существуют после ожидания
                if chunk_path.exists():
                    f.write(f"file '{chunk_path.resolve()}'\n")
                else:
                    logger.warning(f"[ARCHIVE WARNING] Чанк '{chunk_path.name}' не существует, пропущен при создании списка для FFmpeg.")

        list_file = list_file_path
        
        # Создаем временный файл для выходного видео
        # NamedTemporaryFile с delete=False оставляет файл после закрытия,
        # что необходимо для FileResponse. Удаление будет в cleanup_temp_file
        temp_output_video_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
        
        # FFmpeg команда для склейки и нарезки
        # -safe 0 позволяет использовать абсолютные пути в списке concat
        # -ss определяет начальное смещение от начала первого чанка
        # -to определяет конечное смещение от начала первого чанка
        
        # Рассчитываем смещение относительно начала первого релевантного чанка
        first_chunk_start_time, _ = get_chunk_time_range(relevant_chunks[0]) # type: ignore
        # Смещение начала запрошенного фрагмента от начала первого чанка
        start_offset_sec = max(0, (requested_start_time - first_chunk_start_time).total_seconds()) 
        # Общая длительность запрошенного фрагмента
        duration_sec = (requested_end_time - requested_start_time).total_seconds()
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-y", # Перезаписывать выходной файл без запроса
            "-f", "concat",
            "-safe", "0", # Разрешить небезопасные пути (абсолютные в файле списка)
            "-i", str(list_file_path),
            "-ss", str(start_offset_sec), # Начать с этого смещения
            "-t", str(duration_sec),     # Записать такую длительность
            "-c", "copy", # Копировать потоки без перекодирования (быстро!)
            str(temp_output_video_path)
        ]

        logger.info(f"[ARCHIVE INFO] Запуск FFmpeg для архива: {' '.join(ffmpeg_cmd)}")
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"[ARCHIVE ERROR] FFmpeg для архива завершился с ошибкой {process.returncode}:")
            logger.error(f"[ARCHIVE ERROR] STDOUT: {stdout.decode()}")
            logger.error(f"[ARCHIVE ERROR] STDERR: {stderr.decode()}")

            if temp_output_video_path.exists():
                os.remove(temp_output_video_path)
            return None
        
        # Регистрируем удаление временного файла после его отдачи
        atexit.register(lambda: os.remove(temp_output_video_path) if temp_output_video_path.exists() else None)
        logger.info(f"[ARCHIVE INFO] Архивное видео успешно подготовлено: {temp_output_video_path.name}")
        return temp_output_video_path

    except Exception as e:
        logger.error(f"[ARCHIVE ERROR] Непредвиденная ошибка при подготовке архивного видео для {node_id}: {e}")
###         logger.error(f"Ошибка при подготовке архивного видео для {node_id}: {e}")
        return None
    finally:
        if list_file and list_file.exists():
            os.remove(list_file) # Удаляем временный файл списка
            
# Добавляем новый эндпоинт FastAPI
@stream_app.get("/archive_video/{node_id}/{timestamp_iso}")
async def get_archive_video(
    node_id: str,
    timestamp_iso: str,
    pre_event_sec: int = config_monitor.ARCHIVE_DOWNLOAD_PRE_EVENT_SEC,
    post_event_sec: int = config_monitor.ARCHIVE_DOWNLOAD_POST_EVENT_SEC
) -> Any:
    """
    Отдает архивный видеофайл, соответствующий моменту тревоги.
    
    Args:
        node_id: ID ноды, для которой запрашивается архив.
        timestamp_iso: ISO-форматированная строка времени тревоги (например, из AlertPayload).
        pre_event_sec: Сколько секунд видео до момента тревоги скачивать.
        post_event_sec: Сколько секунд видео после момента тревоги скачивать.

    Returns:
        FileResponse с видеофайлом или 404/500 ошибкой.
    """
    logger.info(f"Запрос архивного видео для ноды {node_id} @ {timestamp_iso}, pre:{pre_event_sec}, post:{post_event_sec}")

    temp_video_path = await prepare_archive_video(
        node_id,
        timestamp_iso,
        pre_event_sec,
        post_event_sec,
        video_writer.chunk_repacked_events
    )

    if temp_video_path and temp_video_path.exists():
        response = FileResponse(
            temp_video_path,
            media_type="video/mp4",
            filename=f"archive_{node_id}_{timestamp_iso.replace(':', '-')}.mp4",
            background=None # Отключить background task, если atexit удаляет файл
        )
        # atexit уже зарегистрирован для удаления файла после завершения процесса.
        return response
    else:
        # Если видео не найдено или произошла ошибка

        from fastapi.responses import PlainTextResponse
        logger.error(f"Не удалось подготовить архивное видео для {node_id} @ {timestamp_iso}.")

        return PlainTextResponse(
            content="Archive video not found or could not be prepared.",
            status_code=404,
            headers={"X-Error-Message": "Archive video not found or could not be prepared."}
        )
# ---------------------------

async def get_pu_address() -> Optional[str]:
    """
    Пытается получить адрес ПУ из Telegram (приоритет) или из локального файла.
    Возвращает (timestamp, url) или None.
    """
    best_url: Optional[str] = None
    best_timestamp: float = 0.0

    # 1. Попытка чтения из Telegram (приоритет)
    if config_monitor.USE_TELEGRAM_DISCOVERY and config_monitor.TELEGRAM_BOT_TOKEN and config_monitor.TELEGRAM_CHANNEL_ID:
        try:
            bot_token = config_monitor.TELEGRAM_BOT_TOKEN
            channel_id = config_monitor.TELEGRAM_CHANNEL_ID
            
            response = requests.get(
                f"https://api.telegram.org/bot{bot_token}/getChat?chat_id={channel_id}",
                timeout=5
            )
            response.raise_for_status()
            chat_info = response.json().get("result", {})
            
            pinned_message = chat_info.get("pinned_message", {})
            message_text = pinned_message.get("text", "")
            
            # Парсим строку "SYSTEM_URL: TIMESTAMP,URL"
            match = re.match(r"SYSTEM_URL: (\d+),(https?://.+)", message_text)
            if match:
                timestamp_str, url = match.groups()
                timestamp = float(timestamp_str)
                if timestamp > best_timestamp:
                    best_timestamp = timestamp
                    best_url = url
                    logger.info(f"Обнаружен адрес ПУ в Telegram (Timestamp: {timestamp})")
            else:
                logger.debug("Закрепленное сообщение в Telegram не содержит SYSTEM_URL или неверный формат.")

        except Exception as e:
            logger.warning(f"Ошибка чтения адреса ПУ из Telegram: {e}")

    # 2. Попытка чтения из локального файла
    if config_monitor.USE_FILE_DISCOVERY and config_monitor.CONNECTION_FILE_PATH.exists():
        try:
            with open(config_monitor.CONNECTION_FILE_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            match = re.match(r"(\d+),(https?://.+)", content)
            if match:
                timestamp_str, url = match.groups()
                timestamp = float(timestamp_str)
                if timestamp > best_timestamp: # Сравниваем, чтобы взять более свежий
                    best_timestamp = timestamp
                    best_url = url
                    logger.info(f"Обнаружен адрес ПУ в файле {config_monitor.CONNECTION_FILE_PATH} (Timestamp: {timestamp})")
            else:
                logger.warning(f"Файл {config_monitor.CONNECTION_FILE_PATH} не содержит URL ПУ или неверный формат.")

        except Exception as e:
            logger.warning(f"Ошибка чтения адреса ПУ из файла {config_monitor.CONNECTION_FILE_PATH}: {e}")

    return best_url

def _get_iou(box1: List[float], box2: List[float]) -> float:
    """Вычисляет IoU (Intersection over Union) между двумя bounding box'ами."""
    # box format: [x_center, y_center, width, height]
    # Преобразуем в [x_min, y_min, x_max, y_max]
    
    def convert_to_xyxy(box):
        x_c, y_c, w, h = box
        w_half, h_half = w / 2, h / 2
        x_min = x_c - w_half
        y_min = y_c - h_half
        x_max = x_c + w_half
        y_max = y_c + h_half
        return [x_min, y_min, x_max, y_max]

    box1_xyxy = convert_to_xyxy(box1)
    box2_xyxy = convert_to_xyxy(box2)

    x1_min, y1_min, x1_max, y1_max = box1_xyxy
    x2_min, y2_min, x2_max, y2_max = box2_xyxy

    # Вычисляем координаты пересечения
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Вычисляем площадь пересечения
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height

    # Вычисляем площади bounding box'ов
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Вычисляем площадь объединения
    union_area = box1_area + box2_area - intersection_area

    # Вычисляем IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def _send_request_in_thread(url: str, json_data: dict, headers: dict, event_type: str) -> None:
    """Вспомогательная функция для отправки HTTP-запроса в отдельном потоке."""
    try:
        requests.post(url, json=json_data, headers=headers, timeout=2)
        logger.debug(f"Данные ({event_type}) успешно отправлены на {url} (из потока).")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка отправки данных ({event_type}) на ПУ {url} (из потока): {e}")
        # При ошибке сбрасываем время проверки, чтобы быстрее перепроверить адрес ПУ
        global last_discovery_check_time
        last_discovery_check_time = 0.0
    except Exception as e:
        logger.error(f"[ERROR] Не удалось отправить данные ({event_type}) (из потока): {e}")

def send_data_to_server(
    node_id: str,
    frame: np.ndarray,
    event_type: str,
    box=None,
    class_name: Optional[str] = None,
    confidence: float = 0.0,
    object_id: Optional[int] = None,
    velocity: Optional[Tuple[float, float]] = None,
    prediction: Optional[List[Tuple[float, float]]] = None,
    prediction_iou_score: Optional[float] = None,
    extra_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Формирует и отправляет данные (Alert с изображением (Base64) или Heartbeat).
    """
    global current_pu_notifier_url

    if not current_pu_notifier_url:
        logger.warning(f"Адрес ПУ неизвестен. Данные ({event_type}) не отправлены.")
        return

    try:
        # 1. Кодируем кадр в JPEG (качество завист от типа события)
        quality = 70 if event_type == "alert" else 40
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        # 2. Переводим байты в строку Base64
        b64_image = base64.b64encode(buffer).decode("utf-8")

        # 3. Формируем данные (Payload)
        bbox_list = []
        if box is not None:
            bbox_list = [int(x) for x in box.xyxy[0]]

        payload = {
            "node_id": node_id,
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "object_id": object_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox_list,
            "velocity": velocity,
            "prediction": prediction,
            "prediction_iou_score": prediction_iou_score,
            "image_base64": b64_image
        }

        # Добавляем метаданные, если есть (имя камеры, модель и т.д.)
        if extra_info:
            payload.update(extra_info)

        # Дебаг отправки BBox
        if event_type == "alert":
            logger.info(f"[NODE SEND] Отправка Alert. BBox: {bbox_list}")


        # 4. Формируем заголовки
        headers = {}
        if config_monitor.NODE_AUTH_KEY:
            headers["X-API-Key"] = config_monitor.NODE_AUTH_KEY
        else:
            logger.warning("NODE_AUTH_KEY не установлен в .env. Отправка данных на ПУ без авторизации.")
            
        # 5. Запускаем отправку в отдельном потоке
        send_thread = threading.Thread(
            target=_send_request_in_thread,
            args=(current_pu_notifier_url, payload, headers, event_type),
            daemon=True
        )
        send_thread.start()
        logger.debug(f"Поток отправки данных ({event_type}) запущен.")

    except Exception as e:
        # Логируем ошибку в консоль, но не останавливаем работу ноды
        logger.error(f"[ERROR] Не удалось сформировать и запустить отправку алерта: {e}")

# Определение TypedDict для структуры данных трека
class TrackData(TypedDict):
    centers: Deque[Tuple[float, float]]
    predictions: Deque[List[Tuple[float, float]]]
    prediction_for_current_frame_bbox_norm: Optional[List[float]]

# Глобальная переменная для хранения истории треков
track_history: Dict[Any, TrackData] = {}

def main():
    parser = argparse.ArgumentParser(description="Drone Detection Node")
    parser.add_argument("--list", action="store_true", help="List available cameras")
    parser.add_argument("--select", type=str, help="Start node for camera ID")
    parser.add_argument("--debug", action="store_true", help="Show video window (OpenCV)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_WEIGHTS, help="YOLO model path")
    parser.add_argument("--stream_port", type=int, default=STREAM_PORT_DEFAULT, help="Port for MJPEG stream")
    parser.add_argument("--public_url", type=str, default="", help="Public URL (ngrok) for this node")
    parser.add_argument("--collect_data", nargs='*', help="Save frames and labels for dataset augmentation. Specify classes to collect (e.g., 'helicopter'). If no classes specified, defaults to 'drone' and 'drones'.")    
    args = parser.parse_args()

    global latest_frame_jpeg, current_pu_notifier_url, last_discovery_check_time
    global ACTUAL_PROCESSING_FPS, frame_id_counter
    global video_writer
    # global track_history
    
    logger.info(f"Настройки: PUBLIC_IP_PROVIDER для Ноды - {config_monitor.PUBLIC_IP_PROVIDER}")

    # Создаем экземпляр менеджера публичных адресов для Ноды
    node_public_address_manager = PublicAddressManager(
        service_port=args.stream_port,
        public_url_override=args.public_url,
        service_name="Node"
    )

    # Запускаем инициализацию публичного адреса Ноды (асинхронно в потоке)
    def run_node_public_address_init():
        asyncio.run(node_public_address_manager.initialize_public_address())
        global public_node_stream_url
        # Добавляем "/video_feed" к базовому URL, полученному от менеджера
        public_node_stream_url = node_public_address_manager.get_public_url() + "/video_feed"
        logger.info(f"Публичный URL видеопотока Ноды установлен: {public_node_stream_url}")

    node_public_address_thread = threading.Thread(target=run_node_public_address_init, daemon=True)
    node_public_address_thread.start()

    # Получаем первый адрес ПУ (синхронно при старте)
    # Используем asyncio.run() для вызова асинхронной функции в синхронном контексте main()
    current_pu_notifier_url = asyncio.run(get_pu_address())
    if current_pu_notifier_url:
        logger.info(f"Первичный адрес ПУ получен: {current_pu_notifier_url}")
    else:
        logger.warning(f"Не удалось получить первичный адрес ПУ. Нода не сможет отправлять алерты, пока адрес не будет найден. current_pu_notifier_url = None")
        current_pu_notifier_url = None # Явно устанавливаем None, чтобы не использовать старый NOTIFIER_URL

    last_discovery_check_time = time.time()
    
    if args.list:
        cameras = load_cameras()
        logger.info("Available cameras:")
        for cam in cameras:
            logger.info(f"  {cam['id']}: {cam['name']}")
        return

    if not args.select:
        logger.info("Use --list to see cameras, then --select <id> to run a node.")
        sys.exit(1)

    camera = select_camera(args.select)
    if not camera:
        logger.warning(f"Camera ID '{args.select}' not found.")
        sys.exit(1)

    node_id = camera["id"]
    page_url = camera["page_url"]
    source_type = camera["source_type"]
    loop_video = camera.get("loop", True) # По умолчанию True для удобства тестирования файлов

    # Настройка путей
    log_dir = Path(LOG_DIR) / node_id
    output_dir = Path(OUTPUT_VIDEO_DIR) / node_id
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting node for camera: {camera['name']} (ID: {node_id})")
    logger.info(f"Log dir: {log_dir}")
    logger.info(f"Output dir: {output_dir}")

    # Подготовка директорий для сбора датасета
    aug_images_dir = Path("video_augmentations/images")
    aug_labels_dir = Path("video_augmentations/labels")
    if args.collect_data:
        aug_images_dir.mkdir(parents=True, exist_ok=True)
        aug_labels_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Режим сбора данных ВКЛЮЧЕН. Сохранение в: video_augmentations/")

    # Инициализация
    detector = YOLODetector(args.model)
    video_source = create_video_source(page_url, source_type, loop=loop_video, debug=args.debug)
    csv_logger = DetectionLogger(log_dir)
    csv_logger.start_new_session()


    # Переменные для сбора датасета
    last_data_saved_frame = -999 # Чтобы первый кадр (0) мог быть сохранен
    data_collection_enabled = args.collect_data is not None
    
    # Пытаемся получить общее кол-во кадров, если источник поддерживает это (VideoSource)
    # Для HLS/RTSP это будет атрибут отсутствовать или равен -1/0
    source_total_frames = getattr(video_source, 'total_frames', -1)

    # Определяем список классов для сбора данных
    classes_to_collect = {"drone", "drones"} # Всегда собираем "drone" и "drones"
    if data_collection_enabled and args.collect_data:
        # Добавляем все классы, указанные в командной строке, переводя их в нижний регистр
        classes_to_collect.update(c.lower() for c in args.collect_data)
    classes_to_collect_list = list(classes_to_collect) # Для удобства логирования

    if data_collection_enabled:
        logger.info(f"Режим сбора данных ВКЛЮЧЕН. Сохранение в: video_augmentations/ Только для классов: {classes_to_collect_list}")

    all_model_classes_names = list(detector.model.names.values())

    # Устанавливаем начальное значение ACTUAL_PROCESSING_FPS равным FPS источника
    # Если YOLO будет медленнее, это значение будет скорректировано
    ACTUAL_PROCESSING_FPS = video_source.fps
    logger.info(f"Изначальный FPS обработки/стрима установлен на: {ACTUAL_PROCESSING_FPS:.2f}")

    frame = video_source.read_frame()
    if frame is None:
        raise RuntimeError("Failed to read first frame")

    h, w = frame.shape[:2]
    video_writer = ChunkedVideoWriter(output_dir, video_source.fps, (w, h), CHUNK_DURATION_SEC)

    # Время последнего отправленного алерта и хартбита
    last_alert_time = 0
    last_heartbeat_time = 0

    ### # Хранилище треков: {track_id: {'centers': deque, 'predictions': deque}}
    ### track_history: Dict[Any, Dict[str, Union[deque, Optional[List[float]]]]] = {}
    
    # Запуск стрим-сервера в фоне
    t_stream = threading.Thread(target=start_stream_server, args=(args.stream_port,), daemon=True)
    t_stream.start()
    logger.info(f"Потоковое видео доступно по адресу: http://localhost:{args.stream_port}/video_feed")
    logger.info(f"Эндпоинт для скачивания архивов: http://localhost:{args.stream_port}/archive_video/{node_id}/<timestamp_iso>")

    # Формируем красивое имя модели (один раз)
    model_name_display = Path(args.model).name
    if "yolo" in model_name_display.lower() and "custom" not in model_name_display.lower():
         model_name_display = f"Standard ({model_name_display})"

    frame_count = 0
    
    # Переменные для измерения фактического FPS обработки в главном цикле
    processing_start_time_interval = time.time()
    frames_processed_in_interval = 0

    try:
        while True:
            iteration_start_time = time.time() ## Начало измерения времени итерации

            orig, annotated, boxes = detector.predict_and_track(frame)
            current_time = time.time()

            # ОЧИСТКА СТАРЫХ ТРЕКОВ (УМНАЯ, КАЖДЫЙ КАДР)
            current_ids = set()
            if boxes is not None and boxes.id is not None:
                current_ids = set(int(x) for x in boxes.id.tolist())
            
            # Удаляем из истории ВСЕ данные для ID, которых нет в current_ids
            keys_to_remove = [k for k in track_history.keys() if k not in current_ids]
            for k in keys_to_remove:
                del track_history[k]

            # --- СБОР ДАТАСЕТА (Active Learning) ---
            # Логика:
            # 1. Если это файл и мы прошли его длину - выключаем сбор (защита от дублей в loop)
            # 2. Проверяем интервал (10 кадров от ПОСЛЕДНЕГО СОХРАНЕННОГО)
            # 3. Проверяем наличие детекций

            if data_collection_enabled:
                # Проверка на конец файла (для защиты от loop)
                if source_total_frames > 0 and frame_count >= source_total_frames:
                    logger.info("Достигнут конец видеофайла. Сбор данных для датасета остановлен (во избежание дублей).")
                    data_collection_enabled = False
                
                # Умное прореживание: сохраняем, только если прошло >= 10 кадров с прошлого сохранения
                elif (frame_count - last_data_saved_frame >= DATA_COLLECTION_FRAME_INTERVAL) and (boxes is not None) and len(boxes) > 0:
                    try:
                        last_data_saved_frame = frame_count
                        
                        # Базовое имя файла: node_id_000123
                        base_filename = f"{node_id}_{frame_count:06d}"
                        
                        # --- 1. Сохранение ОРИГИНАЛА ---
                        img_path = aug_images_dir / f"{base_filename}.jpg"
                        lbl_path = aug_labels_dir / f"{base_filename}.txt"
                        
                        cv2.imwrite(str(img_path), orig)
                        
                        with open(lbl_path, "w") as f:
                            for box in boxes:

                                # Получаем имя класса для текущего бокса и приводим к нижнему регистру
                                detected_class_name = all_model_classes_names[int(box.cls[0])].lower()

                                # Сохраняем только если класс есть в списке classes_to_collect И уверенность выше порога
                                if detected_class_name in classes_to_collect and box.conf[0] > 0.3:
                                    # xywhn возвращает нормализованные [x_center, y_center, width, height]
                                    x, y, w_box, h_box = box.xywhn[0].tolist()
                                    # Класс всегда 0 (для последующей разметки)
                                    f.write(f"0 {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

                        # --- 2. Сохранение ЗЕРКАЛЬНОГО (Mirror) ---
                        mirror_filename = f"mirror_{base_filename}"
                        img_mirror_path = aug_images_dir / f"{mirror_filename}.jpg"
                        lbl_mirror_path = aug_labels_dir / f"{mirror_filename}.txt"
                        
                        # Отражаем изображение по горизонтали (код 1)
                        orig_mirror = cv2.flip(orig, 1)
                        cv2.imwrite(str(img_mirror_path), orig_mirror)
                        
                        # Пересчитываем метки для отраженного кадра
                        with open(lbl_mirror_path, "w") as f:
                            for box in boxes:

                                # Получаем имя класса для текущего бокса и приводим к нижнему регистру
                                detected_class_name = all_model_classes_names[int(box.cls[0])].lower()

                                # Сохраняем только если класс есть в списке classes_to_collect И уверенность выше порога
                                if detected_class_name in classes_to_collect and box.conf[0] > 0.3:
                                    x, y, w_box, h_box = box.xywhn[0].tolist()
                                    # При отражении по горизонтали: новый X = 1.0 - старый X
                                    x_mirror = 1.0 - x
                                    f.write(f"0 {x_mirror:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")
                                
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении данных для датасета: {e}")

            # 1. Отрисовка треков (хвостов)
            if boxes is not None and boxes.id is not None:
                # Получаем ID треков и координаты центров
                track_ids = [int(x) for x in boxes.id.tolist()]

                for box, track_id in zip(boxes, track_ids): # type: ignore
                    x1, y1, x2, y2 = box.xyxy[0]
                    center = (float((x1 + x2) / 2), float((y1 + y2) / 2))

                    if track_id not in track_history:

                        track_history[track_id] = TrackData(
                            centers=deque[Tuple[float, float]](maxlen=TRACK_HISTORY_LENGTH),
                            predictions=deque[List[Tuple[float, float]]](maxlen=3),
                            prediction_for_current_frame_bbox_norm=None
                        )
#                        track_history[track_id] = {
#                            'centers': deque(maxlen=TRACK_HISTORY_LENGTH),
#                            'predictions': deque(maxlen=3),
#                            'prediction_for_current_frame_bbox_norm': Optional[List[float]]
#                        }
                    track_history[track_id]['centers'].append(center)

                    pts = list(track_history[track_id]['centers'])

                    # Динамическое формирование градиента для всей длины хвоста
                    # От красного (новый) к серому (старый)
                    
                    # Начальный цвет (RGB): Красный
                    start_color_rgb = (255, 0, 0)
                    # Конечный цвет (RGB): Темно-серый
                    end_color_rgb = (100, 100, 100)
                    
                    # Преобразуем в BGR для OpenCV
                    start_color_bgr = (start_color_rgb[2], start_color_rgb[1], start_color_rgb[0])
                    end_color_bgr = (end_color_rgb[2], end_color_rgb[1], end_color_rgb[0])

                    # Рисуем сегменты от последнего (нового) к первому (старому)
                    for i in range(1, len(pts)):
                        # Вычисляем коэффициент градиента (от 0 до 1)
                        # Чем ближе к началу списка (т.е. старее), тем ближе коэффициент к 1.0
                        gradient_factor = i / len(pts) 
                        
                        # Интерполируем цвета
                        # (1 - factor) * start_color + factor * end_color
                        current_color = (
                            int((1 - gradient_factor) * start_color_bgr[0] + gradient_factor * end_color_bgr[0]),
                            int((1 - gradient_factor) * start_color_bgr[1] + gradient_factor * end_color_bgr[1]),
                            int((1 - gradient_factor) * start_color_bgr[2] + gradient_factor * end_color_bgr[2])
                        )
                        
                        ptA = pts[-i]
                        ptB = pts[-(i+1)]
                        
                        cv2.line(annotated, (int(ptA[0]), int(ptA[1])), (int(ptB[0]), int(ptB[1])), current_color, 3)

            # Фильтрация: только "drone" (перенесено сюда)
            drone_boxes = []
            names_list = [name for _, name in sorted(detector.model.names.items())]
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if names_list[cls_id].lower() in TARGET_CLASSES:
                        drone_boxes.append(box)

            # Расчет velocity и prediction для Payload
            current_velocity: Optional[Tuple[float, float]] = None
            current_prediction: Optional[List[Tuple[float, float]]] = None
            first_drone_track_id: Optional[int] = None
            calculated_iou_score: Optional[float] = None

            if boxes is not None and boxes.id is not None and drone_boxes: # Только если есть дроны и треки
                # Берем первый дрон для расчета velocity/prediction в Payload
                first_drone_track_id = int(drone_boxes[0].id[0])
                if first_drone_track_id is not None and first_drone_track_id in track_history:
                    history_points = list(track_history[first_drone_track_id]['centers'])
                    if len(history_points) >= 2:
                        
                        # Вычисление скорости: усредняем по последним 2-3 точкам
                        # Более сложная логика может использовать несколько точек для
                        vx_sum, vy_sum = 0.0, 0.0
                        count = 0
                        # Берем последние 3 точки (или меньше, если история
                        for i in range(max(1, len(history_points) - 3), len(history_points)):
                            p1 = history_points[i - 1]
                            p2 = history_points[i]
                            vx_sum += (p2[0] - p1[0])
                            vy_sum += (p2[1] - p1[1])
                            count += 1
                        if count > 0:
                            current_velocity = (vx_sum / count, vy_sum / count)
                        
                        # Прогноз: простая линейная экстраполяция на 0.5 секунды (15 кадров)
                        if current_velocity and len(history_points) > 0:
                            current_prediction = []
                            last_pos = history_points[-1]
                            for _ in range(15): # Прогноз на 15 кадров вперед
                                last_pos = (last_pos[0] + current_velocity[0], last_pos[1] + current_velocity[1])
                                current_prediction.append(last_pos)

                    # Расчет IoU для точности предсказания на текущем кадре
                    target_box_xywhn = drone_boxes[0].xywhn[0].tolist() # Текущая детекция (xc,yc,w,h) в нормированных координатах
                    
                    # Прогноз на текущий кадр был сделан в ПРЕДЫДУЩЕМ кадре
                    # prev_frame_predicted_bbox_for_this_frame = track_history[first_drone_track_id].get('prediction_for_current_frame_bbox_norm')
                    prev_frame_predicted_bbox_for_this_frame = track_history[first_drone_track_id]['prediction_for_current_frame_bbox_norm'] 
                    
                    if prev_frame_predicted_bbox_for_this_frame is not None:
                        # Сравниваем текущую детекцию с прогнозом, сделанным для этого кадра
                        calculated_iou_score = _get_iou(target_box_xywhn, prev_frame_predicted_bbox_for_this_frame)
                    
                    # Сохраняем текущий прогноз для следующего кадра
                    # prediction_for_current_frame_bbox_norm будет хранить прогноз на СЛЕДУЮЩИЙ кадр
                    if current_prediction:
                        # Берем ПЕРВУЮ точку текущего прогноза (это прогноз на следующий кадр)
                        predicted_next_center_px = current_prediction[0] # (x, y) в пикселях

                        # Используем размер текущего bounding box'а для прогнозируемого
                        _, _, w_norm, h_norm = target_box_xywhn

                        # Нормализуем предсказанный центр, так как _get_iou работает с нормализованными координатами
                        predicted_next_center_norm_x = predicted_next_center_px[0] / w
                        predicted_next_center_norm_y = predicted_next_center_px[1] / h

                        track_history[first_drone_track_id]['prediction_for_current_frame_bbox_norm'] = [
                            predicted_next_center_norm_x, predicted_next_center_norm_y, w_norm, h_norm
                        ]
                    else:
                        track_history[first_drone_track_id]['prediction_for_current_frame_bbox_norm'] = None

                    # Сохраняем текущий прогноз в историю для отрисовки старых/новых прогнозов
                    if current_prediction:
                        track_history[first_drone_track_id]['predictions'].append(current_prediction)

            # ВИЗУАЛИЗАЦИЯ ПРОГНОЗОВ (старых и новых)
            for track_id, history_data in track_history.items():
                if 'predictions' in history_data and len(history_data['predictions']) > 0:
                    all_predictions = list(history_data['predictions'])

                    # Цвета для прогнозов (от старого к новому)
                    # base_blue - самый старый, light_blue - самый новый
                    base_blue = (150, 0, 0) # BGR: Темно-синий
                    light_blue = (255, 150, 0) # BGR: Светло-голубой
                    
                    for pred_idx, prediction_line in enumerate(all_predictions):
                        # Вычисляем цвет для текущего прогноза
                        # pred_idx=0 (самый старый), pred_idx=len-1 (самый новый)
                        color_factor = pred_idx / (len(all_predictions) - 1) if len(all_predictions) > 1 else 0
                        current_pred_color = (
                            int(base_blue[0] * (1 - color_factor) + light_blue[0] * color_factor),
                            int(base_blue[1] * (1 - color_factor) + light_blue[1] * color_factor),
                            int(base_blue[2] * (1 - color_factor) + light_blue[2] * color_factor)
                        )

                        for i in range(1, len(prediction_line)):
                            p1 = prediction_line[i-1]
                            p2 = prediction_line[i]
                            cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), current_pred_color, 2)
                            if i == len(prediction_line) - 1: # Рисуем точку в конце прогноза
                                cv2.circle(annotated, (int(p2[0]), int(p2[1])), 4, current_pred_color, -1)

            video_writer.write(annotated.copy()) # Передаем копию кадра

            # Обновляем кадр для стрима (сжимаем в JPEG для скорости)
            try:
                _, buf = cv2.imencode('.jpg', annotated.copy(), [int(cv2.IMWRITE_JPEG_QUALITY), 60]) # Передаем копию кадра
                with frame_lock:
                    latest_frame_jpeg = buf.tobytes()
                    frame_id_counter += 1

                if frame_count % 100 == 0:
                    logger.debug(f"[DEBUG] Кадр для стрима обновлен. Размер: {len(latest_frame_jpeg)} байт")
            except Exception as e:
                logger.error(f"[ERROR] Ошибка кодирования стрима: {e}")
                pass

            if drone_boxes:
                logger.info(f"[ВНИМАНИЕ!] Обнаружен подозрительный объект! Окно {frame_count} содержит объектов: {len(drone_boxes)}")
                # Защита от флуда: отправляем не чаще 1 раза в 5 секунд
                if current_time - last_alert_time > ALERT_COOLDOWN:

                    # Берем первый попавшийся дрон для отчета
                    target_box = drone_boxes[0]
                    target_cls_id = int(target_box.cls[0])
                    target_name = names_list[target_cls_id]
                    
                    # Вычисляем URL стрима
                    s_url = args.public_url if args.public_url else f"http://localhost:{args.stream_port}/video_feed"

                    # Формируем метаданные для алерта
                    meta = {
                        "camera_name": camera["name"],
                        "model_name": model_name_display,
                        "source_type": source_type,
                        "stream_url": s_url,
                        "all_classes_names": all_model_classes_names,
                        "target_classes": TARGET_CLASSES
                    }

                    # Отправляем аннотированный кадр (с рамкой) И метаданными
                    send_data_to_server(
                        node_id,
                        annotated,
                        "alert",
                        target_box,
                        target_name,
                        float(target_box.conf[0]),
                        object_id=first_drone_track_id,
                        velocity=current_velocity,
                        prediction=current_prediction,
                        prediction_iou_score=calculated_iou_score,
                        extra_info=meta
                    )

                    last_alert_time = current_time
                    last_heartbeat_time = current_time
                    logger.info(f"[INFO] Alert sent to {current_pu_notifier_url}")

            # Логирование всех объектов
            csv_logger.log(boxes, names_list)

            # Логика HEARTBEAT (Сигнал жизни)
            if current_time - last_heartbeat_time > HEARTBEAT_INTERVAL:

                # Шлем чистый кадр (orig) и  МЕТАДАННЫЕ, чтобы оператор видел обстановку
                # Определяем URL стрима (если передан публичный, иначе локальный)
                s_url = public_node_stream_url

                ### logger.debug(f"[СТАТУС] Heartbeat отправлен. URL видеопотока: {s_url}")

                meta = {
                    "camera_name": camera["name"],
                    "model_name": model_name_display,
                    "source_type": source_type,
                    "stream_url": s_url,
                    "all_classes_names": all_model_classes_names,
                    "target_classes": TARGET_CLASSES
                }

                send_data_to_server(node_id, orig, "heartbeat", object_id=None, velocity=None, prediction=None, prediction_iou_score=None, extra_info=meta)

                last_heartbeat_time = current_time

            # 3. Периодическая проверка адреса ПУ
            if current_time - last_discovery_check_time > config_monitor.DISCOVERY_CHECK_INTERVAL:
                logger.info("Запуск проверки адреса ПУ...")
                new_pu_url = asyncio.run(get_pu_address())
                if new_pu_url and new_pu_url != current_pu_notifier_url:
                    logger.info(f"Адрес ПУ обновлен: {new_pu_url}")
                    current_pu_notifier_url = new_pu_url
                elif not new_pu_url:
                    logger.warning("Не удалось получить адрес ПУ. Будет использован текущий.")
                last_discovery_check_time = current_time

            # Измерение фактического FPS обработки в главном цикле
            frames_processed_in_interval += 1
            if current_time - processing_start_time_interval >= 1.0: # Обновляем FPS каждую секунду
                ACTUAL_PROCESSING_FPS = frames_processed_in_interval / (current_time - processing_start_time_interval)
                logger.info(f"Текущий фактический FPS обработки YOLO: {ACTUAL_PROCESSING_FPS:.2f}")
                processing_start_time_interval = current_time
                frames_processed_in_interval = 0

            # if args.debug:
            if args.debug and frame_count % 1 == 0:
                # Ограничение до ~10 FPS для плавности
                target_delay = 1.0 / min(video_source.fps, 10)  # максимум 10 FPS
                time.sleep(target_delay)
                
                cv2.imshow(f"Node {node_id} - Detection", annotated)
                if cv2.waitKey(1) == ord('q'):
                    break

            # Задержка в основном цикле, чтобы не обрабатывать кадры быстрее,
            # чем они поступают от HLS-источника, если YOLO достаточно быстр.
            # Если YOLO медленный (actual_processing_duration > target_frame_duration),
            # sleep_time будет <= 0, и задержки не будет, позволяя YOLO работать на пределе.
            target_frame_duration = 1.0 / video_source.fps # Ожидаемая длительность кадра от источника (например, 1/30 секунды)
            actual_processing_duration = time.time() - iteration_start_time # Фактическая длительность итерации (YOLO + другие операции)
            sleep_time = target_frame_duration - actual_processing_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

            frame = video_source.read_frame()
            if frame is None:
                # Попытка восстановления соединения для HLS
                if isinstance(video_source, HLSVideoSource):
                    logger.warning("Потеря HLS-потока. Попытка переподключения...")
                    if video_source.refresh():
                        logger.info("HLS-поток успешно восстановлен.")
                        continue
                    else:
                        logger.error("Не удалось восстановить HLS-поток.")
                
                logger.info("End of stream")
                break
            frame_count += 1

    except KeyboardInterrupt:
        logger.info("\nStopping node...")
    finally:
        video_source.release()
        video_writer.release()
        csv_logger.close()
        if args.debug:
            cv2.destroyAllWindows()
        node_public_address_manager.cleanup_ngrok() # Чистим Ngrok Ноды при завершении main()

if __name__ == "__main__":
    main()