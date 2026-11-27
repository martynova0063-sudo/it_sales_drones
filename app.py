# app.py
# стабильная версия с добавленными фичами:
#
# 1. **added** Настроить хранение ключей. (NODE_AUTH_KEY, CLIENT_AUTH_KEY)
# 2. **added** Создать верификацию ноды на пульте. (Проверка ключа ноды)
# 3. **added** Расширить данные событий. (Добавить скорость, прогноз)
# 4. **added** РЕФАКТОРИНГ: Создать общую шину событий. (для логов, ТГ, API)
# 5. **added** Добавить историю событий (REST API)
# 6. **added** Добавить события в реальном времени. (WebSocket API)
# 7. **added** Ускорить работу ноды. (запустить Алерты через treading, чтобы не тормозили процесс)

import base64
import os
import io
import cv2
import numpy as np
import gradio as gr
import uvicorn
import threading
import time
import requests
import atexit 
import asyncio
import socket
import subprocess
import tempfile
import shutil
import platform
import json

from PIL import Image
from typing import Optional, List, Tuple, Any, Dict
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from fastapi import WebSocket, WebSocketDisconnect

from datetime import datetime
from pathlib import Path

import config_monitor
from src.runtime_logger import get_runtime_logger
from src.public_address_manager import PublicAddressManager
from src.camera_registry import CameraState, CameraRegistry
from src.schemas import AlertPayload
from src.telegram_notifier import TelegramNotifier
from src.event_bus import EventBus
from dataclasses import dataclass
from pathlib import Path
import collections

NODE_AUTH_KEY, CLIENT_AUTH_KEY = config_monitor.NODE_AUTH_KEY, config_monitor.CLIENT_AUTH_KEY

logger = get_runtime_logger("App")

@dataclass
class UIState:
    """Класс для хранения состояния пользовательского интерфейса Gradio."""
    zoom_enabled: bool = False
    gallery_layout_key: str = "4_windows"
    video_mode_active: bool = False
    archive_download_status_message: str = "Ожидание..."
    downloaded_file_path: Optional[Path] = None

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================

global_settings = {
    "telegram_enabled": False
}

app = FastAPI()

# Создаем экземпляр менеджера публичных адресов для ПУ
app_public_address_manager = PublicAddressManager(service_port=9000, service_name="App")

# Глобальная переменная для хранения публичного URL ПУ, полученного от менеджера
public_app_url: str = "" # Инициализация пустой строкой, будет заполнена при инициализации

# Создаем экземпляр реестра камер
camera_registry = CameraRegistry()

# Создаем экземпляр EventBus
event_bus = EventBus()

# История последних событий для REST API
event_history = collections.deque(maxlen=100) # Храним последние 100 событий

# Создаем экземпляр TelegramNotifier
telegram_notifier = TelegramNotifier(
    bot_token=config_monitor.TELEGRAM_BOT_TOKEN,
    channel_id=config_monitor.TELEGRAM_CHANNEL_ID,
    chat_id=config_monitor.TELEGRAM_CHAT_ID
)

# Инициализация для API Key Header (для Нод)
api_key_header_node = APIKeyHeader(name="X-API-Key", auto_error=False)
# Инициализация для API Key Header (для Клиентов REST/WS)
api_key_header_client = APIKeyHeader(name="X-API-Key", auto_error=False)

# Функция-подписчик для Telegram
async def telegram_event_subscriber(payload: AlertPayload):
    if global_settings["telegram_enabled"] and payload.type == "alert" and telegram_notifier.is_configured():
        try:
            image_bytes = base64.b64decode(payload.image_base64)
            cam_state = camera_registry.get_camera_by_id(payload.node_id)
            camera_name = cam_state.camera_name if cam_state else "Unknown Camera"
            caption = (f"{payload.class_name} ({payload.confidence:.2f})\n"
                       f"{camera_name}\nID: {payload.node_id}")
            telegram_notifier.send_alert_photo(image_bytes, caption)
        except Exception as e:
            logger.error(f"Ошибка отправки алерта в Telegram через Notifier: {e}")

# Функция для проверки ключа API Ноды
async def verify_node_api_key(request: Request, api_key: Optional[str] = Depends(api_key_header_node)):

    """Функция для проверки ключа API Ноды."""
    if config_monitor.NODE_AUTH_KEY and api_key == config_monitor.NODE_AUTH_KEY:
        return True
    # Используем request.client.host для получения IP клиента
    client_ip = request.client.host if request.client else "Unknown IP"
    logger.warning(f"Попытка несанкционированного доступа к /alert. IP: {client_ip}, Предоставленный ключ: '{api_key}', ожидается: '{config_monitor.NODE_AUTH_KEY}'")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный ключ API Ноды",
        headers={"WWW-Authenticate": "API-Key"},
    )

# Функция для проверки ключа API Клиента
async def verify_client_api_key(request: Request, api_key: Optional[str] = Depends(api_key_header_client)):
    """Функция для проверки ключа API Клиента."""
    if config_monitor.CLIENT_AUTH_KEY and api_key == config_monitor.CLIENT_AUTH_KEY:
        return True
    # Используем request.client.host для получения IP клиента
    client_ip = request.client.host if request.client else "Unknown IP"
    logger.warning(f"Попытка несанкционированного доступа к клиентскому API. IP: {client_ip}, Предоставленный ключ: '{api_key}', ожидается: '{config_monitor.CLIENT_AUTH_KEY}'")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный ключ API Клиента",
        headers={"WWW-Authenticate": "API-Key"},
    )

async def initialize_public_address():
    """
    Определяет публичный/виртуальный адрес сервера в зависимости от PUBLIC_IP_PROVIDER.
    Записывает его в файл и, при необходимости, отправляет/закрепляет в Telegram.
    Вызывается после полного старта FastAPI.
    """
    global public_app_url

    public_app_url = await app_public_address_manager.initialize_public_address()

    if public_app_url == f"http://localhost:9000":
        logger.warning("Используется локальный адрес сервера. Ноды не смогут подключиться извне.")

    # 2. Запись в файл
    timestamp = int(time.time())
    pu_alert_url = f"{public_app_url}/alert"
    conn_info_str = f"{timestamp},{pu_alert_url}"

    try:
        with open(config_monitor.CONNECTION_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(conn_info_str)
        logger.info(f"URL сервера записан в {config_monitor.CONNECTION_FILE_PATH}")
    except Exception as e:
        logger.error(f"Ошибка записи URL в файл {config_monitor.CONNECTION_FILE_PATH}: {e}")

    # 3. Отправка в Telegram-канал
    if telegram_notifier.is_configured():
        try:
            telegram_notifier.send_system_url_message(pu_alert_url, timestamp)
        except Exception as e:
            logger.error(f"Ошибка работы с Telegram API (отправка/закрепление URL) через TelegramNotifier: {e}")
    else:
        logger.warning("TelegramNotifier (токен бота или ID канала) не найдены. URL не будет отправлен в Telegram.")

# ==========================================
# 2. API
# ==========================================

@app.post("/alert")
async def receive_alert(payload: AlertPayload, authenticated: bool = Depends(verify_node_api_key)):
    """Принимает данные алерта/хартбита от ноды."""
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
    except: 
        logger.error("Ошибка декодирования base64 изображения в AlertPayload.")
        return {"status": "error", "message": "Failed to decode image"}

    node_id = payload.node_id
    camera_registry.update_camera_state(node_id, payload, pil_image)
    
    # Опубликовать событие в Event Bus
    await event_bus.publish(payload)
    
    # Добавить в историю для REST API
    event_history.appendleft(payload.model_dump(exclude={"image_base64"}))

    # Выбираем камеру через camera_registry
    if camera_registry.get_selected_camera_id() is None: 
        camera_registry.set_selected_camera_id(node_id)

    return {"status": "ok"}

@app.get("/events")
async def get_events(authenticated: bool = Depends(verify_client_api_key)) -> List[Dict[str, Any]]:
    """
    Возвращает историю последних событий.
    Требует аутентификации по CLIENT_AUTH_KEY.
    """
    # event_history хранит Payload.dict, что уже сериализуемо
    return list(event_history)

# Отдельная функция для обработки WebSocket-подписки на EventBus
async def websocket_event_publisher(websocket: WebSocket, payload: AlertPayload):
    try:
        # Отправляем только часть Payload, исключая image_base64
        await websocket.send_json(payload.model_dump(exclude={"image_base64"}))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket клиент отключился во время отправки события.")
        pass
    except Exception as e:
        logger.error(f"Ошибка при отправке события по WebSocket: {e}", exc_info=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Извлекаем API-ключ вручную из заголовков WebSocket
    api_key = websocket.headers.get("x-api-key")

    # Проверка ключа API клиента для WebSocket
    if not (api_key and api_key == config_monitor.CLIENT_AUTH_KEY):
        logger.warning(f"Попытка несанкционированного подключения к WebSocket. IP: {websocket.client.host}:{websocket.client.port}, Предоставленный ключ: '{api_key}'")
        await websocket.send_json({"error": "Неверный ключ API Клиента"})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    logger.info(f"WebSocket клиент подключен: {websocket.client.host}:{websocket.client.port}")

    # Создаем функцию-колбэк, специфичную для этого WebSocket-соединения
    # (wrapper для передачи конкретного websocket-соединения)
    async def specific_websocket_callback(event_data: AlertPayload):
        await websocket_event_publisher(websocket, event_data)
    
    event_bus.subscribe_websocket(specific_websocket_callback) # Подписываем эту функцию

    try:
        while True:
            # Просто ждем сообщения от клиента. Можно добавить обработку команд от клиента,
            # но для получения событий эта часть может быть пассивной.
            # Если клиент закроет соединение, `receive_text()` выбросит WebSocketDisconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"WebSocket клиент отключился: {websocket.client.host}:{websocket.client.port}")
    except Exception as e:
        logger.error(f"Ошибка WebSocket соединения с клиентом {websocket.client.host}:{websocket.client.port}: {e}", exc_info=True)
    finally:
        # Отписываем колбэк при отключении клиента
        event_bus.unsubscribe_websocket(specific_websocket_callback)
        logger.info(f"WebSocket клиент отписан от EventBus.")

# ==========================================
# 4. ЛОГИКА UI
# ==========================================

def get_gallery_layout_params(layout_key: str) -> tuple[dict, int]:
    """
    Возвращает словарь параметров для gr.Gallery.update() и максимальное количество камер.

    Args:
        layout_key: Ключ, определяющий макет галереи (например, "4_windows").

    Returns:
        Кортеж из двух элементов:
        - Словарь с параметрами 'columns', 'rows', 'height', 'object_fit', 'allow_preview'.
        - Целое число, максимальное количество камер для отображения в этом макете.
    """
    params = {}
    max_cameras_val = 0
    if layout_key == "4_windows":
        params["columns"] = 4
        params["rows"] = 1
        max_cameras_val = 4
    elif layout_key == "8_windows":
        params["columns"] = 8
        params["rows"] = 1
        max_cameras_val = 8
    elif layout_key == "16_windows":
        params["columns"] = 8
        params["rows"] = 2
        max_cameras_val = 16
    else:  # По умолчанию
        params["columns"] = 4
        params["rows"] = 1
        max_cameras_val = 4
    params["object_fit"] = "contain"  # Сохраняет пропорции изображений.
    params["allow_preview"] = False

    return params, max_cameras_val

def get_thumbnails_for_gallery(layout_key: str):

    """
    Данные для Галереи с динамическим количеством окон.
    Args:
        layout_key: Ключ, определяющий текущий макет галереи.
    Returns:
        Список кортежей (PIL.Image, str) для отображения в галерее.
    """

    sorted_ids = sorted(camera_registry.get_all_camera_ids())
    ### sorted_ids = sorted(cameras.keys())

    _, max_cameras_to_display = get_gallery_layout_params(layout_key)

    imgs = []
    labels = []

    # Только те камеры, которые помещаются в выбранный макет
    display_ids = sorted_ids[:max_cameras_to_display]

    for nid in display_ids:
        cam = camera_registry.get_camera_by_id(nid)
        if cam:
            imgs.append(cam.get_display_image())
        
            status_icon = "🔴" if cam.alert_active else "🟢"
            # Используем cam.status напрямую   
            if cam.status == "offline": status_icon = "⚫"
            elif cam.status == "warning": status_icon = "🟠"

            labels.append(f"{status_icon} {cam.camera_name} ({nid})")

        else: 
            # Добавляем заглушку, если камера по какой-то причине не найдена в реестре
            arr = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(arr, "UNKNOWN CAMERA", (int(arr.shape[1]*0.2), int(arr.shape[0]*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) ## ИЗМЕНЕНО
            imgs.append(Image.fromarray(arr))
            labels.append(f"Неизвестная камера ({nid})")
    
    # Заглушки, если активных камер меньше, чем выбранный размер галереи
    while len(imgs) < max_cameras_to_display:
        arr = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(arr, "NO CAMERA", (int(arr.shape[1]*0.3), int(arr.shape[0]*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        imgs.append(Image.fromarray(arr))
        labels.append("Пустой слот")

    return list(zip(imgs, labels))

def get_main_view_content(ui_state: UIState) -> Tuple[Any, str, Any, Any, Any, Any, Any, Any, str]:
    """
    Формирует содержимое для главного монитора, информационного блока и кнопок.

    Args:
        ui_state: Объект UIState, содержащий текущее состояние UI.

    Returns:
        Кортеж из обновленных значений для Gradio-компонентов:
        - main_mon (gr.Image.update)
        - info_box (str)
        - btn_rst (gr.Button.update)
        - btn_vid (gr.Button.update)
        - btn_details (gr.Button.update)
        - video_player (gr.HTML.update)
        - btn_download_archive (gr.Button.update)
        - file_download_output (gr.File.update)
        - debug_file_status (str) - Для отладочного текстового поля
    """
    selected_camera_id = camera_registry.get_selected_camera_id()

    # Если камера не выбрана или неактивна, возвращаем дефолтные значения
    if not selected_camera_id:
        
        # main_mon будет видим, video_player скрыт, остальные кнопки скрыты
        return (
            gr.update(visible=True, value=np.zeros((480,640,3), np.uint8)), # main_mon (1)
            "### Ожидание данных...", # info_box (2)
            gr.update(visible=False), # btn_rst (3)
            gr.update(visible=False), # btn_vid (4)
            gr.update(visible=False), # btn_details (5)
            gr.update(visible=False, value=""), # video_player (6)
            gr.update(visible=False), # btn_download_archive (7)
            gr.update(visible=False, value=None), # file_download_output (8)
            ui_state.archive_download_status_message # для debug_file_status (9)
        )
    
    cam = camera_registry.get_camera_by_id(selected_camera_id)
    
    if not cam: # Дополнительная проверка на случай, если камера исчезла
        return (
            gr.update(visible=True, value=np.zeros((480,640,3), np.uint8)), # (1)
            "### Камера не найдена в реестре.", # (2)
            gr.update(visible=False), # (3)
            gr.update(visible=False), # (4)
            gr.update(visible=False), # (5)
            gr.update(visible=False, value=""), # (6)
            gr.update(visible=False), # (7)
            gr.update(visible=False, value=None), # (8)
            ui_state.archive_download_status_message # (9)
        )

    # Подготовка выходов для main_mon и video_player (По умолчанию скрываем)
    img_output = gr.update(visible=False)
    video_player_output = gr.update(visible=False, value="")


    if ui_state.video_mode_active:
        if cam.stream_url:
            # Если режим видео активен и есть URL потока, показываем видео
            video_html_content = f"""
            <div style="display: flex; justify-content: center; background: black;">
                <img src="{cam.stream_url}" style="height: 480px; width: auto; border: 2px solid red;" />
            </div>
            """
            video_player_output = gr.update(visible=True, value=video_html_content)
        else:
            # Отключаем видео режим, если URL нет. В этом случае далее будет показана картинка
            logger.warning(f"[WARN] Нет URL потока для камеры {selected_camera_id}. Отключаем видео режим.")

            # Отключаем видео режим, если URL нет. В этом случае далее будет показана картинка
            # Важно: тут мы меняем состояние UI, поэтому нужно вернуть обновленный ui_state
            ui_state.video_mode_active = False

    if not ui_state.video_mode_active:
        # Если режим видео не активен (или был отключен из-за отсутствия URL), показываем картинку
        img_val = cam.get_thumbnail_image() if ui_state.zoom_enabled else cam.get_display_image()
        img_output = gr.update(visible=True, value=img_val)

    status_text = "🔴 ТРЕВОГА" if cam.alert_active else ("🟢 АКТИВЕН" if cam.status == "active" else ("🟠 ПРЕДУПРЕЖДЕНИЕ" if cam.status == "warning" else "⚫ ОФФЛАЙН"))

    # Дополнительная проверка оффлайна для текста
    if not cam.alert_active and (time.time() - cam.last_seen > config_monitor.TIMEOUT_OFFLINE):
         status = "⚫ ОФФЛАЙН"

    t_str = time.strftime('%H:%M:%S', time.localtime(cam.last_seen))
    
    info = (f"## {cam.camera_name}\n**ID:** `{cam.node_id}` | **Модель:** `{cam.model_name}`\n\n"
            f"**Статус:** {status_text} | **Обновлено:** {t_str}")
    
    if cam.alert_active:
        info += f"\n\n### {cam.alert_info}"
        # Добавляем точность предсказания, если она есть
        if cam.last_prediction_iou is not None:
            iou_percentage = cam.last_prediction_iou * 100
            info += f"\n**Точность прогноза (IoU):** `{iou_percentage:.1f}%`"

    if cam.all_classes_names:
        info += f"\n\n**Все классы модели:** {', '.join(sorted(set(cam.all_classes_names)))}"

    if cam.target_classes:
        info += f"\n**Целевые классы:** {', '.join(sorted(set(cam.target_classes)))}"

    btn_rst = gr.update(visible=cam.alert_active)

    # Текст кнопки меняется в зависимости от режима видео
    btn_vid = gr.update(visible=True, value="Показать картинку" if ui_state.video_mode_active else "Запросить видео")

    zoom_lbl = "Общий вид" if ui_state.zoom_enabled else "Увеличить объект"
    # Кнопка зума скрывается в режиме видео
    btn_zm = gr.update(visible=not ui_state.video_mode_active, value=zoom_lbl)
        
    # Кнопка скачивания архива видна только если есть активная тревога и есть last_bbox (т.е. был обнаружен объект)
    btn_download_archive = gr.update(visible=cam.alert_active and bool(cam.last_bbox))

    file_download_output = gr.update(visible=ui_state.downloaded_file_path is not None, value=str(ui_state.downloaded_file_path) if ui_state.downloaded_file_path else None) # Сбрасываем File компонент

    return (
        img_output, # (1)
        info, # (2)
        btn_rst, # (3)
        btn_vid, # (4)
        btn_zm, # (5)
        video_player_output, # (6)
        btn_download_archive, # (7)
        file_download_output, # (8)
        ui_state.archive_download_status_message # 9-й элемент для debug_file_status
    )

def refresh_ui(ui_state: UIState) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, str, UIState]:
    """
    Обновляет все элементы пользовательского интерфейса.

    Args:
        ui_state: Объект UIState, содержащий текущее состояние UI.

    Returns:
        Список обновленных значений для Gradio-компонентов, включая обновленный UIState.
    """
    gallery_data = get_thumbnails_for_gallery(ui_state.gallery_layout_key)

    # Передаем video_mode_active_state в get_main_view_content
    main_data_outputs = get_main_view_content(ui_state)

    gallery_update_params, _ = get_gallery_layout_params(ui_state.gallery_layout_key)

    return (
        gr.update(value=gallery_data, **gallery_update_params),
        main_data_outputs[0], # main_mon
        main_data_outputs[1], # info_box
        main_data_outputs[2], # btn_rst
        main_data_outputs[3], # btn_vid
        main_data_outputs[4], # btn_details
        main_data_outputs[5], # video_player
        main_data_outputs[6], # btn_download_archive
        main_data_outputs[7], # file_download_output
        main_data_outputs[8], # debug_file_status
        ui_state # Возвращаем обновленный ui_state
    )

def on_gallery_select(evt: gr.SelectData):

    sorted_ids = sorted(camera_registry.get_all_camera_ids())
    if evt.index < len(sorted_ids):
        camera_registry.set_selected_camera_id(sorted_ids[evt.index])

def toggle_telegram_state(ui_state: UIState, is_enabled: bool) -> UIState:
    """
    Переключает состояние отправки уведомлений в Telegram.

    Args:
        ui_state: Текущее состояние UI.
        is_enabled: Новое булево значение состояния Telegram.

    Returns:
        Обновленный UIState.
    """
    global_settings["telegram_enabled"] = is_enabled
    state_str = "ВКЛЮЧЕНА" if is_enabled else "ВЫКЛЮЧЕНА"
    logger.info(f"[UI] Отправка в Telegram переключена: {state_str}")
    return ui_state

def _open_download_folder(ui_state: UIState) -> UIState:
    """
    Открывает системный файловый менеджер в директории скачанных архивов.
    Обновляет статусное сообщение в UIState в случае ошибки.

    Args:
        ui_state: Текущее состояние UI.

    Returns:
        Обновленный UIState.
    """
    folder_path = config_monitor.DOWNLOADED_ARCHIVES_DIR
    logger.info(f"Попытка открыть папку: {folder_path}")

    try:
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Папка {folder_path} не существовала и была создана.")

        current_platform = platform.system()

        if current_platform == "Windows":
            subprocess.Popen(['start', str(folder_path)], shell=True)
        elif current_platform == "Darwin": # macOS
            subprocess.Popen(["open", str(folder_path)])
        else: # Linux
            subprocess.Popen(["xdg-open", str(folder_path)])
        ui_state.archive_download_status_message = f"Папка '{folder_path.name}' успешно открыта."
    except FileNotFoundError:
        ui_state.archive_download_status_message = f"Ошибка: Команда открытия папки не найдена для вашей ОС. Путь: {folder_path}"
        logger.error(f"Ошибка открытия папки: Команда не найдена. Путь: {folder_path}")
    except Exception as e:
        ui_state.archive_download_status_message = f"Ошибка при открытии папки: {e}. Путь: {folder_path}"
        logger.error(f"Непредвиденная ошибка при открытии папки '{folder_path}': {e}")
    
    return ui_state

def _timer_callback(ui_state: UIState):
    """
    Вспомогательная функция-колбэк для Gradio Timer.Tick.
    Выполняет обновление статусов камер и затем обновляет UI.
    """
    camera_registry.update_camera_statuses()
    return refresh_ui(ui_state)

def toggle_video_mode_state(ui_state: UIState) -> UIState:
    """
    Переключает состояние активности видеорежима.

    Args:
        ui_state: Текущее состояние UI.

    Returns:
        Обновленный UIState с инвертированным video_mode_active.
    """
    ui_state.video_mode_active = not ui_state.video_mode_active
    return ui_state

def _reset_selected_camera_alert():
    """
    Вспомогательная функция для сброса тревоги выбранной камеры.
    Обрабатывает случай, когда камера не выбрана (selected_camera_id is None).
    """
    selected_id = camera_registry.get_selected_camera_id()
    if selected_id:
        camera_registry.reset_alert_for_camera(selected_id)
    else:
        logger.warning("Попытка сбросить тревогу, когда ни одна камера не выбрана.")

async def _handle_archive_download(ui_state: UIState) -> Tuple[Any, Any, str, str, UIState]:
    """
    Обрабатывает запрос на скачивание архивного видео с ноды.

    Args:
        ui_state: Текущее состояние UI.

    Returns:
        Кортеж: Обновление кнопки "Скачать архив", обновление компонента файла,
        статусное сообщение для info_box, статусное сообщение для debug_file_status,
        и обновленный UIState. ## ИЗМЕНЕНО
    """
    current_node_id = camera_registry.get_selected_camera_id()
    
    if not current_node_id:
        ui_state.archive_download_status_message = "Ошибка: Камера не выбрана."
        ui_state.downloaded_file_path = None
        return(
           gr.update(value="Скачать архив", interactive=True), # Кнопка снова активна
            gr.update(visible=False, value=None), # Файл готов к скачиванию
            "Ошибка: Камера не выбрана.",
            ui_state.archive_download_status_message,
            ui_state
        )
   
    cam_state = camera_registry.get_camera_by_id(current_node_id)
    if not cam_state or not cam_state.alert_active:
        ui_state.archive_download_status_message = "Ошибка: Нет активной тревоги для выбранной камеры."
        ui_state.downloaded_file_path = None

        return (
            gr.update(value="Скачать архив", interactive=True), # Кнопка снова активна
            gr.update(visible=False, value=None), # Файл готов к скачиванию
            "Ошибка: Нет активной тревоги для выбранной камеры.",
            ui_state.archive_download_status_message,
            ui_state
        )

    if cam_state.last_alert_timestamp_iso:
        alert_timestamp = cam_state.last_alert_timestamp_iso
    else:
        # Fallback, если по какой-то причине timestamp тревоги не был сохранен (например, старые данные)
        logger.warning(f"Нет last_alert_timestamp_iso для камеры {current_node_id}. Используем last_seen.")
        alert_timestamp = datetime.fromtimestamp(cam_state.last_seen).isoformat()

    node_base_url = cam_state.stream_url.replace("/video_feed", "") # Получаем базовый URL ноды
    if not node_base_url:
        ui_state.archive_download_status_message = "Ошибка: Не удалось получить базовый URL ноды."
        ui_state.downloaded_file_path = None

        return (
            gr.update(value="Скачать архив", interactive=True),
            gr.update(visible=False, value=None),
            "Ошибка: Не удалось получить базовый URL ноды.",
            ui_state.archive_download_status_message,
            ui_state
        )

    archive_url = (f"{node_base_url}/archive_video/{current_node_id}/{alert_timestamp}"
                   f"?pre_event_sec={config_monitor.ARCHIVE_DOWNLOAD_PRE_EVENT_SEC}"
                   f"&post_event_sec={config_monitor.ARCHIVE_DOWNLOAD_POST_EVENT_SEC}")
    
    logger.info(f"Запрос архивного видео: {archive_url}")
    ui_state.archive_download_status_message = "Запрос отправлен Ноде..."
    ui_state.downloaded_file_path = None

    try:
        # Gradio колбэки являются синхронными. Для выполнения асинхронного HTTP-запроса
        # и обработки потенциально большого файла, используем asyncio.to_thread.run_sync.
        # requests.get будет блокировать поток, но не UI поток Gradio.
        def _download_sync_task() -> Path:
            _response = requests.get(archive_url, stream=True, timeout=300) # Увеличиваем таймаут для больших файлов
            _response.raise_for_status() # Вызывает HTTPError для плохих статусов (4xx, 5xx)

            # Создаем уникальное имя файла для сохранения в DOWNLOADED_ARCHIVES_DIR
            timestamp_for_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"archive_{current_node_id}_{timestamp_for_filename}.mp4"
            local_download_path = config_monitor.DOWNLOADED_ARCHIVES_DIR / filename

            # Нельзя напрямую менять ui_state из вложенной функции, запущенной в отдельном потоке.
            # Статус будет обновлен после возврата из await asyncio.to_thread(_download_sync_task).
            with open(local_download_path, 'wb') as f:
                for chunk in _response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return local_download_path
        
        ui_state.downloaded_file_path = await asyncio.to_thread(_download_sync_task)
        ui_state.archive_download_status_message = "Видео успешно подготовлено и сохранено."
        
        logger.info(f"Архивное видео успешно скачано и сохранено: {ui_state.downloaded_file_path}")

        # Gradio может отдать временный файл, если мы укажем его путь.
        # Нужно зарегистрировать его удаление после того, как Gradio с ним закончит,
        # или использовать `atexit.register` (уже есть для Node, но тут другое место).
        # Gradio сам удалит временный файл, если он передан как value=pathlib.Path.
        return (
            gr.update(value="Скачать архив", interactive=True), 
            gr.update(visible=True, value=str(ui_state.downloaded_file_path)),
            "Видео успешно подготовлено. Нажмите для скачивания.",
            f"Видео успешно подготовлено и сохранено в: {ui_state.downloaded_file_path}",
            ui_state
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка HTTP запроса при скачивании архива: {e}")
        # В случае ошибки убедиться, что файл, если он был начат, удален
        if ui_state.downloaded_file_path and ui_state.downloaded_file_path.exists():
            os.remove(ui_state.downloaded_file_path)
            ui_state.downloaded_file_path = None
        ui_state.archive_download_status_message = f"Ошибка сети при скачивании: {e}"
        return gr.update(value="Скачать архив", interactive=True), gr.update(visible=False, value=None), f"Ошибка сети: {e}", ui_state.archive_download_status_message, ui_state
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при скачивании архива: {e}")
        # В случае ошибки убедиться, что файл, если он был начат, удален
        if ui_state.downloaded_file_path and ui_state.downloaded_file_path.exists():
            os.remove(ui_state.downloaded_file_path)
            ui_state.downloaded_file_path = None
        ui_state.archive_download_status_message = f"Непредвиденная ошибка при скачивании: {e}"
        return gr.update(value="Скачать архив", interactive=True), gr.update(visible=False, value=None), f"Ошибка: {e}", ui_state.archive_download_status_message, ui_state

# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================

with gr.Blocks(title="Security Center") as demo:
    gr.Markdown("# 🛡️ Ситуационный Центр")
    
    with gr.Row():
        tg_sw = gr.Checkbox(label="Отправка уведомлений в Telegram", value=False, interactive=True)

    gr.Markdown("### Обзор камер")

    ### app_ui_state = gr.State(UIState())

    with gr.Row(): # Кнопки выбора макета галереи
        gr.Markdown("#### Макет галереи:")
        btn_layout_4 = gr.Button("4 окна", size="sm", scale=1)
        btn_layout_8 = gr.Button("8 окон", size="sm", scale=1)
        btn_layout_16 = gr.Button("16 окон", size="sm", scale=1)
    
    with gr.Row(): # Галерея камер
        camera_gallery = gr.Gallery(
            label="Камеры", show_label=False, 
            columns=4, rows=1,
            interactive=True, allow_preview=False, object_fit="contain"
        )

    with gr.Row():
        with gr.Column(scale=3):
            main_mon = gr.Image(label="Монитор", interactive=False)
            # Видеоплеер (скрыт по умолчанию)
            video_player = gr.HTML(visible=False)

            # Компонент для скачивания файла, изначально скрыт
            file_download_output = gr.File(
                label="Скачать архивное видео", 
                file_count="single", 
                visible=False,
                interactive=False # Не позволяет пользователю загружать, только скачивать
            )

            # Временный текстовый блок для отладки статуса загрузки файла
            debug_file_status = gr.Textbox(label="Статус файла", visible=True, interactive=False, value="Ожидание...")
          
        with gr.Column(scale=1):
            info_box = gr.Markdown("Загрузка...")
            gr.Markdown("---")
            
            btn_details = gr.Button("Подробнее...", visible=False)

            with gr.Row():
                btn_rst = gr.Button("Сброс тревоги", variant="stop", visible=False)
                btn_vid = gr.Button("Запросить видео", visible=False)
                btn_download_archive = gr.Button("Скачать архив", variant="primary", visible=False)

            btn_open_download_folder = gr.Button("Открыть папку с архивами", visible=True)

    app_ui_state = gr.State(UIState())

    # ОБНОВЛЕНИЕ - timer.tick теперь принимает gallery_layout_state
    timer = gr.Timer(1.0) # Вернул 1 секунду, как было в стабильной версии

    # ОБНОВЛЕНИЕ
    timer.tick(
        fn=_timer_callback,
        inputs=[app_ui_state],
        outputs=[camera_gallery, main_mon, info_box, btn_rst, btn_vid, btn_details, video_player, btn_download_archive, file_download_output, debug_file_status, app_ui_state]
    )

    # СОБЫТИЯ
    camera_gallery.select(
        fn=on_gallery_select,
        inputs=None
    ).then(
        fn=refresh_ui,
        inputs=[app_ui_state],
        outputs=[camera_gallery, main_mon, info_box, btn_rst, btn_vid, btn_details, video_player, btn_download_archive, file_download_output, debug_file_status, app_ui_state]
    )

    tg_sw.change(fn=toggle_telegram_state, inputs=[app_ui_state, tg_sw], outputs=[app_ui_state])

    btn_rst.click(
        fn=_reset_selected_camera_alert,
        inputs=None,
        outputs=None 
    ).then(
        fn=refresh_ui,
        inputs=[app_ui_state],
        outputs=[camera_gallery, main_mon, info_box, btn_rst, btn_vid, btn_details, video_player, btn_download_archive, file_download_output, debug_file_status, app_ui_state]
    )

    btn_vid.click(
        fn=toggle_video_mode_state,
        inputs=[app_ui_state],
        outputs=[app_ui_state]
    ).then(
        fn=refresh_ui,
        inputs=[app_ui_state],
        outputs=[camera_gallery, main_mon, info_box, btn_rst, btn_vid, btn_details, video_player, btn_download_archive, file_download_output, debug_file_status, app_ui_state]
    )

    # Логика переключения Зума (пока через refresh_ui с параметром, но так как стейта нет - просто заглушка перерисовки)
    # Важно: Чтобы зум работал полноценно, нужно вернуть zoom_state. 
    # В текущем твоем коде zoom_state удален. 
    # Пока сделаем кнопку "пустышкой", чтобы код не падал, или вернем zoom_state?
    # Давай вернем zoom_state для работы.
    # zoom_state = gr.State(False) 

    btn_details.click(
        fn=lambda ui_state: UIState(zoom_enabled=not ui_state.zoom_enabled, gallery_layout_key=ui_state.gallery_layout_key, video_mode_active=ui_state.video_mode_active, archive_download_status_message=ui_state.archive_download_status_message, downloaded_file_path=ui_state.downloaded_file_path),
        inputs=[app_ui_state],
        outputs=[app_ui_state]
    ).then(
        fn=refresh_ui,
        inputs=[app_ui_state],
        outputs=[camera_gallery, main_mon, info_box, btn_rst, btn_vid, btn_details, video_player, btn_download_archive, file_download_output, debug_file_status, app_ui_state]
    )

    # Обработчики для новых кнопок выбора макета галереи
    btn_layout_4.click(
        fn=lambda ui_state: UIState(zoom_enabled=ui_state.zoom_enabled, gallery_layout_key="4_windows", video_mode_active=ui_state.video_mode_active, archive_download_status_message=ui_state.archive_download_status_message, downloaded_file_path=ui_state.downloaded_file_path),
        inputs=[app_ui_state],
        outputs=app_ui_state
    ).then(
        fn=refresh_ui,
        inputs=[app_ui_state],
        outputs=[camera_gallery, main_mon, info_box, btn_rst, btn_vid, btn_details, video_player, btn_download_archive, file_download_output, debug_file_status, app_ui_state]
    )

    btn_layout_8.click(
        fn=lambda ui_state: UIState(zoom_enabled=ui_state.zoom_enabled, gallery_layout_key="8_windows", video_mode_active=ui_state.video_mode_active, archive_download_status_message=ui_state.archive_download_status_message, downloaded_file_path=ui_state.downloaded_file_path),
        inputs=[app_ui_state],
        outputs=app_ui_state
    ).then(
        fn=refresh_ui,
        inputs=[app_ui_state],
        outputs=[camera_gallery, main_mon, info_box, btn_rst, btn_vid, btn_details, video_player, btn_download_archive, file_download_output, debug_file_status, app_ui_state]
    )

    btn_layout_16.click(
        fn=lambda ui_state: UIState(zoom_enabled=ui_state.zoom_enabled, gallery_layout_key="16_windows", video_mode_active=ui_state.video_mode_active, archive_download_status_message=ui_state.archive_download_status_message, downloaded_file_path=ui_state.downloaded_file_path),
        inputs=[app_ui_state],
        outputs=app_ui_state
    ).then(
        fn=refresh_ui,
        inputs=[app_ui_state],
        outputs=[camera_gallery, main_mon, info_box, btn_rst, btn_vid, btn_details, video_player, btn_download_archive, file_download_output, debug_file_status, app_ui_state]
    )

    # Обработчик кнопки "Скачать архив"
    btn_download_archive.click(
        fn=lambda: gr.update(value="Подготовка...", interactive=False), # Сначала отключаем кнопку и меняем текст
        inputs=None,
        outputs=btn_download_archive,
    ).then(
        fn=_handle_archive_download,
        inputs=[app_ui_state],
        outputs=[btn_download_archive, file_download_output, info_box, debug_file_status, app_ui_state],
    )

    # Обработчик кнопки "Открыть папку с архивами"
    btn_open_download_folder.click(
        fn=_open_download_folder,
        inputs=[app_ui_state],
        outputs=[debug_file_status, app_ui_state]
    )

app = gr.mount_gradio_app(app, demo, path="/")

# FastAPI startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Выполнение startup задач FastAPI...")
    # Регистрируем подписчика Telegram в EventBus
    event_bus.subscribe(telegram_event_subscriber)
    logger.info("Telegram subscriber зарегистрирован в EventBus.")
    # Инициализируем публичный адрес App здесь, чтобы EventBus был готов
    await initialize_public_address()
    logger.info("Startup задачи FastAPI завершены.")

if __name__ == "__main__":
    logger.info("Запуск на порту 9000...")

    # Запускаем ngrok сразу в основном потоке (так как uvicorn ниже)
    # Но оборачиваем в поток, чтобы не блокировать uvicorn
    # ВНИМАНИЕ: Если uvicorn стартует до того, как ngrok успеет создать туннель, 
    # может быть "connection refused". Это временное решение.
    # Более надежно - запускать ngrok из shell-скрипта или отдельного процесса.
    
    # Запускаем Ngrok и публикацию в отдельном фоновом потоке, 
    # который не блокирует основной процесс Uvicorn.
    # Это решает проблему "connection refused" при раннем старте ngrok.
    # Процесс Ngrok будет запущен до старта Uvicorn, но в фоне.
    def run_ngrok_in_background():
        import asyncio
        asyncio.run(initialize_public_address())

    ngrok_thread = threading.Thread(target=run_ngrok_in_background, daemon=True)
    ngrok_thread.start()

    # Регистрируем функцию отключения Ngrok для вызова при выходе из программы
    def cleanup_ngrok():
        app_public_address_manager.cleanup_ngrok()
    atexit.register(cleanup_ngrok)

    uvicorn.run(app, host="0.0.0.0", port=9000)