# config_monitor.py

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

####################################################################
# ПУТИ И ДИРЕКТОРИИ
####################################################################
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
OUTPUT_VIDEO_DIR = BASE_DIR / "output_videos"

# Создаем директории, если их нет
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADED_ARCHIVES_DIR = BASE_DIR / "downloaded_archives"
DOWNLOADED_ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_DIR = BASE_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
CONNECTION_FILE_PATH = CONFIGS_DIR / "connection.txt"

####################################################################
# НАСТРОЙКИ МОДЕЛИ
####################################################################
DEFAULT_MODEL_WEIGHTS = "training_model/checkpoints/YOLO_Drone_Detection/best_yolo26n.pt"
# DEFAULT_MODEL_WEIGHTS = "training_model/checkpoints/YOLO_Drone_Detection/best_yolo26s.pt"
# DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"

# Если используете кастомную модель по умолчанию:
# DEFAULT_MODEL_WEIGHTS = "models/custom_yolo.pt"

####################################################################
# НАСТРОЙКИ ВИДЕО
####################################################################
CHUNK_DURATION_SEC = 30  # Длительность одного видеофайла (сек)
TARGET_FPS = 30.0

####################################################################
# СЕТЕВЫЕ НАСТРОЙКИ (Notifier & Telegram)
####################################################################
# URL куда нода отправляет алерты (Notifier)
NOTIFIER_URL = os.getenv("NOTIFIER_URL", "http://localhost:9000/alert")

# Токены Telegram (используются в Notifier, но храним конфигурацию здесь)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Настройки Service Discovery (Канал и Ngrok)
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID") # Канал для пина сообщений
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
# USE_NGROK = False  # Включать ли ngrok при старте ПУ автоматически?

# Ключи аутентификации API (загружаются из .env)
NODE_AUTH_KEY: Optional[str] = os.getenv("NODE_AUTH_KEY")
CLIENT_AUTH_KEY: Optional[str] = os.getenv("CLIENT_AUTH_KEY")

####################################################################
# УНИВЕРСАЛЬНЫЕ НАСТРОЙКИ SERVICE DISCOVERY
####################################################################
# Возможные значения: "NGROK", "STATIC_IP", "TAILSCALE", "LOCAL_ONLY"
PUBLIC_IP_PROVIDER = os.getenv("PUBLIC_IP_PROVIDER", "TAILSCALE")

# Если выбран "STATIC_IP", укажите адрес:
STATIC_IP_ADDRESS = os.getenv("STATIC_IP_ADDRESS", "")

# Если выбрана "TAILSCALE", Tailscale должен быть установлен и авторизован. 
# IP будет определяться автоматически.

####################################################################
# ПАРАМЕТРЫ ДЕТЕКЦИИ
####################################################################
# Классы, которые считаем угрозой (названия как в модели)
#TARGET_CLASSES = ["drone", "дрон", "uav", "person", "car"]
TARGET_CLASSES = ["drone", "дрон", "uav", "car"]
TARGET_CLASSES = ["drone", "drones", "дрон", "uav"]

####################################################################
# НАСТРОЙКИ НОДЫ (Edge Node)
####################################################################
STREAM_PORT_DEFAULT = 8001        # Порт для MJPEG стрима
HEARTBEAT_INTERVAL = 10.0         # Как часто слать сигнал жизни (сек)
ALERT_COOLDOWN = 1.0              # Задержка между алертами (сек)

# Настройки трекинга
TRACK_HISTORY_LENGTH = 300 # Длина хвоста трекера в кадрах (например, 150 кадров при 30 FPS = 5 секунд)

# Настройки Service Discovery для Ноды
USE_TELEGRAM_DISCOVERY = True    # Использовать Telegram для поиска URL ПУ
USE_FILE_DISCOVERY = True        # Использовать файл для поиска URL ПУ
DISCOVERY_CHECK_INTERVAL = 10.0  # Как часто НУ / Нода проверяет адрес ПУ (сек)

# Настройки сбора данных для датасета
DATA_COLLECTION_FRAME_INTERVAL = 15 # Сохранять каждый N-й кадр при сборе данных (--collect_data)

# Настройки скачивания архивного видео
ARCHIVE_DOWNLOAD_PRE_EVENT_SEC = 10 # Сколько секунд видео до момента тревоги скачивать
ARCHIVE_DOWNLOAD_POST_EVENT_SEC = 20 # Сколько секунд видео после момента тревоги скачивать

# Настройки скачивания архивного видео
ARCHIVE_DOWNLOAD_PRE_EVENT_SEC = 10 # Сколько секунд видео до момента тревоги скачивать
ARCHIVE_DOWNLOAD_POST_EVENT_SEC = 20 # Сколько секунд видео после момента тревоги скачивать

####################################################################
# НАСТРОЙКИ ПУЛЬТА (Dashboard)
####################################################################
TIMEOUT_WARNING = 35   # Если молчит 35 сек -> Желтый/Синий (потеря связи)
TIMEOUT_OFFLINE = 60   # Если молчит 60 сек -> Серый (Оффлайн)