# config.py

from pathlib import Path

####################################################################
# ЦЕЛЕВЫЕ ПАРАМЕТРЫ
# Эталонный размер
TARGET_WIDTH = 640
TARGET_HEIGHT = 640

####################################################################
# ПУТИ
# Пути к динамически изменяемым файлам (логи, манифесты)
PROCESSING_LOG_FILE = Path("logs/processing_log.json")
FULL_VALIDATION_MANIFEST_FILE = Path("manifests/full_validation_manifest.json")

# Входные данные - файлы хранятся в папке 'configs/'
ARCHIVE_MAP_PATH = Path("configs/archive_map.json")
CLASS_MAPPING_PATH = Path("configs/class_mapping.json")

# Лог проблемных BBoxes
PROBLEM_BBOX_LOG_FILE = Path("logs/problem_bboxes.json")
# Лог дубликатов
DUPLICATE_LOG_FILE = Path("logs/duplicate_images.json")
# Лог структуры исследуемого архива
ARCHIVE_STRUCTURE_LOG = Path("logs/archive_structure_log.json")

# ПАРАМЕТРЫ КОНВЕРТАЦИИ MAT-TO-YOLO
# Корень папки, содержащей Training_data_V (обычно, master-папка датасета)
MATLAB_DATASET_MASTER_DIR = Path("data/Drone-detection-dataset-master")
# Исходная папка с PNG и MAT-файлами
MATLAB_IMAGES_SOURCE_DIR = MATLAB_DATASET_MASTER_DIR / 'Training_data_V'
# Целевая папка для финального YOLO-датасета
YOLO_OUTPUT_BASE_DIR = MATLAB_DATASET_MASTER_DIR / 'train'

# Словарь для маппинга меток
MATLAB_CLASS_MAPPING = {'AIRPLANE': 0, 'BIRD': 1, 'DRONE': 2, 'HELICOPTER': 3}

####################################################################
# ОГРАНИЧЕНИЕ RUN
# Максимальное количество файлов для обработки (0 = все файлы)
MAX_FILES_TO_PROCESS = 0

# Частота кадров для пропуска (для снижения избыточности). 1 = все кадры.
FRAME_SKIP_N = 1 

# Файл для статистики конвертации YOLO
YOLO_CONVERSION_LOG_FILE = Path("logs/yolo_conversion_stats.txt")

####################################################################
# НАСТРОЙКИ АНАЛИЗА ДУБЛИКАТОВ
# Включить поиск схожих (приблизительных) изображений по pHash distance
ENABLE_APPROXIMATE_DUPE_CHECK = True 

####################################################################
# ТРЕБОВАНИЯ
# Размеры BBoxes в пикселях
SIZE_CATEGORIES = {
    "Micro": (0, 20),
    "Tiny": (21, 40),
    "Small": (41, 80),
    "Medium": (81, 160),
    "Large": (161, 320),
    "XLarge": (321, 2000)
}
# Порог (Hamming Distance) для определения "схожести" изображений (pHash)
PHASH_DUPE_THRESHOLD = 5 # (0 - идентичны, 5-10 - очень схожи)

####################################################################
# НАСТРОЙКИ К ВИЗУАЛИЗАЦИИ
DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"  # из Ultralytics
CUSTOM_MODEL_PATH = "models/custom_yolo.pt"
OUTPUT_VIDEO_DIR = Path("output_videos")
LOG_DIR = Path("logs")
CHUNK_DURATION_SEC = 30

