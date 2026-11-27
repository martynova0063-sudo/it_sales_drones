# src/utils.py

import json
import sys
from pathlib import Path
from src.logger import get_logger

log = get_logger()

def load_json_file(file_path: Path) -> dict:
    """
    Загружает JSON-файл или возвращает пустой словарь при ошибке/отсутствии.

    :param file_path: Путь к JSON-файлу.
    :return: Словарь с данными или пустой словарь.
    """
    if not file_path.exists():
        log.warning(f"Файл не найден: {file_path}. Возвращен пустой словарь.")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.error(f"Ошибка декодирования JSON в файле: {file_path}. Проверьте синтаксис.")
        return {}
    except Exception as e:
        log.error(f"Непредвиденная ошибка при чтении {file_path}: {e}")
        return {}

def save_json_file(data: dict, file_path: Path) -> None:
    """
    Безопасно сохраняет словарь в JSON-файл.

    :param data: Словарь для сохранения.
    :param file_path: Путь к JSON-файлу.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        log.error(f"Не удалось сохранить JSON в {file_path}: {e}")

def append_to_json_list(data_list: list, file_path: Path) -> None:
    """
    Загружает существующий JSON-файл (список), добавляет в него новые данные
    и сохраняет обратно.

    :param data_list: Список данных для добавления.
    :param file_path: Путь к JSON-файлу.
    """
    current_data = load_json_file(file_path) # Используем load_json_file
    
    # Если файл пуст или не список, инициализируем
    if not isinstance(current_data, list):
        current_data = []

    current_data.extend(data_list)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=4)
    except Exception as e:
        log.error(f"Не удалось добавить в JSON-список {file_path}: {e}")