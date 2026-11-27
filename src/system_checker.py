# src/system_checker.py

import json
import sys
from pathlib import Path
from src.logger import get_logger
import config
from src.utils import load_json_file

log = get_logger()

class SystemChecker:
    """Класс для проверки готовности системы, конфигов и инициализации структур."""

    def __init__(self):
        """Инициализирует пути к файлам и папкам."""
        self.config_files = [
            config.ARCHIVE_MAP_PATH,
            config.CLASS_MAPPING_PATH,
            config.PROCESSING_LOG_FILE,
            config.FULL_VALIDATION_MANIFEST_FILE
        ]
        self.required_dirs = [
            Path("logs"),
            Path("manifests"),
            Path("configs"),
            Path("datasets") # Папка для локального хранения данных
        ]

    def _ensure_dirs_exist(self) -> None:
        """Создает необходимые папки."""
        for d in self.required_dirs:
            d.mkdir(exist_ok=True)

    def _ensure_json_files_exist(self) -> None:
        """Проверяет наличие критических JSON-файлов и создает их пустыми, если отсутствуют."""
        for file_path in self.config_files:
            if not file_path.exists():
                # Создаем пустой JSON-словарь '{}'
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=4)
                log.info(f"Создана заглушка для файла: {file_path}")

    def run_checkup(self) -> dict[str, int]:
        """
        Проверяет наличие конфигов, папок, запускает инициализацию и проверяет готовность данных.

        :return: Список имен датасетов, готовых к обработке (run).
        """        
        # 1. Обеспечить наличие папок и файлов
        self._ensure_dirs_exist()
        self._ensure_json_files_exist()
        
        # 2. Проверка критических конфигов
        try:
            with open(config.ARCHIVE_MAP_PATH, 'r', encoding='utf-8') as f:
                archive_map = json.load(f)
            with open(config.CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                class_mapping = json.load(f)
        except json.JSONDecodeError as e:
            log.error(f"Ошибка декодирования JSON в критическом файле. Проверьте синтаксис: {e}")
            sys.exit(1)
        
        log.info("Режим CHECKUP: Проверка готовности системы.")
        
        # 3. Отчет о системе
        log.info(f"Конфигурация (archive_map): {len(archive_map)} датасетов.")
        log.info(f"Словарь классов (class_mapping): {len(class_mapping)} записей.")
        log.info(f"Целевой BBox эталон: {config.TARGET_WIDTH}x{config.TARGET_HEIGHT} (W x H).")
        
        log.info("Система готова к обработке.")

        # 4. Проверка готовности данных (Новая логика)
        processing_log = load_json_file(config.PROCESSING_LOG_FILE)
        
        datasets_to_process_map = {}
        
        for ds_name, ds_info in archive_map.items():
            ds_ready, file_count = self._check_dataset_readiness(ds_name, ds_info, class_mapping, processing_log)
            if ds_ready:
                datasets_to_process_map[ds_name] = file_count
                
        log.info(f"Обнаружено {len(datasets_to_process_map)} датасетов, готовых к **полной** обработке (run).")
        return datasets_to_process_map
    
    def _check_dataset_readiness(self, ds_name: str, ds_info: dict, class_mapping: dict, processing_log: dict) -> tuple[bool, int]:
        """
        Проверяет существование локальной папки, подсчитывает файлы и сверяет с протоколом.

        :param ds_name: Имя датасета.
        :param ds_info: Запись из archive_map.
        :param class_mapping: Полный словарь классов.
        :param processing_log: Протокол обработки.
        :return: True, если датасет готов к обработке, False в противном случае.
        """
        local_path = Path(ds_info.get('local_path', ''))
        ds_config = class_mapping.get(ds_name, {})
        subsets: list = ds_config.get('subsets_to_process', [])
        log_entry: dict = processing_log.get(ds_name, {})
        
        # 1. Проверка существования локальной папки
        if not local_path.exists():
            log.warning(f"[{ds_name}]: Не готов. Локальный путь не найден: {local_path.resolve()}")
            return False, 0

        # 2. Сверка с протоколом (предварительная)
        if log_entry.get('status') == 'COMPLETED':
            # Сверка по file_count будет проведена после подсчета, но пока считаем, что готов
            log.info(f"[{ds_name}]: Статус COMPLETED. Проверка на изменения...")

        # 3. Проверка и подсчет файлов в поднаборах
        total_image_count = 0
        is_ready = True

        # Словарь для агрегирования расширений по всему датасету
        total_image_extensions = {} 
        # all_image_extensions = set()
        
        for subset in subsets:
            img_dir = local_path / subset.get('image_path_in_archive', '')
            lbl_path_str = subset.get('label_path_in_archive', '')
            ### lbl_dir = local_path / subset.get('label_path_in_archive', '')
            
            # Подсчет изображений всех типов
            image_files = {} # Словарь для {'.jpg': 1000, '.png': 5}
            
            # Находим все файлы в папке изображений
            for item in img_dir.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    # Исключаем JSON-файлы из подсчета изображений, если они по ошибке в папке изображений
                    if ext == '.json':
                        continue 
                    image_files[ext] = image_files.get(ext, 0) + 1

                    # Агрегируем в суммарный словарь
                    total_image_extensions[ext] = total_image_extensions.get(ext, 0) + 1
            
            img_count = sum(image_files.values())

            # 3b. Проверка аннотаций в зависимости от формата
            annotation_format = ds_config.get('annotation_format', 'YOLO')
            lbl_count = 0 
            
            if annotation_format == 'YOLO':
                # Считаем количество .txt файлов
                lbl_dir = local_path / subset.get('label_path_in_archive', '')
                lbl_count = len(list(lbl_dir.glob('*.txt')))
                if lbl_count != img_count:
                    log.error(f"[{ds_name} / {subset.get('name')}]: Ошибка (YOLO). Найдены {img_count} изображений, но {lbl_count} аннотаций. Пропуск.")
                    is_ready = False
                
            elif annotation_format == 'COCO':
                # Проверяем наличие ровно одного JSON-файла
                coco_file = local_path / lbl_path_str
                if not coco_file.exists() or coco_file.suffix.lower() != '.json':
                    log.error(f"[{ds_name} / {subset.get('name')}]: Ошибка (COCO). COCO JSON файл не найден по пути: {coco_file}")
                    is_ready = False
                # NOTE: В COCO формате не сравниваем img_count с lbl_count (JSON всегда 1)
                
            else:
                log.error(f"[{ds_name}]: Неизвестный формат аннотаций: {annotation_format}. Пропуск.")
                is_ready = False
            
            ### lbl_count = len(list(lbl_dir.glob('*.txt'))) # Только YOLO-аннотации
            total_image_count += img_count

            if img_count == 0:

                log.warning(f"[{ds_name} / {subset.get('name')}]: Нет файлов изображений в: {img_dir}")
                is_ready = False
            
            ### if img_count != lbl_count:
            ###     log.error(f"[{ds_name} / {subset.get('name')}]: Ошибка. Найдены {img_count} изображений, но {lbl_count} аннотаций. Пропуск.")
            ###     is_ready = False

        if not is_ready:
            log.warning(f"[{ds_name}]: Пропущен. Обнаружены ошибки в структуре или несовпадение кол-ва файлов.")
            return False, 0
        
        # 4. Финальная сверка с протоколом
        if log_entry.get('status') == 'COMPLETED' and log_entry.get('file_count') == total_image_count:
            log.info(f"[{ds_name}]: Пропущено. {total_image_count} файлов. Статус COMPLETED и без изменений.")
            return False, 0

        # Отчет по расширениям (Используем суммарный словарь)
        ext_report = ", ".join([f"{ext}: {total_image_extensions[ext]}" for ext in sorted(total_image_extensions.keys())])
        log.info(f"[{ds_name}]: Расширения: {ext_report}.")

        log.info(f"[{ds_name}]: Готов к RUN. Найдено {total_image_count} изображений в {len(subsets)} поднаборах.")
        return True, total_image_count