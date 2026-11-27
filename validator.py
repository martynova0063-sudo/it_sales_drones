# validator.py

import sys
import argparse
from pathlib import Path
import config
from src.logger import get_logger
from src.system_checker import SystemChecker
from src.bbox_validator import BBoxValidator
from src.stats_analyser import StatsAnalyser
from src.dupe_checker import DupeChecker
from src.utils import load_json_file, save_json_file

log = get_logger()

def main() -> None:
    """Основной оркестратор программы."""
    parser = argparse.ArgumentParser(description="Программа для отбора и проверки датасетов летательных объектов.")
    parser.add_argument('mode', nargs='?', default='checkup', choices=['checkup', 'run', 'stats', 'doubles'], help='Режим работы: "checkup" (по умолчанию), "run" или "stats" (анализ манифеста) или "doubles" (поиск и пометка дубликатов).')

    args = parser.parse_args()
    
    log.info("Программа Dataset Validator: Сбор и проверка датасетов летательных объектов.")
    
    # 1. Проверка и инициализация системы
    checker = SystemChecker()
    
    # 2. Выбор режима
    if args.mode == 'checkup':
        checker.run_checkup()

    elif args.mode == 'run':
        # Сначала проверяем систему и получаем список готовых
        datasets_to_run_map = checker.run_checkup()
        
        # Загружаем конфиги и лог
        archive_map = load_json_file(config.ARCHIVE_MAP_PATH)
        class_mapping = load_json_file(config.CLASS_MAPPING_PATH)
        processing_log = load_json_file(config.PROCESSING_LOG_FILE)
        
        total_bboxes_collected = 0
        
        # Итерация только по тем, что checker признал готовыми
        for ds_name, file_count_to_save in datasets_to_run_map.items():
            log.info(f"Запуск полной валидации для {ds_name}...")
            
            ds_info = archive_map.get(ds_name, {})
            ds_config = class_mapping.get(ds_name, {})
            
            # Установка статуса IN_PROGRESS
            processing_log[ds_name] = {"status": "IN_PROGRESS", "file_count": file_count_to_save}
            save_json_file(processing_log, config.PROCESSING_LOG_FILE)

            # Вызов валидатора
            validator = BBoxValidator(ds_name, ds_info, ds_config)
            collected_count = validator.process_all_subsets()
            total_bboxes_collected += collected_count
            
            # Обновление статуса COMPLETED
            processing_log[ds_name]["status"] = "COMPLETED"
            processing_log[ds_name]["total_valid_bboxes"] = collected_count
            processing_log[ds_name]["file_count"] = file_count_to_save
            save_json_file(processing_log, config.PROCESSING_LOG_FILE)

        log.info(f"*** RUN: Обработка всех датасетов завершена. Собрано {total_bboxes_collected} валидных BBoxes. ***")

    elif args.mode == 'stats':
        # В режиме stats не нужно проверять данные на месте, только наличие манифеста
        analyser = StatsAnalyser()
        analyser.run_stats()

    elif args.mode == 'doubles':
        dupe_checker = DupeChecker()
        dupe_checker.run_dupe_check()

if __name__ == "__main__":
    main()