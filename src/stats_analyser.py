# src/stats_analyser.py

import sys
from pathlib import Path
from src.logger import get_logger
from src.utils import load_json_file, save_json_file
import config
from collections import defaultdict
from typing import Any

log = get_logger()

class StatsAnalyser:
    """Класс для анализа Full Validation Manifest и вывода статистики (режим STATS)."""

    def __init__(self):
        self.manifest = load_json_file(config.FULL_VALIDATION_MANIFEST_FILE)
        if not self.manifest:
            log.error("Манифест пуст или не найден. Сначала выполните 'run'.")
            sys.exit(1)

    def run_stats(self) -> None:
        """Собирает и выводит подробную статистику по всем датасетам в манифесте."""
        
        log.info("Режим STATS: Сбор и вывод статистики.")
        
        for ds_name, bboxes_list in self.manifest.items():
            
            stats = {
                "total_bboxes": len(bboxes_list),
                "images": defaultdict(int),
                "class_counts": defaultdict(int),
                "sub_class_counts": defaultdict(int),
                "size_category_counts": defaultdict(int),
                "frame_category_counts": defaultdict(int),
            }
            
            # Используем set для подсчета уникальных изображений
            unique_images = set()

            for bbox in bboxes_list:
                # Статистика по BBoxes
                stats['class_counts'][bbox['target_class']] += 1
                stats['sub_class_counts'][bbox['sub_class']] += 1
                stats['size_category_counts'][bbox['size_category']] += 1
                
                # Статистика по Кадрам (Считаем только 1 раз за кадр)
                image_key = (bbox['image_path_in_archive'], bbox['original_width'], bbox['original_height'])
                if image_key not in unique_images:
                    unique_images.add(image_key)
                    stats['frame_category_counts'][bbox['frame_category']] += 1
                    stats['images']['total'] += 1
                    
            self._print_report(ds_name, stats)

        # 2. Сводный отчет по всем датасетам
        self._print_combined_report()

        # 3. Поиск дубликатов (интегрирован)
        self._print_dupe_report()


    def _print_dupe_report(self) -> None:
        """Выводит отчет по дубликатам, прочитанный из DUPLICATE_LOG_FILE."""
        
        log.info("\n=======================================================================================================")
        log.info("                                  АНАЛИЗ ДУБЛИКАТОВ И СХОЖЕСТИ")
        log.info("=======================================================================================================")

        # 1. Загрузка лога дубликатов
        dupes_data = load_json_file(config.DUPLICATE_LOG_FILE)
        if not dupes_data:
            log.warning("Лог дубликатов не найден. Запустите программу в режиме 'doubles' для анализа.")
            log.info("-------------------------------------------------------------------------------------------------------")
            return

        exact_dupes = dupes_data.get("exact_duplicates_groups", [])
        approx_dupes = dupes_data.get("approximate_duplicates_pairs", [])
        
        log.info(f"\n[ 100% ИДЕНТИЧНЫЕ ДУБЛИКАТЫ (MD5 == MD5) - {len(exact_dupes)} групп ]")
        for i, group in enumerate(exact_dupes[:5]):
             log.info(f"  Группа {i+1} (MD5: {group.get('md5_hash')[:10]}...): Master: {group.get('master_image')} | Дубликатов: {len(group.get('duplicates', []))}")
        if len(exact_dupes) > 5:
             log.info(f"  ... и еще {len(exact_dupes) - 5} групп.")

        if config.ENABLE_APPROXIMATE_DUPE_CHECK:
            log.info(f"\n[ ПРИБЛИЗИТЕЛЬНЫЕ ДУБЛИКАТЫ (Hamming Dist <= {config.PHASH_DUPE_THRESHOLD}) - {len(approx_dupes)} пар ]")
            if approx_dupes:
                for i, pair in enumerate(approx_dupes[:5]):
                    log.info(f"  Пара {i+1} (Dist={pair['distance']}): {pair.get('image1')} <-> {pair.get('image2')}")
                if len(approx_dupes) > 5:
                    log.info(f"  ... и еще {len(approx_dupes) - 5} пар.")
        else:
            log.info("Поиск приблизительных дубликатов отключен (ENABLE_APPROXIMATE_DUPE_CHECK = False).")

        log.info("-------------------------------------------------------------------------------------------------------")

    def _generate_combined_stats(self) -> dict[str, Any]:
        """Агрегирует статистику по всем датасетам в манифесте."""
        combined_stats = {
            "total_bboxes": 0,
            "images": defaultdict(int),
            "class_counts": defaultdict(int),
            "sub_class_counts": defaultdict(int),
            "size_category_counts": defaultdict(int),
            "frame_category_counts": defaultdict(int),
        }
        
        unique_images_global = set()

        for ds_name, bboxes_list in self.manifest.items():
            
            # Агрегация по BBoxes
            combined_stats['total_bboxes'] += len(bboxes_list)
            
            # Агрегация по Кадрам (Считаем только 1 раз за кадр)
            unique_images_local = set()

            for bbox in bboxes_list:
                # 1. BBox Статистика
                combined_stats['class_counts'][bbox['target_class']] += 1
                combined_stats['sub_class_counts'][bbox['sub_class']] += 1
                combined_stats['size_category_counts'][bbox['size_category']] += 1
                
                # 2. Кадровая Статистика (проверка на глобальную уникальность, чтобы не дублировать)
                # Идеальная глобальная уникальность невозможна без хеша картинки, но мы используем (ds_name, path)
                image_key_global = (ds_name, bbox['image_path_in_archive'])
                
                if image_key_global not in unique_images_global:
                    unique_images_global.add(image_key_global)
                    combined_stats['images']['total'] += 1
                    combined_stats['frame_category_counts'][bbox['frame_category']] += 1
                    
        return combined_stats

    def _print_combined_report(self) -> None:
        """Форматированный вывод сводного отчета."""
        
        combined_stats = self._generate_combined_stats()
        total_bboxes = combined_stats['total_bboxes']
        total_images = combined_stats['images']['total']

        log.info("\n=======================================================================================================")
        log.info("                                 СВОДНЫЙ ОТЧЕТ ПО ВСЕМ ДАТАСЕТАМ")
        log.info("=======================================================================================================")
        log.info(f"Общее количество уникальных кадров: {total_images}")
        log.info(f"Общее количество валидных BBoxes: {total_bboxes}")
        
        if total_bboxes == 0:
            log.warning("Нет валидных BBoxes для анализа в манифесте.")
            return

        # 1. BBox Size Categories
        log.info("\n-> Сводное распределение BBox по размеру (Tiny/Small/Medium):")

        # Сортировка по минимальному значению (берем из config)
        sorted_sizes = sorted(combined_stats['size_category_counts'].items(), 
                              key=lambda item: config.SIZE_CATEGORIES[item[0]][0])
        
        for cat, count in sorted_sizes:
            log.info(f"  {cat}: {count} ({count/total_bboxes:.2%})")

        # 2. Target Classes
        log.info("\n-> Сводное распределение BBox по основному классу (DRONE/NON_DRONE):")
        for cat, count in sorted(combined_stats['class_counts'].items()):
            log.info(f"  {cat}: {count} ({count/total_bboxes:.2%})")

        # 3. Sub Classes
        log.info("\n-> Сводное распределение BBox по подклассу:")
        for cat, count in sorted(combined_stats['sub_class_counts'].items()):
            log.info(f"  {cat}: {count}")

        # 4. Frame Categories
        log.info("\n-> Сводное распределение кадров по исходному размеру:")
        for cat, count in sorted(combined_stats['frame_category_counts'].items()):
            log.info(f"  {cat}: {count} ({count/total_images:.2%})")
        log.info("=" * 95)

        # 5. Сводная матрица (Класс vs Размер BBox)
        self._print_class_size_matrix(combined_stats)
        log.info("=" * 95)

    def _print_class_size_matrix(self, combined_stats: dict[str, Any]) -> None:
        """Выводит сводную матрицу (Класс vs Размер BBox) только с абсолютными показателями."""
        
        log.info("\n-------------------------------------------------------------------------------------------------------")
        log.info("                             СВОДНАЯ МАТРИЦА: КЛАССЫ vs РАЗМЕР BBOX")
        log.info("-------------------------------------------------------------------------------------------------------")

        # Создаем список всех категорий размеров, включая Large (из всех собранных данных)
        # Сортируем по конфигу, чтобы порядок был логическим
        valid_size_order = sorted(config.SIZE_CATEGORIES.items(), key=lambda item: item[1][0])
        all_sizes = [name for name, _ in valid_size_order]
        all_classes = sorted(combined_stats['class_counts'].keys())
        
        # Создаем заголовок таблицы
        header = ["Класс (строка)"] + all_sizes
        
        # Инициализируем матрицу
        matrix = defaultdict(lambda: defaultdict(int))
        
        # Заполняем матрицу
        for bbox in self.manifest.values(): # self.manifest - это словарь {ds_name: [bboxes]}
            for item in bbox:
                class_name = item['target_class']
                size_name = item['size_category']
                matrix[class_name][size_name] += 1
                
        # Форматирование вывода (Используем простую ASCII-таблицу)
        col_width = max(len(word) for row in [header] for word in row) + 2
        
        # Печать заголовка
        log.info("".join(word.ljust(col_width) for word in header))
        log.info("".join("-" * (col_width) for _ in header))
        
        # Печать строк данных
        for class_name in all_classes:
            row = [class_name.ljust(col_width)]
            for size_name in all_sizes:
                row.append(str(matrix[class_name][size_name]).ljust(col_width))
            log.info("".join(row))
            
        log.info("-" * 95)

    def _print_report(self, ds_name: str, stats: dict) -> None:
        """Форматированный вывод статистики."""

        log.info(f"\n--- ОТЧЕТ ПО ДАТАСЕТУ: {ds_name} ---")
        log.info(f"Общее количество кадров с валидными BBoxes: {stats['images']['total']}")
        log.info(f"Общее количество валидных BBoxes (объектов): {stats['total_bboxes']}")
        
        # 1. BBox Size Categories
        log.info("\n-> Распределение BBox по размеру (Tiny/Small/Medium):")
        for cat, count in sorted(stats['size_category_counts'].items()):
            log.info(f"  {cat}: {count} ({count/stats['total_bboxes']:.2%})")

        # 2. Target Classes
        log.info("\n-> Распределение BBox по основному классу (DRONE/NON_DRONE):")
        for cat, count in sorted(stats['class_counts'].items()):
            log.info(f"  {cat}: {count} ({count/stats['total_bboxes']:.2%})")

        # 3. Sub Classes
        log.info("\n-> Распределение BBox по подклассу:")
        for cat, count in sorted(stats['sub_class_counts'].items()):
            log.info(f"  {cat}: {count}")

        # 4. Frame Categories
        log.info("\n-> Распределение кадров по исходному размеру:")
        for cat, count in sorted(stats['frame_category_counts'].items()):
            log.info(f"  {cat}: {count} ({count/stats['images']['total']:.2%})")
        log.info("-" * 40)