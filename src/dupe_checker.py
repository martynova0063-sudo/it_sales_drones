# src/dupe_checker.py

import sys
import json
from pathlib import Path
from src.logger import get_logger
from src.utils import load_json_file, save_json_file
import config
from collections import defaultdict
import imagehash
from tqdm import tqdm
from typing import Any, List, Tuple

log = get_logger()

# Тип для ключа изображения (dataset_name, image_path)
ImageKey = Tuple[str, str]

class DupeChecker:
    """Класс для поиска и пометки дубликатов (100% и приблизительных) в манифесте."""

    def __init__(self):
        """Загружает манифест и инициализирует словари."""
        self.manifest: dict = load_json_file(config.FULL_VALIDATION_MANIFEST_FILE)
        
        if not self.manifest:
            log.error("Манифест пуст. Сначала выполните 'run'.")
            sys.exit(1)
            
        self.total_processed_images = 0
        self.total_marked_images = 0
        self.hash_to_images: defaultdict[str, List[ImageKey]] = defaultdict(list)
        self.duplicate_log_data: dict = {
            "exact_duplicates_groups": [],
            "approximate_duplicates_pairs": []
        }
        
        # 1. Агрегация всех записей по pHash
        self._aggregate_hashes()
        self._aggregate_md5_hashes()

    def _aggregate_hashes(self) -> None:
        """Собирает pHash и ключи изображений для всех BBoxes (учитывая только уникальные кадры)."""
        
        for ds_name, bboxes_list in self.manifest.items():
            processed_images_in_ds = set() # Используем для уникальности кадров внутри DS
            
            for bbox in bboxes_list:
                # Кадр идентифицируется по пути и хешу
                image_key = (ds_name, bbox.get('image_path_in_archive', ''))
                p_hash_val = bbox.get('p_hash')
                
                if image_key not in processed_images_in_ds and p_hash_val:
                    self.total_processed_images += 1
                    processed_images_in_ds.add(image_key)
                    # Храним pHash как строку
                    self.hash_to_images[p_hash_val].append(image_key)

    def _aggregate_md5_hashes(self) -> None:
        """Собирает MD5-хеши (для 100% дубликатов)."""
        # Хранит: md5_hash_str -> [(ds_name, image_path), ...]
        self.md5_to_images: defaultdict[str, List[ImageKey]] = defaultdict(list)
        
        for ds_name, bboxes_list in self.manifest.items():
            processed_images_in_ds = set() 
            
            for bbox in bboxes_list:
                image_key = (ds_name, bbox.get('image_path_in_archive', ''))
                md5_hash_val = bbox.get('md5_hash')
                
                if image_key not in processed_images_in_ds and md5_hash_val:
                    processed_images_in_ds.add(image_key)
                    self.md5_to_images[md5_hash_val].append(image_key)

    def _mark_bbox_as_duplicate(self, ds_name: str, image_path: str, status: str) -> None:
        """Находит BBoxes, принадлежащие кадру, и обновляет их поле duplicate_status."""
        
        for bbox in self.manifest[ds_name]:
            if bbox['image_path_in_archive'] == image_path and bbox['duplicate_status'] is None:
                bbox['duplicate_status'] = status
                self.total_marked_images += 1

    def _find_exact_duplicates(self) -> None:
        """Находит и помечает 100% дубликаты (MD5 == MD5) в манифесте (внутри и между DS)."""

        log.info("\n--- Поиск 100% ИДЕНТИЧНЫХ ДУБЛИКАТОВ (MD5 == MD5) ---")
        
        for md5_hash, images in self.md5_to_images.items():
            if len(images) > 1:
                # 1. Регистрируем группу дубликатов

                # Первый элемент в списке - "мастер" (остается null)
                master_image = images[0]
                duplicate_images = images[1:]
                
                # Получаем хеши для лога
                p_hash_master, md5_hash_master = self._get_image_hashes(master_image[0], master_image[1])

                self.duplicate_log_data["exact_duplicates_groups"].append({
                    "md5_hash": md5_hash_master,
                    "p_hash": p_hash_master,
                    "master_image": f"{master_image[0]}/{master_image[1]}",
                    "duplicates": [f"{ds}/{path}" for ds, path in duplicate_images]
                })
                
                # 2. Помечаем каждый файл в манифесте
                for ds_name, image_path in duplicate_images:
                    self._mark_bbox_as_duplicate(ds_name, image_path, "exact_duplicate")
        
        log.info(f"Найдено и помечено {len(self.duplicate_log_data['exact_duplicates_groups'])} групп дубликатов (100%).")

    def _find_approximate_duplicates(self) -> None:
        """Находит и помечает приблизительные дубликаты (pHash Distance <= THRESHOLD) только кросс-датасет."""

        unique_hashes = list(self.hash_to_images.keys())
        hash_objects = {h: imagehash.hex_to_hash(h) for h in unique_hashes}
        
        log.info(f"\n--- Поиск ПРИБЛИЗИТЕЛЬНЫХ ДУБЛИКАТОВ (Dist <= {config.PHASH_DUPE_THRESHOLD}) ---")
        log.warning("Запуск O(N^2) сравнения. Это может занять время...")
        
        hashes_to_check = sorted(hash_objects.items())
        
        for i in tqdm(range(len(hashes_to_check)), desc="Сравнение хешей"):
            hash1_str, hash1_obj = hashes_to_check[i]
            for j in range(i + 1, len(hashes_to_check)):
                hash2_str, hash2_obj = hashes_to_check[j]
                
                distance = hash1_obj - hash2_obj
                
                if distance <= config.PHASH_DUPE_THRESHOLD:
                    images1 = self.hash_to_images[hash1_str]
                    images2 = self.hash_to_images[hash2_str]

                    # Фильтр: Сравниваем только те пары, у которых разные датасеты
                    for ds1, path1 in images1:
                        for ds2, path2 in images2:
                            if ds1 != ds2:
                                # 1. Регистрируем пару
                                p_hash_1, md5_hash_1 = self._get_image_hashes(ds1, path1)
                                p_hash_2, md5_hash_2 = self._get_image_hashes(ds2, path2)

                                self.duplicate_log_data["approximate_duplicates_pairs"].append({
                                    "distance": distance,
                                    "p_hash_1": p_hash_1,
                                    "md5_hash_1": md5_hash_1,
                                    "image1": f"{ds1}/{path1}",
                                    "p_hash_2": p_hash_2,
                                    "md5_hash_2": md5_hash_2,
                                    "image2": f"{ds2}/{path2}"
                                })

                                # 2. Помечаем BBoxes (обе стороны)
                                self._mark_bbox_as_duplicate(ds1, path1, "semi_duplicate")
                                self._mark_bbox_as_duplicate(ds2, path2, "semi_duplicate")

        log.info(f"Найдено и помечено {len(self.duplicate_log_data['approximate_duplicates_pairs'])} пар приблизительных дубликатов (кросс-датасет).")

    def _log_hash_statistics(self) -> None:
        """Сбор и вывод статистики по уникальным хешам (MD5 и pHash) по датасетам и их объединениям."""
        
        log.info("\n--- СТАТИСТИКА УНИКАЛЬНЫХ ХЕШЕЙ (КАДРОВ) ---")
        
        ds_hashes = defaultdict(lambda: {"md5": set(), "phash": set(), "images": set()})
        global_md5 = set()
        global_phash = set()
        
        for ds_name, bboxes_list in self.manifest.items():
            for bbox in bboxes_list:
                image_key = (bbox.get('image_path_in_archive', ''), bbox.get('original_width'), bbox.get('original_height'))
                md5 = bbox.get('md5_hash')
                phash = bbox.get('p_hash')
                
                # Собираем данные по датасету (учитываем только 1 раз на уникальный кадр внутри DS)
                if image_key not in ds_hashes[ds_name]["images"]:
                    ds_hashes[ds_name]["images"].add(image_key)
                    if md5:
                        ds_hashes[ds_name]["md5"].add(md5)
                        global_md5.add(md5)
                    if phash:
                        ds_hashes[ds_name]["phash"].add(phash)
                        global_phash.add(phash)

        # Вывод статистики по каждому датасету
        ds_names = sorted(ds_hashes.keys())
        log.info(f"{'Датасет':<25}{'Всего Кадров':>15}{'Уник. MD5':>15}{'Уник. pHash':>15}")
        log.info("-" * 70)
        for ds_name in ds_names:
            data = ds_hashes[ds_name]
            log.info(f"{ds_name:<25}{len(data['images']):>15}{len(data['md5']):>15}{len(data['phash']):>15}")
            
        # Вывод объединенной статистики
        log.info("-" * 70)
        log.info(f"Общее уникальное количество кадров: {self.total_processed_images}")
        log.info(f"Общее уникальное MD5 (по всем DS): {len(global_md5)}")
        log.info(f"Общее уникальное pHash (по всем DS): {len(global_phash)}")
        log.info("-" * 70)
        
        # Анализ пересечения MD5 (100% дубликаты)
        exact_dupe_count = sum(len(group.get('duplicates', [])) for group in self.duplicate_log_data.get('exact_duplicates_groups', []))
        log.info(f"  -> Файлов, помеченных как 100% дубликаты (MD5): {exact_dupe_count}")
        log.info(f"  -> Если {len(global_md5)} == {self.total_processed_images} и дубликаты == 0, то все кадры уникальны.")

    def _get_image_hashes(self, ds_name: str, image_path: str) -> Tuple[str, str]:
        """Вспомогательный метод для получения хешей изображения по его пути."""
        # Ищем любой BBox, принадлежащий этому изображению, и возвращаем его хеши
        if ds_name in self.manifest:
            for bbox in self.manifest[ds_name]:
                if bbox['image_path_in_archive'] == image_path:
                    return bbox.get('p_hash', ''), bbox.get('md5_hash', '')
        return '', ''

    def run_dupe_check(self) -> None:
        """Основной метод для запуска поиска и пометки дубликатов."""
        
        log.info("Режим DOUBLES: Поиск и пометка дубликатов в манифесте.")
        
        # 1. Поиск и пометка 100% дубликатов
        self._find_exact_duplicates()
        
        # 2. Поиск и пометка приблизительных дубликатов (если разрешено)
        if config.ENABLE_APPROXIMATE_DUPE_CHECK:
            self._find_approximate_duplicates()
        else:
            log.warning("Поиск приблизительных дубликатов (pHash) отключен в конфиге (ENABLE_APPROXIMATE_DUPE_CHECK = False).")
            
        # 3. Сохранение обновленного манифеста
        save_json_file(self.manifest, config.FULL_VALIDATION_MANIFEST_FILE)
        
        # 4. Сохранение лога дубликатов
        save_json_file(self.duplicate_log_data, config.DUPLICATE_LOG_FILE)
        
        # 5. Вывод статистики по хешам
        self._log_hash_statistics()

        log.info(f"\n*** DOUBLES: Обработка завершена. ***")
        log.info(f"Обработано уникальных кадров: {self.total_processed_images}")
        log.info(f"Всего BBoxes помечено как дубликаты: {self.total_marked_images}")
        log.info(f"Подробный отчет сохранен в: {config.DUPLICATE_LOG_FILE}")