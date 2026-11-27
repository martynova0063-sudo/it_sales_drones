# src/bbox_validator.py

import json
import sys
from pathlib import Path
import config
from src.logger import get_logger
from src.utils import load_json_file, save_json_file, append_to_json_list
from PIL import Image
from src.coco_parser import COCOParser
import imagehash
import hashlib

log = get_logger()

class BBoxValidator:
    """Класс для выполнения полной валидации (RUN) и сбора метаданных BBoxes."""

    def __init__(self, ds_name: str, ds_info: dict, ds_config: dict):
        """
        Инициализирует валидатор для конкретного датасета.

        :param ds_name: Имя датасета.
        :param ds_info: Информация о локальном пути из archive_map.
        :param ds_config: Информация о классах и поднаборах из class_mapping.
        """
        self.ds_name: str = ds_name
        self.ds_info: dict = ds_info
        self.ds_config: dict = ds_config
        self.local_path: Path = Path(ds_info.get('local_path', ''))
        self.target_bboxes: list = [] # Сюда собираем все валидные BBoxes
        self.rejected_stats: dict = {"Oversize": 0, "Invalid_Class": 0, "Invalid_Format": 0}
        self.processed_files_count: int = 0
        self.total_images_processed: int = 0
        self.images_with_valid_bboxes: int = 0
        self.problem_bboxes_to_log: list = []
        self.all_manifest: dict = load_json_file(config.FULL_VALIDATION_MANIFEST_FILE)

        # Инициализация списка для хранения существующих записей, чтобы избежать дублирования
        if self.ds_name not in self.all_manifest:
            self.all_manifest[self.ds_name] = []
        else:
            log.warning(f"Начинается повторная обработка {self.ds_name}. Существующие {len(self.all_manifest[self.ds_name])} записей будут удалены!")
            self.all_manifest[self.ds_name] = [] # Очищаем старые записи для повторной обработки

    def _get_image_size(self, image_path: Path) -> tuple[int, int, str, str, str, float]:
        """
        Читает размер изображения (W, H) и классифицирует размер кадра.

        :param image_path: Абсолютный путь к файлу изображения.
        :return: (W_orig, H_orig, frame_category, p_hash, md5_hash, scale_factor)
        """
        W_orig, H_orig = 0, 0
        p_hash = ""
        md5_hash = ""
        scale_factor = 1.0
        frame_category = "UNKNOWN"

        try:
            with Image.open(image_path) as img:
                W_orig, H_orig = img.size
                p_hash = str(imagehash.phash(img))
                # Расчет MD5-хеша для 100% уникальности
                md5_obj = hashlib.md5()
                with open(image_path, 'rb') as f:
                    md5_obj.update(f.read())
                md5_hash = md5_obj.hexdigest()

                # 1. Расчет коэффициента масштабирования (Scale Factor)
                if W_orig > 0 and H_orig > 0:
                    scale_factor = min(config.TARGET_WIDTH / W_orig, config.TARGET_HEIGHT / H_orig)
                else:
                    scale_factor = 1.0 # Заглушка, если размеры нулевые
                
                # 2. Классификация размера кадра (на основе Scale Factor)
                if scale_factor > 1.05: # Увеличение более чем на 5%
                    frame_category = "SMALLER"
                elif scale_factor < 0.95: # Уменьшение более чем на 5%
                    frame_category = "BIGGER"
                else:
                    frame_category = "TARGET"

        except Exception:
            log.warning(f"[{self.ds_name}]: Не удалось прочитать изображение: {image_path}")
            scale_factor = 1.0
            frame_category = "UNKNOWN"
        
###         # Классификация размера кадра
###         max_side = max(W_orig, H_orig)
###         frame_category = "TARGET"
### 
###         # Логика классификации
###         if max_side > config.TARGET_WIDTH and W_orig > config.TARGET_WIDTH and H_orig > config.TARGET_HEIGHT:
###             frame_category = "LARGE"
###         elif max_side < config.TARGET_WIDTH or H_orig < config.TARGET_HEIGHT:
###              frame_category = "SMALL"

        return W_orig, H_orig, frame_category, p_hash, md5_hash, scale_factor

    def _validate_bbox(self, class_id: str, x_norm: float, y_norm: float, w_norm: float, h_norm: float, W_orig: int, H_orig: int, frame_category: str, image_path: Path, p_hash: str, md5_hash: str, scale_factor: float) -> dict | None:
        """
        Валидация одного BBox, расчет размера по эталону 640x640 и присвоение категории.

        :param class_id: Исходный ID класса (строка).
        :param x_norm: x-центр (нормализованный).
        :param y_norm: y-центр (нормализованный).
        :param w_norm: Ширина BBox (нормализованная).
        :param h_norm: Высота BBox (нормализованная).
        :param W_orig: Исходная ширина кадра.
        :param H_orig: Исходная высота кадра.
        :param frame_category: Категория размера исходного кадра (LARGE, SMALL, TARGET).
        :param image_path: Относительный путь к файлу изображения (для лога проблем).
        :param p_hash: Перцептивный хеш изображения.
        :param md5_hash: Криптографический хеш изображения.
        :param scale_factor: Коэффициент масштабирования до 640x640.
        :return: Словарь с метаданными BBox или None, если BBox отброшен.
        """        
        # 1. Перевод нормализованных размеров в пиксели по эталону 640x480
        W_px = w_norm * config.TARGET_WIDTH
        H_px = h_norm * config.TARGET_HEIGHT
        
        ### Size_px = max(W_px, H_px)
        Size_px_float = max(W_px, H_px)
        Size_px = int(Size_px_float)
        
        # 2. Классификация по размеру (Tiny, Small, Medium)
        size_category = None
        for name, (min_px, max_px) in config.SIZE_CATEGORIES.items():
            if min_px <= Size_px <= max_px:
                size_category = name
                break
        
        # Отбрасываем Large (Size_px > 96)
        if size_category is None:

            # Регистрируем для лога проблемных файлов (Oversize)
            self.problem_bboxes_to_log.append({
                "dataset": self.ds_name, 
                "reason": "Oversize", 
                "size_px": f"{int(W_px)}x{int(H_px)}",
                "max_side_px": int(Size_px_float),
                "bbox": f"{x_norm} {y_norm} {w_norm} {h_norm}", 
                "image_path": str(image_path)
            })

            self.rejected_stats["Oversize"] += 1
            return None 

        # 3. Классификация по классу
        class_map_entry = self.ds_config['class_map'].get(class_id)
        if not class_map_entry:
            log.warning(f"[{self.ds_name}]: Неизвестный class_id '{class_id}'. Пропуск BBox.")

            # Регистрируем для лога проблемных файлов
            self.problem_bboxes_to_log.append({
                "dataset": self.ds_name, "reason": "Invalid_Class", "class_id": class_id, "bbox": f"{x_norm} {y_norm} {w_norm} {h_norm}", "image_path": str(image_path)
            })
            
            self.rejected_stats["Invalid_Class"] += 1
            return None
            
        # 4. Формирование записи
        return {
            "dataset_name": self.ds_name,
            "original_width": W_orig,
            "original_height": H_orig,
            "scale_factor": scale_factor,
            "target_class": class_map_entry['target'],
            "sub_class": class_map_entry['sub_class'],
            "frame_category": frame_category,
            "size_category": size_category,
            "yolo_bbox_norm": f"{class_id} {x_norm} {y_norm} {w_norm} {h_norm}",
            "source_type": "STATIC", # !!! ЗАГЛУШКА: Будет доработано
            "image_path_in_archive": "", # Заполнится в main_run
            "annotation_path_in_archive": "", # Заполнится в main_run
            "p_hash": p_hash,
            "md5_hash": md5_hash,
            "duplicate_status": None
        }

    def _process_yolo_subset(self, subset: dict) -> None:
        """Обрабатывает поднабор в формате YOLO."""
        img_dir = self.local_path / subset.get('image_path_in_archive', '')
        lbl_dir = self.local_path / subset.get('label_path_in_archive', '')
        
        # Предполагаем, что аннотации и картинки совпадают по именам (YOLO стандарт)
        for lbl_file in lbl_dir.glob('*.txt'):
            
            # Проверка ограничения
            if config.MAX_FILES_TO_PROCESS > 0 and self.processed_files_count >= config.MAX_FILES_TO_PROCESS:
                raise StopIteration 

            # Проверка наличия картинки (меняем .txt на .jpg/.png)
            img_file = img_dir / (lbl_file.stem + '.jpg')
            if not img_file.exists():
                img_file = img_dir / (lbl_file.stem + '.png')
                if not img_file.exists():
                    continue # Игнорируем, если нет изображения

            W_orig, H_orig, frame_category, p_hash, md5_hash, scale_factor = self._get_image_size(img_file)

            current_image_bboxes = []
            has_valid_bbox = False
            
            self.total_images_processed += 1 

            with open(lbl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = parts
                        
                        validated_bbox = self._validate_bbox(
                            class_id, float(x), float(y), float(w), float(h), W_orig, H_orig, frame_category, img_file.relative_to(self.local_path), p_hash, md5_hash, scale_factor
                        )

                        if validated_bbox:
                            validated_bbox['image_path_in_archive'] = str(img_file.relative_to(self.local_path))
                            validated_bbox['annotation_path_in_archive'] = str(lbl_file.relative_to(self.local_path))
                            current_image_bboxes.append(validated_bbox)
                            has_valid_bbox = True  

                    else:
                        self.problem_bboxes_to_log.append({
                            "dataset": self.ds_name, "reason": "Invalid_Format", "line": line.strip(), "image_path": str(img_file.relative_to(self.local_path))
                        })
                        self.rejected_stats["Invalid_Format"] += 1

            if has_valid_bbox:
                self.target_bboxes.extend(current_image_bboxes)
                self.images_with_valid_bboxes += 1
            
            self.processed_files_count += 1
    
    def _process_coco_subset(self, subset: dict) -> None:
        """Обрабатывает поднабор в формате COCO."""
        
        parser = COCOParser(self.ds_name, self.local_path, subset.get('label_path_in_archive', ''))
        
        if not parser.data:
            log.error(f"[{self.ds_name}]: Пропуск COCO поднабора из-за ошибки парсинга: {parser.coco_file_path}")
            return
            
        img_dir = self.local_path / subset.get('image_path_in_archive', '')
        
        for image_info in parser.get_image_list():
            
            # Проверка ограничения
            if config.MAX_FILES_TO_PROCESS > 0 and self.processed_files_count >= config.MAX_FILES_TO_PROCESS:
                raise StopIteration 
            
            image_id = image_info['id']
            file_name = image_info['file_name']
            W_orig = image_info.get('width', 0)
            H_orig = image_info.get('height', 0)
            
            img_file_path = img_dir / file_name
            
            if not img_file_path.exists():
                log.warning(f"[{self.ds_name}]: Изображение из COCO JSON не найдено: {img_file_path.relative_to(self.local_path)}")
                continue

            current_image_bboxes = []
            has_valid_bbox = False
            self.total_images_processed += 1 
            
            # W_orig, H_orig берутся из JSON. frame_category - из них же.
            # Если размеры (0, 0) - берем из PIL
            if W_orig == 0 or H_orig == 0:
                W_orig, H_orig, frame_category, p_hash, md5_hash, scale_factor = self._get_image_size(img_file_path)
            else:
                # Расчет scale_factor и frame_category по уже имеющимся размерам
                if W_orig > 0 and H_orig > 0:
                    scale_factor = min(config.TARGET_WIDTH / W_orig, config.TARGET_HEIGHT / H_orig)
                else:
                    scale_factor = 1.0
                
                if scale_factor > 1.05:
                    frame_category = "SMALLER"
                elif scale_factor < 0.95:
                    frame_category = "BIGGER"
                else:
                    frame_category = "TARGET"

                # В COCO-JSON размеры уже есть, но нужно классифицировать кадр
                p_hash = ""
                md5_hash = ""
                try:
                    # Расчет pHash
                    with Image.open(img_file_path) as img:
                        p_hash = str(imagehash.phash(img))
                    # Расчет MD5-хеша
                    md5_obj = hashlib.md5()
                    with open(img_file_path, 'rb') as f:
                        md5_obj.update(f.read())
                    md5_hash = md5_obj.hexdigest()
                except Exception:
                    log.warning(f"[{self.ds_name}]: Не удалось рассчитать хеши: {img_file_path.relative_to(self.local_path)}")

###                 max_side = max(W_orig, H_orig)
###                 frame_category = "TARGET"
###                 if max_side > config.TARGET_WIDTH and W_orig > config.TARGET_WIDTH and H_orig > config.TARGET_HEIGHT:
###                     frame_category = "LARGE"
###                 elif max_side < config.TARGET_WIDTH or H_orig < config.TARGET_HEIGHT:
###                     frame_category = "SMALL"
            
            for ann in parser.get_annotations_for_image(image_id):
                
                # COCO: [x_top_left, y_top_left, w, h] (в пикселях)
                x_tl, y_tl, w_px, h_px = ann.get('bbox', [0, 0, 0, 0])
                class_id = str(ann.get('category_id'))
                
                # 1. Конвертация COCO BBox (пиксели) в YOLO BBox (нормализованный)
                x_center_norm = (x_tl + w_px / 2) / W_orig if W_orig > 0 else 0
                y_center_norm = (y_tl + h_px / 2) / H_orig if H_orig > 0 else 0
                w_norm = w_px / W_orig if W_orig > 0 else 0
                h_norm = h_px / H_orig if H_orig > 0 else 0
                
                # 2. Валидация
                validated_bbox = self._validate_bbox(
                    class_id, x_center_norm, y_center_norm, w_norm, h_norm, W_orig, H_orig, frame_category, img_file_path.relative_to(self.local_path), p_hash, md5_hash, scale_factor
                )

                if validated_bbox:
                    # Добавление путей
                    validated_bbox['image_path_in_archive'] = str(img_file_path.relative_to(self.local_path))
                    validated_bbox['annotation_path_in_archive'] = str(parser.coco_file_path.relative_to(self.local_path))
                    current_image_bboxes.append(validated_bbox)
                    has_valid_bbox = True
                
            # Фиксация результатов для текущего изображения
            if has_valid_bbox:
                self.target_bboxes.extend(current_image_bboxes)
                self.images_with_valid_bboxes += 1
            
            self.processed_files_count += 1

    def process_all_subsets(self) -> int:
        """
        Проходит по всем поднаборам датасета, читает BBoxes, валидирует и сохраняет в манифест.

        :return: Общее количество собранных валидных BBoxes.
        """
        try:
            for subset in self.ds_config['subsets_to_process']:
                # Проверка ограничения (для общих случаев)
                if config.MAX_FILES_TO_PROCESS > 0 and self.processed_files_count >= config.MAX_FILES_TO_PROCESS:
                    log.warning(f"Достигнуто ограничение MAX_FILES_TO_PROCESS ({config.MAX_FILES_TO_PROCESS}). Остановка обработки.")
                    raise StopIteration 

                annotation_format = self.ds_config.get('annotation_format', 'YOLO')
                
                if annotation_format == 'YOLO':
                    self._process_yolo_subset(subset)
                elif annotation_format == 'COCO':
                    self._process_coco_subset(subset)
                else:
                    log.error(f"[{self.ds_name}]: Неизвестный формат аннотаций: {annotation_format}. Пропуск поднабора.")

        except StopIteration:
            pass

        # Сохранение результатов
        self.all_manifest[self.ds_name] = self.target_bboxes
        save_json_file(self.all_manifest, config.FULL_VALIDATION_MANIFEST_FILE)

        # Логируем проблемные BBoxes в отдельный файл
        append_to_json_list(self.problem_bboxes_to_log, config.PROBLEM_BBOX_LOG_FILE)
        
        # Отчет по собранным и отброшенным BBoxes
        total_collected = len(self.target_bboxes)
        rejected_total = sum(self.rejected_stats.values())
        images_lost = self.total_images_processed - self.images_with_valid_bboxes
        
        log.info(f"[{self.ds_name}]: RUN завершен.")
        log.info(f"[{self.ds_name}]: Обработано: {self.total_images_processed} изображений.")
        log.info(f"[{self.ds_name}]: Собрано: {self.images_with_valid_bboxes} изображений с валидными BBoxes.")
        log.info(f"[{self.ds_name}]: Потеряно: {images_lost} изображений (без подходящих BBoxes).")
        log.info(f"[{self.ds_name}]: Всего BBoxes: {total_collected} валидных / {rejected_total} отброшенных. Причины отброса: Oversize={self.rejected_stats['Oversize']}, Invalid_Class={self.rejected_stats['Invalid_Class']}, Invalid_Format={self.rejected_stats['Invalid_Format']}.")

        return len(self.target_bboxes)