# src/coco_parser.py

import sys
from pathlib import Path
from src.logger import get_logger
from src.utils import load_json_file
from collections import defaultdict
from typing import Any

log = get_logger()

class COCOParser:
    """Класс для чтения, индексации и преобразования аннотаций COCO JSON в удобочитаемый формат."""

    def __init__(self, ds_name: str, local_path: Path, label_path_in_archive: str):
        """
        Загружает COCO JSON и создает индексы.

        :param ds_name: Имя датасета.
        :param local_path: Корень датасета.
        :param label_path_in_archive: Путь к _annotations.coco.json
        """
        self.ds_name = ds_name
        self.coco_file_path = local_path / label_path_in_archive
        self.data: dict = load_json_file(self.coco_file_path)
        
        if not self.data or 'images' not in self.data or 'annotations' not in self.data:
            log.error(f"[{ds_name}]: COCO JSON невалиден или пуст: {self.coco_file_path}")
            self.images_map = {}
            self.annotations_by_image = {}
            return

        # Индекс 1: image_id -> image_metadata
        self.images_map: dict = {img['id']: img for img in self.data.get('images', [])}
        
        # Индекс 2: image_id -> [annotation_list]
        self.annotations_by_image: defaultdict = defaultdict(list)
        for ann in self.data.get('annotations', []):
            self.annotations_by_image[ann['image_id']].append(ann)

    def get_image_list(self) -> list[dict]:
        """Возвращает список метаданных всех изображений."""
        return list(self.images_map.values())

    def get_annotations_for_image(self, image_id: int) -> list[dict]:
        """Возвращает список аннотаций для данного image_id."""
        return self.annotations_by_image.get(image_id, [])

    def get_image_info(self, image_id: int) -> dict | None:
        """Возвращает метаданные изображения по его ID."""
        return self.images_map.get(image_id)