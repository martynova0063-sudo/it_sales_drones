# training_model/src/dataset_manager.py

import os
import shutil
import zipfile
import glob
import yaml
import gdown
from collections import defaultdict

from .config import Config
from .logger import Logger

class DatasetManager:
    """
    Управляет загрузкой, извлечением датасетов и их организацией.
    Строго сохраняет исходное разделение на train/val/test из архивов.
    """
    def __init__(self, config: Config, logger: Logger) -> None:
        self.config: Config = config
        self.logger: Logger = logger
        self.data_root: str = os.path.join(config.dataset_base_dir, "combined_dataset")
        
        # Основные пути
        self.images_dir: str = os.path.join(self.data_root, "images")
        self.labels_dir: str = os.path.join(self.data_root, "labels")

        # Целевые директории (стандарт YOLO)
        self.split_dirs = {
            "train": {
                "images": os.path.join(self.images_dir, "train"),
                "labels": os.path.join(self.labels_dir, "train")
            },
            "val": { # Сюда будем класть и 'valid', и 'val'
                "images": os.path.join(self.images_dir, "val"),
                "labels": os.path.join(self.labels_dir, "val")
            },
            "test": {
                "images": os.path.join(self.images_dir, "test"),
                "labels": os.path.join(self.labels_dir, "test")
            }
        }

    def _prepare_dirs(self):
        """Очищает и пересоздает целевые директории."""
        if os.path.exists(self.data_root):
            self.logger.info(f"Очистка старой директории данных: {self.data_root}")
            shutil.rmtree(self.data_root)
        
        for split in self.split_dirs.values():
            os.makedirs(split["images"], exist_ok=True)
            os.makedirs(split["labels"], exist_ok=True)

    def download_and_extract_datasets(self) -> None:
        """Скачивает и распаковывает архивы по их ID."""
        self.logger.info("Начало загрузки и извлечения датасетов.")
        download_folder_path: str = os.path.join(self.config.dataset_base_dir, "downloaded_archives")
        # Убеждаемся, что директория для скачивания архивов существует
        os.makedirs(download_folder_path, exist_ok=True)

        try:
            # Итерируемся по каждому архиву, который мы хотим скачать, используя его ID
            for src_key, archive_info in self.config.dataset_archives.items():
                archive_name = archive_info["name"]
                file_id = archive_info.get("id") # Получаем ID файла из конфига

                # Проверяем, что ID файла был указан
                if not file_id:
                    self.logger.warning(f"ID файла не указан для '{src_key}' ('{archive_name}'). Пропускаем.")
                    continue

                # Формируем ожидаемый путь к скачанному архиву
                expected_archive_path = os.path.join(download_folder_path, archive_name)

                # Проверяем, существует ли уже нужный архив локально
                if not os.path.exists(expected_archive_path):
                    self.logger.info(f"Архив '{archive_name}' не найден локально. Скачивание файла с ID: {file_id}...")
                    
                    try:
                        # --- !!! ПРАВИЛЬНОЕ ИСПОЛЬЗОВАНИЕ gdown.download !!! ---
                        # Формируем правильный URL для скачивания файла по ID
                        # gdown.download ожидает URL вида 'https://drive.google.com/uc?id=FILE_ID'
                        gdown_url = f"https://drive.google.com/uc?id={file_id}"
                        
                        # Скачиваем файл по сформированному URL
                        gdown.download(gdown_url, output=expected_archive_path, quiet=False, use_cookies=False)
                        self.logger.info(f"Скачивание архива '{archive_name}' успешно завершено.")

                    except Exception as e:
                        self.logger.error(f"Ошибка при скачивании архива '{archive_name}' (ID: {file_id}) по URL '{gdown_url}': {e}", exc_info=True)
                        # Если скачивание не удалось, переходим к следующему архиву
                        continue 
                else:
                    # Если архив уже существует, сообщаем об этом и пропускаем скачивание
                    self.logger.info(f"Архив '{archive_name}' уже существует локально. Пропуск скачивания.")

                # --- Теперь работаем с уже скачанным (или существующим) архивом ---
                extract_to = os.path.join(self.config.dataset_base_dir, "extracted", src_key)
                
                # Распаковываем только если директория извлечения еще не существует
                if not os.path.exists(extract_to): 
                    self.logger.info(f"Извлечение '{archive_name}' в '{extract_to}'...")
                    os.makedirs(extract_to, exist_ok=True)
                    try:
                        # Проверяем, что файл действительно существует перед распаковкой
                        if os.path.exists(expected_archive_path):
                            with zipfile.ZipFile(expected_archive_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_to)
                            self.logger.info(f"Извлечение '{archive_name}' успешно завершено.")
                        else:
                            # Это сообщение может появиться, если скачивание файла не удалось
                            self.logger.error(f"Файл '{expected_archive_path}' не найден для извлечения. Скачивание могло завершиться с ошибкой.")
                            continue # Переходим к следующему архиву, если файл не скачался
                    except zipfile.BadZipFile:
                        self.logger.error(f"Ошибка: Архив '{expected_archive_path}' поврежден или не является zip-файлом.")
                        # По желанию: удалить поврежденный файл, чтобы попытаться скачать его заново при следующем запуске
                        # os.remove(expected_archive_path) 
                        continue # Переходим к следующему архиву
                    except Exception as e:
                        self.logger.error(f"Ошибка при распаковке '{expected_archive_path}': {e}", exc_info=True)
                        continue # Переходим к следующему архиву
                else:
                    self.logger.info(f"Папка извлечения '{extract_to}' уже существует. Пропуск распаковки.")

        except Exception as e:
            # Общий обработчик ошибок, если что-то пошло не так в общем цикле
            self.logger.error(f"Произошла непредвиденная ошибка при обработке датасетов: {e}", exc_info=True)
            # Можно либо прервать выполнение, либо продолжить, если возможно.
            # raise # Раскомментируйте, если хотите, чтобы ошибка прерывала программу

    def organize_and_split_data(self, train_ratio=None, val_ratio=None, test_ratio=None) -> None:
        """
        Копирует данные из extracted в combined_dataset, сохраняя исходную структуру папок.
        Аргументы ratio игнорируются, так как мы берем сплиты из исходников.
        """
        self.logger.info("Начало организации данных (сохранение исходных сплитов train/val/test).")
        self._prepare_dirs()

        # Карта соответствия имен папок в исходниках -> нашим целевым папкам
        # 'valid' мапим в 'val'
        split_mapping = {
            "train": "train",
            "val": "val",
            "valid": "val",
            "test": "test"
        }

        for src_key in self.config.dataset_archives.keys():
            src_path = os.path.join(self.config.dataset_base_dir, "extracted", src_key)
            if not os.path.exists(src_path):
                continue

            self.logger.info(f"Обработка источника: {src_key}")
            
            # Пытаемся найти папки train/valid/test внутри src_path
            # Иногда они лежат прямо в корне (src_path/train), иногда вложены (src_path/dataset_name/train)
            # Используем os.walk для поиска папок с именами из split_mapping
            
            found_splits = False
            for root, dirs, files in os.walk(src_path):
                # Проверяем, есть ли в текущей директории папки с именами сплитов
                for dir_name in dirs:
                    if dir_name.lower() in split_mapping:
                        target_split = split_mapping[dir_name.lower()]
                        source_split_dir = os.path.join(root, dir_name)
                        
                        # Внутри сплита ищем images и labels
                        self._process_split_folder(source_split_dir, target_split, src_key)
                        found_splits = True
                
                # Если нашли сплиты на этом уровне, глубже не идем, чтобы не дублировать
                if found_splits:
                    break
            
            if not found_splits:
                self.logger.warning(f"В источнике {src_key} не найдены стандартные папки train/valid/test. Проверьте структуру архива.")

        self._create_data_yaml()
        self.logger.info("Организация данных завершена.")

    def _process_split_folder(self, source_dir: str, target_split_name: str, src_key: str):
        """
        Копирует изображения и метки из одной папки сплита (например, .../src1/valid)
        в целевую папку (combined/val).
        """
        # Обычно внутри лежат папки 'images' и 'labels'
        src_images = os.path.join(source_dir, "images")
        src_labels = os.path.join(source_dir, "labels")

        # Если папок images/labels нет, возможно файлы лежат прямо в корне сплита (редко, но бывает)
        # Но YOLO формат подразумевает images/labels. Проверим.
        if not os.path.exists(src_images):
            # Попробуем найти просто картинки в корне сплита
            image_files = glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.png"))
            if image_files:
                self.logger.warning(f"В {source_dir} нет папки images, но есть файлы. Копируем напрямую (плоская структура).")
                self._copy_bulk(image_files, self.split_dirs[target_split_name]["images"])
                # Ищем метки рядом
                label_files = [f.replace(".jpg", ".txt").replace(".png", ".txt") for f in image_files]
                self._copy_bulk(label_files, self.split_dirs[target_split_name]["labels"])
                return
            else:
                self.logger.warning(f"В папке {source_dir} не найдена папка 'images'. Пропуск.")
                return

        # Копируем images
        images = glob.glob(os.path.join(src_images, "*.*"))
        # Фильтруем только картинки
        images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if images:
            self.logger.info(f"  Копирование {len(images)} изображений из {target_split_name.upper()} ({src_key})...")
            self._copy_bulk(images, self.split_dirs[target_split_name]["images"])
            
            # Копируем labels
            # Надеемся, что они лежат в папке labels рядом
            if os.path.exists(src_labels):
                labels = []
                for img_path in images:
                    basename = os.path.basename(img_path)
                    txt_name = os.path.splitext(basename)[0] + ".txt"
                    labels.append(os.path.join(src_labels, txt_name))
                
                self._copy_bulk(labels, self.split_dirs[target_split_name]["labels"])
            else:
                self.logger.warning(f"  В {source_dir} есть images, но нет папки labels!")

    def _copy_bulk(self, file_list: list[str], dest_dir: str):
        """Вспомогательная функция для массового копирования."""
        for src_path in file_list:
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

    def _create_data_yaml(self) -> None:
        """Создает data.yaml, используя классы из найденного data.yaml архива или из конфига."""
        found_class_legend = None

        # Ищем data.yaml в распакованных архивах
        for src_key in self.config.dataset_archives.keys():
            extracted_path = os.path.join(self.config.dataset_base_dir, "extracted", src_key)
            if os.path.exists(extracted_path):
                # Ищем data.yaml внутри распакованной папки
                potential_data_yaml = os.path.join(extracted_path, "data.yaml")
                if os.path.exists(potential_data_yaml):
                    try:
                        with open(potential_data_yaml, 'r') as f:
                            archive_data_yaml = yaml.safe_load(f)
                            if 'names' in archive_data_yaml:
                                found_class_legend = archive_data_yaml['names']
                                self.logger.info(f"Найдены классы из data.yaml архива '{src_key}': {found_class_legend}")
                                break # Берем первые найденные классы
                    except Exception as e:
                        self.logger.warning(f"Не удалось прочитать data.yaml из архива '{src_key}': {e}")

        # Если классы не найдены в архиве, используем классы из конфига
        if found_class_legend is None:
            self.logger.warning("Классы из data.yaml архива не найдены. Используем классы из config.py.")
            found_class_legend = self.config.class_legend

        # Создаем финальный data.yaml
        data_yaml_content: dict = {
            'path': os.path.abspath(self.data_root),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': found_class_legend # Используем найденные или дефолтные классы
        }
        with open(self.config.data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        self.logger.info(f"Создан data.yaml: {self.config.data_yaml_path} с классами: {found_class_legend}")

    def get_dataset_statistics(self) -> dict:
        """Собирает статистику по созданным папкам."""
        self.logger.info("Сбор статистики...")
        stats = {k: {"images": 0, "objects": defaultdict(int)} for k in self.split_dirs.keys()}
        
        # Обратная карта имен
        id_to_name = self.config.class_legend

        for split, paths in self.split_dirs.items():
            label_dir = paths["labels"]
            if not os.path.exists(label_dir): continue
            
            txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
            stats[split]["images"] = len(txt_files)
            
            for txt_file in txt_files:
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cls_id = int(parts[0])
                                cls_name = id_to_name.get(cls_id, f"unknown_{cls_id}")
                                stats[split]["objects"][cls_name] += 1
                            except ValueError:
                                pass

        self.logger.info("\n--- Статистика датасета ---")
        for split, data in stats.items():
            self.logger.info(f"[{split.upper()}] Изображений: {data['images']}")
            for cls_name, count in data["objects"].items():
                self.logger.info(f"  - {cls_name}: {count}")
        self.logger.info("---------------------------")
        return stats