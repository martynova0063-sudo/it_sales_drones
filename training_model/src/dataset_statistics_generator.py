# training_model/src/dataset_statistics_generator.py

import os
import glob
import yaml
import cv2
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from .config import Config
from .logger import Logger

class DatasetStatisticsGenerator:
    """
    Класс для генерации подробной статистики датасета:
    - Общее количество изображений и объектов.
    - Количество объектов по классам.
    - Количество объектов по категориям размеров (микро, маленький, средний и т.д.).
    Статистика может быть сгенерирована для train, val, test и всего датасета.
    """
    def __init__(self, config: Config, logger: Logger) -> None:
        """
        Инициализирует генератор статистики датасета.

        Args:
            config: Объект конфигурации.
            logger: Объект логгера.
        """
        self.config: Config = config
        self.logger: Logger = logger
        self.data_yaml_content: dict = {}
        self.dataset_root: str = ""
        self.model_names: list[str] = []
        self._load_data_yaml()

    def _load_data_yaml(self) -> None:
        """Загружает data.yaml для получения путей к данным и имен классов."""
        try:
            with open(self.config.data_yaml_path, 'r') as f:
                self.data_yaml_content = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Ошибка чтения файла data.yaml: {e}", exc_info=True)
            raise FileNotFoundError(f"Файл data.yaml не найден или поврежден: {self.config.data_yaml_path}")

        self.dataset_root = self.data_yaml_content.get('path', '')
        self.model_names = self.data_yaml_content.get('names', [])

        if not self.dataset_root or not self.model_names:
            self.logger.error(f"Не удалось найти 'path' или 'names' в {self.config.data_yaml_path}.")
            raise ValueError(f"Неполный data.yaml: {self.config.data_yaml_path}")

    def _get_object_size_category(self, obj_width_px: int, obj_height_px: int) -> str:
        """
        Определяет категорию размера объекта по наибольшему измерению его bounding box'а в пикселях.

        Args:
            obj_width_px: Ширина объекта в пикселях.
            obj_height_px: Высота объекта в пикселях.

        Returns:
            Строковое название категории размера (например, "Micro", "Small", "XLarge").
        """
        max_dim = max(obj_width_px, obj_height_px)
        for category, (min_px, max_px) in self.config.size_categories.items():
            if min_px <= max_dim <= max_px:
                return category
        return "XLarge" # Если размер больше максимального определенного

    def _plot_class_distribution(self, stats: dict, title: str, output_file_path: str) -> None:
        """
        Строит и сохраняет гистограмму распределения объектов по классам.

        Args:
            stats: Словарь со статистикой (должен содержать 'objects_by_class').
            title: Заголовок для графика.
            output_file_path: Путь к файлу для сохранения графика.
        """
        class_counts = stats["objects_by_class"]
        if not class_counts:
            self.logger.warning(f"Нет данных для построения графика распределения классов для '{title}'.")
            return

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(classes, counts, color='skyblue')
        ax.set_title(title)
        ax.set_xlabel("Класс")
        ax.set_ylabel("Количество объектов")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_file_path)
        plt.close(fig)
        self.logger.info(f"График распределения классов сохранен в: {output_file_path}")

    def _plot_size_distribution(self, stats: dict, title: str, output_file_path: str) -> None:
        """
        Строит и сохраняет столбчатую диаграмму с накоплением для распределения объектов по размерам
        для каждого класса.

        Args:
            stats: Словарь со статистикой (должен содержать 'objects_by_size').
            title: Заголовок для графика.
            output_file_path: Путь к файлу для сохранения графика.
        """
        objects_by_size = stats["objects_by_size"]
        if not objects_by_size:
            self.logger.warning(f"Нет данных для построения графика распределения размеров для '{title}'.")
            return

        all_classes = sorted(objects_by_size.keys())
        size_categories = list(self.config.size_categories.keys())

        # Подготовка данных для stacked bar chart
        data_for_plot = {cat: [0] * len(all_classes) for cat in size_categories}
        for i, cls_name in enumerate(all_classes):
            for size_cat in size_categories:
                data_for_plot[size_cat][i] = objects_by_size.get(cls_name, {}).get(size_cat, 0)

        fig, ax = plt.subplots(figsize=(12, 7))

        bottom = np.zeros(len(all_classes))
        for size_cat in size_categories:
            counts = data_for_plot[size_cat]
            ax.bar(all_classes, counts, bottom=bottom, label=size_cat)
            bottom += counts

        ax.set_title(title)
        ax.set_xlabel("Класс")
        ax.set_ylabel("Количество объектов")
        ax.legend(title="Категория размера")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_file_path)
        plt.close(fig)
        self.logger.info(f"График распределения размеров сохранен в: {output_file_path}")

    def _process_split_data(self, split_name: str) -> dict:
        """
        Обрабатывает данные для заданного сплита (train, val, test) и собирает статистику.

        Args:
            split_name: Имя сплита ('train', 'val', 'test').

        Returns:
            Словарь со статистикой для данного сплита:
            {
                "images": int,
                "objects_by_class": {class_name: count, ...},
                "objects_by_size": {class_name: {size_category: count, ...}}
            }
        """
        split_images_relative_path = self.data_yaml_content.get(split_name)
        if not split_images_relative_path:
            self.logger.warning(f"Относительный путь для '{split_name}' изображений не найден в data.yaml.")
            return {"images": 0, "objects_by_class": defaultdict(int), "objects_by_size": defaultdict(lambda: defaultdict(int))}

        split_images_path = os.path.join(self.dataset_root, split_images_relative_path)
        split_labels_path = os.path.join(self.dataset_root, split_images_relative_path.replace("images", "labels"))
        
        if not os.path.exists(split_images_path):
            self.logger.warning(f"Директория изображений для '{split_name}' не найдена: {split_images_path}")
            return {"images": 0, "objects_by_class": defaultdict(int), "objects_by_size": defaultdict(lambda: defaultdict(int))}
        if not os.path.exists(split_labels_path):
            self.logger.warning(f"Директория меток для '{split_name}' не найдена: {split_labels_path}")
            return {"images": 0, "objects_by_class": defaultdict(int), "objects_by_size": defaultdict(lambda: defaultdict(int))}

        stats = {
            "images": 0,
            "objects_by_class": defaultdict(int),
            "objects_by_size": defaultdict(lambda: defaultdict(int))
        }

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(split_images_path, '**', ext), recursive=True))
        
        stats["images"] = len(image_files)

        self.logger.info(f"Сбор статистики для {split_name.upper()}: {len(image_files)} изображений.")
        for img_path in image_files:
            img_basename, _ = os.path.splitext(os.path.basename(img_path))
            label_file_path = os.path.join(split_labels_path, img_basename + ".txt")

            if os.path.exists(label_file_path):
                img = cv2.imread(img_path)
                if img is None:
                    self.logger.warning(f"Не удалось загрузить изображение для получения размеров: {img_path}")
                    continue
                h, w, _ = img.shape

                with open(label_file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                x_c_norm, y_c_norm, w_norm, h_norm = map(float, parts[1:5])

                                class_name = self.model_names[cls_id] if cls_id < len(self.model_names) else f"unknown_{cls_id}"
                                stats["objects_by_class"][class_name] += 1

                                # Денормализация размеров объекта для определения категории размера
                                obj_width_px = int(w_norm * w)
                                obj_height_px = int(h_norm * h)
                                size_category = self._get_object_size_category(obj_width_px, obj_height_px)
                                stats["objects_by_size"][class_name][size_category] += 1
                            except ValueError as e:
                                self.logger.warning(f"Некорректный формат строки в файле метки {label_file_path}: {line.strip()} - {e}")
            # else: # Если нет файла меток, то и объектов нет, пропускаем
            #     self.logger.debug(f"Файл метки не найден для {img_basename}.txt")

        return stats

    def _print_statistics(self, stats: dict, title: str, output_file_path: str) -> None:
        """
        Выводит собранную статистику в логгер и сохраняет в файл.

        Args:
            stats: Словарь со статистикой.
            title: Заголовок для вывода статистики.
            output_file_path: Путь к файлу для сохранения статистики.
        """
        with open(output_file_path, 'w', encoding='utf-8') as f:
            def log_and_write(message: str) -> None:
                self.logger.info(message)
                f.write(message + "\n")

            log_and_write(f"\n--- {title} Статистика Датасета ---")
            log_and_write(f"Всего изображений: {stats['images']}")
            log_and_write("\nОбъектов по классам:")
            if stats["objects_by_class"]:
                for cls_name, count in sorted(stats["objects_by_class"].items()):
                    log_and_write(f"  - {cls_name}: {count}")
            else:
                log_and_write("  Нет объектов в этом сплите.")

            log_and_write("\nОбъектов по классам и размерам:")
            if stats["objects_by_size"]:
                # Формируем заголовок таблицы размеров
                size_header_cols = ["Class\\Size"] + list(self.config.size_categories.keys()) + ["Total"]
                size_col_widths = [max(len(h), 10) for h in size_header_cols]
                size_header_str = " ".join(f"{h:<{size_col_widths[idx]}}" for idx, h in enumerate(size_header_cols))
                log_and_write(size_header_str)
                log_and_write("-" * (sum(size_col_widths) + len(size_col_widths) - 1))

                for cls_name in sorted(stats["objects_by_class"].keys()): # Итерируемся по классам, чтобы сохранить порядок
                    class_size_data = stats["objects_by_size"].get(cls_name, {})
                    row_data = [cls_name]
                    total_for_class = 0
                    for size_cat in self.config.size_categories.keys():
                        count = class_size_data.get(size_cat, 0)
                        row_data.append(str(count))
                        total_for_class += count
                    row_data.append(str(total_for_class))
                    
                    row_str = [f"{row_data[0]:<{size_col_widths[0]}}"] + \
                              [f"{val:<{size_col_widths[j+1]}}" for j, val in enumerate(row_data[1:]) if j+1 < len(size_col_widths)]
                    log_and_write(" ".join(row_str))
            else:
                log_and_write("  Нет объектов в этом сплите.")
            log_and_write("------------------------------------")
        self.logger.info(f"Подробная статистика сохранена в: {output_file_path}")

        # Построение и сохранение графиков
        class_plot_path = getattr(self.config, f"{title.split(' ')[0].lower()}_class_plot", None)
        size_plot_path = getattr(self.config, f"{title.split(' ')[0].lower()}_size_plot", None)
        
        if class_plot_path:
            self._plot_class_distribution(stats, f"Распределение объектов по классам ({title})", class_plot_path)
        if size_plot_path:
            self._plot_size_distribution(stats, f"Распределение объектов по размерам ({title})", size_plot_path)

    def generate_statistics(self) -> None:
        """
        Генерирует и выводит статистику для train, val, test и всего датасета.
        """
        all_stats = {
            "images": 0,
            "objects_by_class": defaultdict(int),
            "objects_by_size": defaultdict(lambda: defaultdict(int))
        }
        
        splits_to_process = ["train", "val", "test"]
        
        for split_name in splits_to_process:
            split_stats = self._process_split_data(split_name)
            self._print_statistics(split_stats, f"{split_name.upper()} Сплит", getattr(self.config, f"{split_name}_stats_file"))

            # Агрегируем для общей статистики
            all_stats["images"] += split_stats["images"]
            for cls_name, count in split_stats["objects_by_class"].items():
                all_stats["objects_by_class"][cls_name] += count
            for cls_name, size_counts in split_stats["objects_by_size"].items():
                for size_cat, count in size_counts.items():
                    all_stats["objects_by_size"][cls_name][size_cat] += count

        self._print_statistics(all_stats, "Общая Статистика (Все Сплиты)", self.config.overall_stats_file)

        # Отдельные вызовы для графиков общей статистики, так как _print_statistics 
        # использует title для поиска пути, а "Общая Статистика (Все Сплиты)" не совпадает
        # с префиксом "overall_" в именах полей config.
        self._plot_class_distribution(all_stats, "Общая Статистика: Распределение объектов по классам", self.config.overall_class_plot)
        self._plot_size_distribution(all_stats, "Общая Статистика: Распределение объектов по размерам", self.config.overall_size_plot)