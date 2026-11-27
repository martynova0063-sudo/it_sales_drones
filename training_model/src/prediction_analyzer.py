# training_model/src/prediction_analyzer.py

import os
import shutil
import glob
import yaml
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm
import csv

from .config import Config
from .logger import Logger

class PredictionJournal:
    """
    Класс для ведения журнала подробной информации о False Negatives и False Positives.
    Сохраняет данные в CSV-файл.
    """
    def __init__(self, config: Config, logger: Logger, class_names: list[str]) -> None:
        """
        Инициализирует журнал предсказаний.

        Args:
            config: Объект конфигурации.
            logger: Объект логгера.
            class_names: Список имен классов, используемых моделью.
        """
        self.config: Config = config
        self.logger: Logger = logger
        self.class_names: list[str] = class_names
        self.journal_path: str = self.config.prediction_journal_path
        self._write_header()

    def _write_header(self) -> None:
        """Записывает заголовок CSV-файла, если файл новый или пустой."""
        try:
            with open(self.journal_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Тип Ошибки", "Имя Файла",
                    "GT Класс ID", "GT Имя Класса", "GT x_center", "GT y_center", "GT width", "GT height",
                    "Pred Класс ID", "Pred Имя Класса", "Pred x_center", "Pred y_center", "Pred width", "Pred height"
                ])
            self.logger.info(f"Журнал предсказаний создан: {self.journal_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при создании журнала предсказаний: {e}", exc_info=True)

    def _log_entry(
        self,
        error_type: str,
        img_filename: str,
        gt_cls_id: int | None,
        gt_box_norm: list[float] | None,
        pred_cls_id: int | None,
        pred_box_norm: list[float] | None
    ) -> None:
        """
        Записывает одну запись об ошибке в журнал.

        Args:
            error_type: Тип ошибки ("FN" или "FP").
            img_filename: Имя файла изображения.
            gt_cls_id: ID истинного класса объекта (None, если нет).
            gt_box_norm: Нормализованные координаты истинного bounding box [x_c, y_c, w, h] (None, если нет).
            pred_cls_id: ID предсказанного класса объекта (None, если нет).
            pred_box_norm: Нормализованные координаты предсказанного bounding box [x_c, y_c, w, h] (None, если нет).
        """
        row_data = [error_type, img_filename]

        if gt_cls_id is not None and gt_box_norm is not None:
            gt_class_name = self.class_names[gt_cls_id] if gt_cls_id < len(self.class_names) else f"Class_{gt_cls_id}"
            row_data.extend([gt_cls_id, gt_class_name] + [f"{coord:.4f}" for coord in gt_box_norm])
        else:
            row_data.extend(["", "", "", "", "", ""]) # Заполняем прочерками

        if pred_cls_id is not None and pred_box_norm is not None:
            pred_class_name = self.class_names[pred_cls_id] if pred_cls_id < len(self.class_names) else f"Class_{pred_cls_id}"
            row_data.extend([pred_cls_id, pred_class_name] + [f"{coord:.4f}" for coord in pred_box_norm])
        else:
            row_data.extend(["", "", "", "", "", ""]) # Заполняем прочерками
            
        try:
            with open(self.journal_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as e:
            self.logger.error(f"Ошибка при записи в журнал предсказаний для файла {img_filename}: {e}", exc_info=True)

    def log_false_negative(self, img_filename: str, true_cls_id: int, true_box_norm: list[float]) -> None:
        """
        Логирует случай False Negative (пропущенный истинный объект).
        """
        self._log_entry("FN", img_filename, true_cls_id, true_box_norm, None, None)
        self.logger.debug(f"FN: Файл={img_filename}, GT Класс ID={true_cls_id}, GT BBox={true_box_norm}")

    def log_false_positive(self, img_filename: str, pred_cls_id: int, pred_box_norm: list[float], gt_cls_id: int | None = None, gt_box_norm: list[float] | None = None) -> None:
        """
        Логирует случай False Positive (ложное срабатывание).

        Args:
            img_filename: Имя файла изображения.
            pred_cls_id: ID предсказанного класса ложного срабатывания.
            pred_box_norm: Нормализованные координаты предсказанного bounding box [x_c, y_c, w, h].
            gt_cls_id: ID истинного класса, если FP связано с GT объектом (опционально).
            gt_box_norm: Нормализованные координаты GT bounding box, если FP связано с GT объектом (опционально).
        """
        # Передаем все полученные параметры в _log_entry
        self._log_entry("FP", img_filename, gt_cls_id, gt_box_norm, pred_cls_id, pred_box_norm)
        self.logger.debug(f"FP: Файл={img_filename}, Pred Класс ID={pred_cls_id}, Pred BBox={pred_box_norm}, GT ID={gt_cls_id}, GT BBox={gt_box_norm}")

class PredictionAnalyzer:
    def __init__(self, config: Config, logger: Logger) -> None:
        self.config: Config = config
        self.logger: Logger = logger
        self.model: YOLO | None = None
        self._load_model()
        self.analysis_results_dir: str = os.path.join(self.config.output_dir, "prediction_analysis")
        os.makedirs(self.analysis_results_dir, exist_ok=True)

        # Инициализация и очистка корневой директории для всех результатов
        if os.path.exists(self.config.global_results_dir):
            self.logger.info(f"Очистка существующей общей директории результатов: {self.config.global_results_dir}")
            shutil.rmtree(self.config.global_results_dir)
        os.makedirs(self.config.global_results_dir, exist_ok=True)
        
        self.results_dir: str = self.config.prediction_visualizations_dir # Путь для сохранения изображений
        os.makedirs(self.results_dir, exist_ok=True)

        # Создание директорий для False Negatives и False Positives
        os.makedirs(self.config.false_negatives_output_dir, exist_ok=True)
        os.makedirs(self.config.false_positives_output_dir, exist_ok=True)

    def _load_model(self) -> None:
        if os.path.exists(self.config.best_model_path):
            self.logger.info(f"Загрузка сохраненной модели для анализа: {self.config.best_model_path}")
            self.model = YOLO(self.config.best_model_path)
        else:
            self.logger.info(f"Загрузка базовой модели для анализа: {self.config.base_model_name}")
            self.model = YOLO(self.config.base_model_name)

    def _get_iou(self, box1: list[float], box2: list[float]) -> float:
        """Вычисляет IoU (Intersection over Union) между двумя bounding box'ами."""
        # box format: [x_center, y_center, width, height]
        # Преобразуем в [x_min, y_min, x_max, y_max]
        
        def convert_to_xyxy(box):
            x_c, y_c, w, h = box
            w_half, h_half = w / 2, h / 2
            x_min = x_c - w_half
            y_min = y_c - h_half
            x_max = x_c + w_half
            y_max = y_c + h_half
            return [x_min, y_min, x_max, y_max]

        box1_xyxy = convert_to_xyxy(box1)
        box2_xyxy = convert_to_xyxy(box2)

        x1_min, y1_min, x1_max, y1_max = box1_xyxy
        x2_min, y2_min, x2_max, y2_max = box2_xyxy

        # Вычисляем координаты пересечения
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Вычисляем площадь пересечения
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        intersection_area = inter_width * inter_height

        # Вычисляем площади bounding box'ов
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Вычисляем площадь объединения
        union_area = box1_area + box2_area - intersection_area

        # Вычисляем IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def _draw_boxes(self, image: np.ndarray, boxes: list[dict], is_ground_truth: bool, class_names: list[str], image_dims: tuple[int, int]) -> np.ndarray:
        """
        Рисует bounding box'ы и метки на изображении.

        Args:
            image: Изображение OpenCV (numpy array).
            boxes: Список словарей с информацией о боксах и классах.
                   Формат: [{"class_id": int, "box": [x_center_norm, y_center_norm, width_norm, height_norm]}, ...]
            is_ground_truth: True, если это истинные метки (GT); False, если предсказания.
            class_names: Список имен классов.
            image_dims: Кортеж (ширина, высота) изображения в пикселях (w, h).

        Returns:
            Изображение с нарисованными боксами.
        """
        img_width, img_height = image_dims
        # Цвета BGR
        color = (0, 255, 0) if is_ground_truth else (0, 0, 255) # Зеленый для GT, Красный для Pred
        label_prefix = "GT" if is_ground_truth else "Pred"

        for box_info in boxes:
            cls_id = box_info["class_id"]
            x_c_norm, y_c_norm, w_norm, h_norm = box_info["box"]
            confidence = box_info.get("confidence") # Получаем уверенность, если она есть

            # Денормализация координат
            x_center_px = int(x_c_norm * img_width)
            y_center_px = int(y_c_norm * img_height)
            width_px = int(w_norm * img_width)
            height_px = int(h_norm * img_height)

            x_min_px = int(x_center_px - width_px / 2)
            y_min_px = int(y_center_px - height_px / 2)
            x_max_px = int(x_center_px + width_px / 2)
            y_max_px = int(y_center_px + height_px / 2)

            # Рисуем прямоугольник
            cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), color, 2)

            # Рисуем метку
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown_{cls_id}"

            if confidence is not None:
                label_text = f"{label_prefix}:{cls_id} {class_name} ({confidence:.2f})" # Добавляем уверенность
            else:
                label_text = f"{label_prefix}:{cls_id} {class_name}"
            
            # Определяем размер текста для подложки
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Позиция текста (попытка разместить над боксом с небольшим отступом)
            text_x = x_min_px
            text_y = y_min_px - 5

            # Если текст выходит за верхнюю границу изображения, или слишком близко к боксу,
            # размещаем его внутри бокса, чуть ниже верхнего края.
            # Отступ от верхнего края изображения или бокса
            margin = 5

            if (text_y - text_height - baseline) < margin or (y_min_px < text_height + baseline + margin): # Если метка вылезает вверх или бокс слишком маленький сверху
                text_y = y_min_px + text_height + margin # Размещаем внутри бокса, отступив от верхнего края
                # Если все равно выходит за изображение, тогда просто сверху изображения
                if text_y + baseline > img_height:
                    text_y = img_height - baseline - margin # Размещаем внизу изображения, если нет места вверху

            # Корректировка text_x, чтобы не выходил за горизонтальные границы изображения
            if text_x < 0:
                text_x = 0
            # Убедимся, что фон метки не выходит за правую границу изображения
            if text_x + text_width > img_width:
                text_x = img_width - text_width
            
            # Рисуем подложку для текста
            cv2.rectangle(image, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), color, -1)
            # Рисуем текст
            cv2.putText(image, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image

    def analyze_predictions(self) -> None:
        """
        Выполняет инференс, сравнивает предсказания с метками, собирает статистику
        и сортирует результаты по папкам.
        """
        if self.model is None:
            self.logger.error("Модель не загружена. Анализ невозможен.")
            return
        if not os.path.exists(self.config.data_yaml_path):
             self.logger.error(f"Файл data.yaml не найден по пути {self.config.data_yaml_path}. Анализ невозможен.")
             return

        # Читаем data.yaml для получения путей и имен классов
        try:
            with open(self.config.data_yaml_path, 'r') as f:
                data_yaml_content: dict = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Ошибка чтения файла data.yaml: {e}", exc_info=True)
            return

        dataset_root: str | None = data_yaml_content.get('path')
        test_images_relative_path: str | None = data_yaml_content.get('test')
        model_names = data_yaml_content.get('names')

        if not dataset_root or not test_images_relative_path or not model_names:
            self.logger.error(f"Не удалось найти 'path', 'test' или 'names' в {self.config.data_yaml_path}.")
            return

        # Инициализация журнала предсказаний, теперь, когда model_names доступен
        self.prediction_journal = PredictionJournal(self.config, self.logger, model_names)

        test_images_path: str = os.path.join(dataset_root, test_images_relative_path)
        
        if not os.path.exists(test_images_path):
            self.logger.error(f"Тестовая директория изображений не найдена: {test_images_path}.")
            return

        # --- Определяем ПОЛНЫЕ пути для сохранения результатов анализа ---
        analysis_output_dir = os.path.join(self.config.output_dir, "prediction_analysis")
        os.makedirs(analysis_output_dir, exist_ok=True)

        # --- Статистика ---
        # Инициализируем счетчики для Confusion Matrix и классификации по размерам
        # Формат: {class_id: {predicted_class_id: count, ...}, ...}
        confusion_matrix_counts = defaultdict(lambda: defaultdict(int))
        # Формат: {class_id: {size_category: count, ...}, ...}
        size_classification = defaultdict(lambda: defaultdict(int))
        
        # Переменная для подсчета пропущенных объектов
        missed_objects = defaultdict(int)
        # Переменная для подсчета ложных срабатываний на фон
        false_positives_on_background = defaultdict(int)

        # --- Читаем все файлы разметки ---
        # Правильно формируем путь к директории с тестовыми метками
        # test_images_relative_path = "images/test"
        # Меняем "images/test" на "labels/test"
        test_labels_relative_path = test_images_relative_path.replace("images", "labels")
        test_labels_path = os.path.join(dataset_root, test_labels_relative_path)

        # Получаем список всех изображений
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(test_images_path, '**', ext), recursive=True))

        self.logger.info(f"Начинаем анализ для {len(image_files)} изображений...")


        ## НАЧАЛО ИЗМЕНЯЕМОГО БЛОКА
        # Создаем подпапки для классов в False Negatives и False Positives
        for class_name in model_names:
            os.makedirs(os.path.join(self.config.false_negatives_output_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(self.config.false_positives_output_dir, class_name), exist_ok=True)

        # --- Обработка каждого изображения ---
        for img_path in tqdm(image_files, desc="Анализ предсказаний"):
            img_filename = os.path.basename(img_path)
            img_basename, _ = os.path.splitext(img_filename)
            
            # Ищем соответствующий файл метки
            label_file_path = os.path.join(test_labels_path, img_basename + ".txt")
            
            true_boxes = []
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cls_id = int(parts[0])
                                # YOLO format: [class_id, x_center, y_center, width, height]
                                box_data = [float(p) for p in parts[1:]]
                                true_boxes.append({"class_id": cls_id, "box": box_data})
                            except ValueError:
                                continue # Пропускаем строки с неверным форматом
###                 self.logger.debug(f"Изображение {img_filename}: Найдено {len(true_boxes)} истинных меток.")
###             else:
###                 self.logger.debug(f"Изображение {img_filename}: Файл метки не найден: {label_file_path}")

            # Делаем предсказание для изображения
            try:
                preds_results = self.model.predict(
                    source=img_path,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    save=False,
                    save_txt=False,
                    project=self.analysis_results_dir,
                    name="temp_preds",
                    exist_ok=True,
                    verbose=False,
                )

                predicted_boxes = []
                if preds_results and preds_results[0].boxes:
                    for i, pred_box_xywhn in enumerate(preds_results[0].boxes.xywhn):
                        confidence = float(preds_results[0].boxes.conf[i])
                        predicted_boxes.append({"class_id": int(preds_results[0].boxes.cls[i]), "box": pred_box_xywhn.tolist(), "confidence": confidence})
                        
                # --- Сравнение предсказаний с метками ---
                # Отмечаем, какие истинные объекты были найдены
                matched_true_objects = [False] * len(true_boxes)
                
                # Списки для сбора классов FN/FP для текущего изображения
                current_image_false_negatives_classes: list[int] = []
                current_image_false_positives_classes: list[int] = []

                # Заполняем confusion_matrix_counts и size_classification
                for pred_idx, pred in enumerate(predicted_boxes):
                    pred_cls_id = pred["class_id"]
                    pred_box = pred["box"]
                    
                    best_iou = -1
                    best_true_match_idx = -1
                    
                    self.logger.debug(f"Изображение {img_filename}, Предсказание {pred_idx}: Класс {pred_cls_id}, BBox {pred_box}")

                    # Ищем наиболее подходящий истинный объект для данного предсказания
                    for i, true_obj in enumerate(true_boxes):
                        self.logger.debug(f"  Сравнение с GT {i}: Класс {true_obj['class_id']}, BBox {true_obj['box']}, matched={matched_true_objects[i]}")

                        if not matched_true_objects[i]:
                            iou = self._get_iou(pred_box, true_obj["box"])
                            self.logger.debug(f"    IoU с GT {i}: {iou:.4f}")                            

                            if iou > best_iou and iou > 0.5:
                                best_iou = iou
                                best_true_match_idx = i

                    if best_true_match_idx != -1:
                        true_obj_match = true_boxes[best_true_match_idx]
                        true_cls_id = true_obj_match["class_id"]
                        self.logger.debug(f"  Предсказание {pred_idx} МАТЧИРОВАНО с GT {best_true_match_idx} (IoU={best_iou:.4f}). Классифицировано как TP.")
                        
                        # Обновляем Confusion Matrix (ЭТО TRUE POSITIVE)
                        confusion_matrix_counts[true_cls_id][pred_cls_id] += 1
                        
                        # Классификация по размерам объекта (не изображения!)
                        img_height, img_width = preds_results[0].orig_shape # Размеры оригинального изображения
                        # Денормализуем бокс истинного объекта для получения пиксельных размеров
                        _, _, obj_w_norm, obj_h_norm = true_obj_match["box"]
                        obj_width_px = int(obj_w_norm * img_width)
                        obj_height_px = int(obj_h_norm * img_height)

                        matched_true_objects[best_true_match_idx] = True # Отмечаем, что истинный объект найден

                    else: # Ложное срабатывание (FP) Это FALSE POSITIVE
                        self.logger.debug(f"  Предсказание {pred_idx} НЕ МАТЧИРОВАНО (IoU <= 0.5 со всеми несопоставленными GT). Классифицировано как FP.")

                        false_positives_on_background[pred_cls_id] += 1
                        current_image_false_positives_classes.append(pred_cls_id) # Добавляем класс FP

                        # Определяем, связано ли это FP с каким-либо GT объектом (даже уже сопоставленным)
                        # Это нужно для более информативного журнала, чтобы объяснить, почему предсказание,
                        # возможно, хорошее, но считается FP (например, дубликат).
                        associated_gt_cls_id = None
                        associated_gt_box_norm = None
                        
                        # Низкий порог IoU для связи FP с GT объектом для цели объяснения
                        # Если предсказание имеет хоть какое-то значимое пересечение с любым GT,
                        # мы его связываем с этим GT в журнале для ясности.
                        explanation_iou_threshold = 0.2 
                        
                        for true_obj_for_explanation in true_boxes:
                            iou_for_explanation = self._get_iou(pred_box, true_obj_for_explanation["box"])
                            if iou_for_explanation > explanation_iou_threshold:
                                associated_gt_cls_id = true_obj_for_explanation["class_id"]
                                associated_gt_box_norm = true_obj_for_explanation["box"]
                                self.logger.debug(f"  FP {pred_idx} (Класс {pred_cls_id}) связан для логирования с GT Классом {associated_gt_cls_id} (IoU={iou_for_explanation:.4f}).")
                                # Нашли GT, который может объяснить этот FP.
                                # Выходим из цикла, чтобы взять первое такое объяснение.
                                break 
                        
                        # Логируем FP в журнал, включая GT информацию, если она найдена
                        if self.prediction_journal:
                            self.prediction_journal.log_false_positive(
                                img_filename,
                                pred_cls_id, pred_box,
                                associated_gt_cls_id, associated_gt_box_norm # Передаем GT информацию для объяснения
                            )

                # Пропущенные объекты (FN)
                for i, true_obj in enumerate(true_boxes):
                    if not matched_true_objects[i]:
                        self.logger.debug(f"Изображение {img_filename}, GT {i} (Класс {true_obj['class_id']}, BBox {true_obj['box']}) НЕ МАТЧИРОВАН. Классифицировано как FN.")
                        missed_objects[true_obj["class_id"]] += 1
                        current_image_false_negatives_classes.append(true_obj["class_id"]) # Добавляем класс FN

                        # Логируем FN в журнал
                        if self.prediction_journal:
                            self.prediction_journal.log_false_negative(img_filename, true_obj["class_id"], true_obj["box"])

                # --- Генерация и сохранение визуализаций ---
                img_orig = cv2.imread(img_path)
                if img_orig is None:
                    self.logger.warning(f"Не удалось загрузить изображение для визуализации: {img_path}")
                    continue

                h, w, _ = img_orig.shape
                image_dims_for_drawing = (w, h)

                img_gt = img_orig.copy()
                img_pred = img_orig.copy()
                img_original_unannotated = img_orig.copy()
                
                # Рисуем истинные метки
                self._draw_boxes(img_gt, true_boxes, True, model_names, image_dims_for_drawing)
                # Рисуем предсказания
                self._draw_boxes(img_pred, predicted_boxes, False, model_names, image_dims_for_drawing)

                # Склеиваем изображения
                combined_img = np.hstack((img_original_unannotated, img_gt, img_pred))

                # Сохраняем комбинированное изображение
                save_path = os.path.join(self.results_dir, img_filename)
                cv2.imwrite(save_path, combined_img)

                # После сохранения комбинированного изображения, копируем его в FN/FP папки
                if current_image_false_negatives_classes:
                    for cls_id in set(current_image_false_negatives_classes): # Используем set, чтобы копировать по одному разу на класс
                        fn_class_name = model_names[cls_id] if cls_id < len(model_names) else f'Class_{cls_id}'
                        fn_save_dir = os.path.join(self.config.false_negatives_output_dir, fn_class_name)
                        os.makedirs(fn_save_dir, exist_ok=True) # Гарантируем, что папка класса существует
                        shutil.copy(save_path, os.path.join(fn_save_dir, img_filename))

                if current_image_false_positives_classes:
                    for cls_id in set(current_image_false_positives_classes): # Используем set, чтобы копировать по одному разу на класс
                        fp_class_name = model_names[cls_id] if cls_id < len(model_names) else f'Class_{cls_id}'
                        fp_save_dir = os.path.join(self.config.false_positives_output_dir, fp_class_name)
                        os.makedirs(fp_save_dir, exist_ok=True) # Гарантируем, что папка класса существует
                        shutil.copy(save_path, os.path.join(fp_save_dir, img_filename))

            except Exception as e:
                self.logger.error(f"Ошибка при обработке изображения {img_path} для предсказания: {e}", exc_info=True)
                continue # Переходим к следующему изображению

        # --- Построение итоговых таблиц ---
        self.logger.info("="*50)
        self.logger.info("АНАЛИЗ ПРЕДСКАЗАНИЙ")
        self.logger.info("="*50)

        # 1. Confusion Matrix
        self.logger.info("\n--- Confusion Matrix (Counts) ---")
        # Формируем заголовки
        class_labels = model_names

        # Добавляем "background" к списку классов для Confusion Matrix
        all_classes_for_cm = class_labels + ['background']
        num_classes_cm = len(all_classes_for_cm)
        
        # Инициализируем матрицу нулями
        matrix = np.zeros((num_classes_cm, num_classes_cm), dtype=int)
        
        # Заполняем матрицу TP/FP/FN/TN
        for true_cls_id, predictions in confusion_matrix_counts.items():
            for pred_cls_id, count in predictions.items():
                if true_cls_id < len(class_labels) and pred_cls_id < len(class_labels):
                    matrix[true_cls_id, pred_cls_id] = count
        
        # Заполняем False Negatives (пропущенные истинные объекты) - это предсказания как "background"
        background_col_idx = num_classes_cm - 1 # Индекс колонки "background"
        for cls_id, count in missed_objects.items():
            if cls_id < len(class_labels): # Убедимся, что это реальный класс, а не фоновый
                matrix[cls_id, background_col_idx] += count
                
        # Заполняем False Positives (ложные срабатывания) - это истинный "background", но предсказано как объект
        background_row_idx = num_classes_cm - 1 # Индекс строки "background"
        for cls_id, count in false_positives_on_background.items():
            if cls_id < len(class_labels): # Предсказано как один из реальных классов
                matrix[background_row_idx, cls_id] += count

        # Пересечение "background * background" всегда 0, так как мы не "предсказываем" фон и не "ищем" фон
        # в истинных метках как отдельный класс.


        # Выводим матрицу
        header = ["True\\Pred"] + all_classes_for_cm        
        col_widths = [max(len(h), 10) for h in header]
        header_str = " ".join(f"{h:<{col_widths[idx]}}" for idx, h in enumerate(header))
        self.logger.info(header_str)
        self.logger.info("-" * (sum(col_widths) + len(col_widths) - 1))
        
        for i, row in enumerate(matrix):
            label = all_classes_for_cm[i]
            row_str = [f"{label:<{col_widths[0]}}"] + [f"{val:<{col_widths[j+1]}}" for j, val in enumerate(row)]
            self.logger.info(" ".join(row_str))

        # 2. Статистика по размерам объектов
        self.logger.info("\n--- Классификация по размерам объектов ---")
        size_header = ["Class\\Size"] + list(self.config.size_categories.keys()) + ["Total True"]
        size_col_widths = [max(len(h), 10) for h in size_header]
        
        size_header_str = " ".join(f"{h:<{size_col_widths[idx]}}" for idx, h in enumerate(size_header))
        self.logger.info(size_header_str)
        self.logger.info("-" * (sum(size_col_widths) + len(size_col_widths) - 1))

        for cls_id, size_counts in size_classification.items():
            class_name = model_names[cls_id] if cls_id < len(model_names) else f"Class_{cls_id}"
            row_data = [class_name]
            total_true_for_class = 0
            for size_cat in self.config.size_categories.keys():
                count = size_counts.get(size_cat, 0)
                row_data.append(str(count))
                total_true_for_class += count
            row_data.append(str(total_true_for_class))
            
            row_str = [f"{row_data[0]:<{size_col_widths[0]}}"] + [f"{val:<{size_col_widths[j+1]}}" for j, val in enumerate(row_data[1:]) if j+1 < len(size_col_widths)]
            self.logger.info(" ".join(row_str))

        # 3. Пропущенные объекты (False Negatives)
        self.logger.info("\n--- Пропущенные объекты (False Negatives) ---")

        total_missed = sum(missed_objects.values())
        if total_missed > 0:
            for cls_id, count in missed_objects.items():
                if count > 0:
                    class_name = model_names[cls_id] if cls_id < len(model_names) else f'Class_{cls_id}'
                    self.logger.info(f"Класс '{class_name}': Пропущено (False Negative) - {count}")
        else:
            self.logger.info("Пропущенных объектов (False Negatives) не обнаружено.")
            
        # 4. Ложные срабатывания (False Positives)
        self.logger.info("\n--- Ложные срабатывания (False Positives) ---")
        total_false_positives = sum(false_positives_on_background.values())
        if total_false_positives > 0:
            for cls_id, count in false_positives_on_background.items():
                if count > 0:
                    class_name = model_names[cls_id] if cls_id < len(model_names) else f'Class_{cls_id}'
                    self.logger.info(f"Класс '{class_name}': Ложное срабатывание (False Positive) - {count}")
        else:
            self.logger.info("Ложных срабатываний (False Positives) не обнаружено.")

        self.logger.info("\nАнализ предсказаний завершен.")