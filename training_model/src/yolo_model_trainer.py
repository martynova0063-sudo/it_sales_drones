# training_model/src/yolo_model_trainer.py

import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import yaml
import glob
import numpy as np

from .config import Config
from logger import Logger # Предполагается, что logger импортируется корректно

class YOLOModelTrainer:
    def __init__(self, config: Config, logger: Logger) -> None:
        self.config: Config = config
        self.logger: Logger = logger
        self.model: YOLO | None = None
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.config.best_model_path):
            self.logger.info(f"Загрузка сохраненной модели: {self.config.best_model_path}")
            self.model = YOLO(self.config.best_model_path)
        else:
            self.logger.info(f"Загрузка базовой модели: {self.config.base_model_name}")
            self.model = YOLO(self.config.base_model_name)

    def train(self) -> None:
        if self.model is None or not os.path.exists(self.config.data_yaml_path):
            self.logger.error("Ошибка запуска обучения: нет модели или data.yaml")
            return

        self.logger.info(f"Старт обучения. Вывод: {self.config.output_dir}")
        
        # Убедимся, что директория вывода существует перед обучением
        os.makedirs(self.config.output_dir, exist_ok=True)

        results = self.model.train(
            data=self.config.data_yaml_path,
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch_size,
            lr0=self.config.learning_rate,
            lrf=self.config.final_lr_factor,
            cos_lr=self.config.use_cos_lr,
            optimizer=self.config.optimizer,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            project=self.config.output_dir, # Используем настроенную директорию вывода
            name="train_run", # Имя для конкретного запуска обучения
            exist_ok=True,
            seed=self.config.seed,
            cache=False,
            verbose=True,
        )
        self.logger.info("Обучение завершено.")

        if results and hasattr(results, 'save_dir'): # Проверяем наличие результатов и директории сохранения
            run_dir = results.save_dir
            best_pt = os.path.join(run_dir, "weights", "best.pt")
            if os.path.exists(best_pt):
                shutil.copy(best_pt, self.config.best_model_path)
                self.logger.info(f"Best model saved to: {self.config.best_model_path}")
            
            self.plot_training_results(run_dir)
        else:
            self.logger.warning("Не найдены результаты обучения или директория сохранения.")


    def plot_training_results(self, run_dir: str) -> None:
        results_csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(results_csv): 
            self.logger.warning(f"Файл results.csv не найден по пути {results_csv}. Невозможно построить графики обучения.")
            return
        
        try:
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            plt.figure(figsize=(15, 5))
            
            # График Learning Rate
            plt.subplot(1, 3, 1)
            if 'lr/pg0' in df.columns: 
                plt.plot(df['epoch'], df['lr/pg0'], label='LR')
            plt.title('Learning Rate'); plt.legend(); plt.grid(True)
            
            # Графики потерь (Losses)
            plt.subplot(1, 3, 2)
            if 'train/box_loss' in df.columns: plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
            if 'train/cls_loss' in df.columns: plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
            plt.title('Losses'); plt.legend(); plt.grid(True)
            
            # График mAP
            plt.subplot(1, 3, 3)
            if 'metrics/mAP50(B)' in df.columns: plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
            plt.title('mAP'); plt.legend(); plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(run_dir, "custom_metrics.png")
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Графики обучения сохранены в {plot_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при построении графиков обучения: {e}", exc_info=True)

    def validate_on_test_set(self) -> None:
        """
        Выполняет валидацию обученной модели на тестовом датасете и выводит основные метрики.
        Визуализации и подробный анализ ошибок теперь выполняются в режиме '--predict'.
        """
        if self.model is None:
            self.logger.error("Модель не загружена. Валидация невозможна.")
            return
        if not os.path.exists(self.config.data_yaml_path):
             self.logger.error(f"Файл data.yaml не найден по пути {self.config.data_yaml_path}. Инференс невозможен без конфигурации датасета.")
             return

        # Читаем data.yaml для получения путей
        try:
            with open(self.config.data_yaml_path, 'r') as f:
                data_yaml_content: dict = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Ошибка чтения файла data.yaml: {e}", exc_info=True)
            return

        ###############################################################################
        # Задаем папку с изображениями Тестовой выборки
        ###############################################################################
        dataset_root: str | None = data_yaml_content.get('path')
        test_images_relative_path: str | None = data_yaml_content.get('test')

        if not dataset_root or not test_images_relative_path:
            self.logger.error(f"Не удалось найти 'path' или 'test' в {self.config.data_yaml_path}.")
            return

        test_images_path: str = os.path.join(dataset_root, test_images_relative_path)
        
        if not os.path.exists(test_images_path):
            self.logger.error(f"Тестовая директория изображений не найдена: {test_images_path}.")
            return
        ###############################################################################
        
        # --- Убедимся, что базовая директория существует ---
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # --- Определяем ПОЛНЫЕ пути для сохранения метрик и визуализаций ---
        # Мы хотим, чтобы результаты сохранялись в ./runs/YOLO_Drone_Detection/TIMESTAMP/metrics
        # и ./runs/YOLO_Drone_Detection/TIMESTAMP/visuals
        metrics_run_dir = os.path.join(self.config.output_dir, "metrics") # Здесь будут метрики
        visuals_run_dir = os.path.join(self.config.output_dir, "visuals") # Здесь будут предикты

        # Убедимся, что директории существуют
        os.makedirs(metrics_run_dir, exist_ok=True)
        os.makedirs(visuals_run_dir, exist_ok=True)

        self.logger.info(">>> ШАГ 1: Расчет метрик и Confusion Matrix...")
        
        # 1. Валидация (получение метрик)
        # Указываем project как директорию, куда должны попасть результаты,
        # а name указываем как "." чтобы избежать создания дополнительной подпапки "metrics/."
        # Ultralytics может создавать подпапку "predict" или "val" внутри project, даже если name="."
        # Поэтому, более надежный вариант - явно указать имя запуска.
        # Например, если мы хотим, чтобы все было в "metrics_run_dir", можем указать:
        # project=metrics_run_dir, name="metrics_run" (тогда результаты будут в metrics_run_dir/metrics_run)
        # Но если мы хотим, чтобы результаты были ПРЯМО в metrics_run_dir, то `name='.'` — это правильный подход.
        # Если это не работает, попробуем совсем убрать name или установить его как "val"
        
        # Вариант 1 (как было): project=metrics_run_dir, name="."
        # Вариант 2 (более явный): project=self.config.output_dir, name="metrics" (результат будет в output_dir/metrics)
        # Вариант 3 (для чистоты): project=metrics_run_dir, name="." - остается прежним

        # --- ПОПРОБУЕМ УКАЗАТЬ ПРОЕКТ КАК ОБЩУЮ ПАПКУ, А NAME КАК НАЗВАНИЕ РАНА ---
        # Это должно привести к сохранению в: self.config.output_dir/metrics/val/
        # И тогда нам нужно будет переместить файлы.
        # Это не очень удобно. Лучше искать способ сохранения прямо в указанную папку.
        
        # --- Пересмотрим документацию ultalytics ---
        # https://docs.ultralytics.com/usage/cli/#validate
        # "project: The directory to save results to. Defaults to runs/val."
        # "name: The name of the run. Defaults to exp."
        # Таким образом, если project=metrics_run_dir, то результаты будут сохранены в metrics_run_dir/exp
        # А если мы хотим сохранить прямо в metrics_run_dir, то `name` должно быть "."?
        # Или `project` должен быть `self.config.output_dir`, а `name` - "metrics"?
        # Давайте попробуем так:
        
        val_results = self.model.val(
            data=self.config.data_yaml_path,
            split='test',
            imgsz=self.config.imgsz,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            project=self.config.output_dir, # Общая директория проекта
            name="metrics",                # Название этого "рана"
            exist_ok=True,
            cache=False,
            verbose=True,
        )
        # Результаты будут в: self.config.output_dir/metrics/

        # Вывод метрик
        self.logger.info("\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
        self.logger.info(f"{'Class':<15} {'mAP50':<10} {'mAP50-95':<10}")
        self.logger.info("-" * 35)
        
        if hasattr(val_results, 'box') and val_results.box is not None:
            self.logger.info(f"{'ALL':<15} {val_results.box.map50:<10.3f} {val_results.box.map:<10.3f}")

            # Получаем имена классов из результатов валидации
            # val_results.names должен быть списком ['drone', 'bird', ...]
            model_names = val_results.names
            
            # Получаем индексы классов, для которых были найдены метрики
            if hasattr(val_results.box, 'ap_class_index') and val_results.box.ap_class_index is not None:
                found_class_indices_in_metrics = val_results.box.ap_class_index.tolist()
            else:
                found_class_indices_in_metrics = [] # Пустой список, если метрик нет

            # Теперь для вывода, мы должны пройти по ВСЕМ классам, которые есть в model_names
            # и проверить, есть ли для них метрики
            for i, cname in enumerate(model_names): # i - это индекс, cname - имя класса
                if i in found_class_indices_in_metrics:
                    # Если метрики найдены для этого класса, извлекаем их
                    # Мы используем индекс 'i' напрямую, так как он соответствует индексу в model_names
                    try:
                        map50 = val_results.box.all_ap[i, 0] # Используем 'i' вместо 'idx'
                        map95 = val_results.box.maps[i]     # Используем 'i' вместо 'idx'
                        self.logger.info(f"{cname:<15} {map50:<10.3f} {map95:<10.3f}")
                    except IndexError: 
                        self.logger.warning(f"Не удалось извлечь метрики для {cname} (IndexError).")
                    except Exception as e:
                        self.logger.warning(f"Не удалось извлечь метрики для {cname}: {e}")
                else:
                    # Если метрики не найдены для этого класса
                    self.logger.info(f"{cname:<15} {'0.000':<10} {'0.000':<10} (нет метрик)")

        else:
            self.logger.warning("Метрики Box не найдены в результатах валидации.")

        # --- Вывод Confusion Matrix ---
        self.logger.info("\n=== CONFUSION MATRIX (Normalized) ===")
        if hasattr(val_results, 'confusion_matrix') and val_results.confusion_matrix is not None:
            cm = val_results.confusion_matrix.matrix
            n_classes = len(names) if 'names' in locals() else 0 
            
            if n_classes > 0:
                matrix_labels = []
                for i in range(cm.shape[0]):
                    if i < n_classes:
                        matrix_labels.append(names[i][:8]) 
                    else:
                        matrix_labels.append("backg")

                row_sums = cm.sum(axis=1, keepdims=True)
                norm_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
                
                col_widths = [max(len(h), 10) for h in ["True\\Pred"] + matrix_labels]
                
                header_str = " ".join(f"{h:<{col_widths[idx]}}" for idx, h in enumerate(["True\\Pred"] + matrix_labels))
                self.logger.info(header_str)
                self.logger.info("-" * (sum(col_widths) + len(col_widths) - 1))
                
                for i, row in enumerate(norm_cm):
                    label = matrix_labels[i] if i < len(matrix_labels) else f"Class_{i}"
                    row_str = [f"{label:<{col_widths[0]}}"] + [f"{val:<{col_widths[j+1]}.3f}" for j, val in enumerate(row) if j+1 < len(col_widths)]
                    self.logger.info(" ".join(row_str))
            else:
                self.logger.warning("Имена классов недоступны для меток Confusion Matrix.")
        else:
            self.logger.warning("Confusion matrix не найдена в результатах валидации.")
        
        # Сообщение о пути к метрикам теперь будет вести к project/name
        self.logger.info(f"\nОтчет метрик сохранен в директории: {os.path.join(self.config.output_dir, 'metrics')}")

        ### # --- Шаг 2: Сохранение визуализаций ---
        ### self.logger.info("\n>>> ШАГ 2: Сохранение визуализаций (без логов)...")
        ### 
        ### images = []
        ### for ext in ['*.jpg', '*.jpeg', '*.png']:
        ###     images.extend(glob.glob(os.path.join(test_images_path, '**', ext), recursive=True))