# src/config.py
import os
from datetime import datetime
import yaml

class Config:
    """
    Класс для управления всеми параметрами конфигурации: датасетом, моделью и обучением.
    Включает настройки для YOLO26.
    """
    def __init__(self) -> None:
        # Общие настройки
        self.project_name: str = "YOLO_Drone_Detection"
        self.timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Определяем абсолютный путь к директории, где будет лежать проект
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        # Идем на уровень выше, чтобы выйти из директории src
        project_root = os.path.dirname(base_dir) 
        self.output_dir: str = os.path.join(project_root, "runs", self.project_name, self.timestamp)

        # Конфигурация датасета
        self.dataset_base_dir: str = os.path.join(project_root, "datasets")
        # ID папки на Google Drive. gdown.download_folder будет использовать его.
        # Убедитесь, что папка доступна по ссылке.
        self.google_drive_folder_id: str = "19AP1eJsZoJRtNohxRYQUIzByYgAqokCc"
        self.dataset_archives: dict[str, dict[str, str]] = {
        # Первый датасет (от 02.02.2026)
        # https://drive.google.com/file/d/1wT6KcRkcROKLMCSLKQZnw4zTMEIBLvwX/view?usp=drive_link
        #    "src1": {"name": "drone and nodrone.v3i.yolo26.zip", "id": "1wT6KcRkcROKLMCSLKQZnw4zTMEIBLvwX"}
        
        # Второй датасет (от 16.02.2026)
        # https://drive.google.com/file/d/116OCDAwY9NH4TKTNlyNJyzTaStAez2Gq/view?usp=drive_link
            "src1": {"name": "drone_plandrone_nodrone.v1i.yolo26.zip", "id": "116OCDAwY9NH4TKTNlyNJyzTaStAez2Gq"}

#             "src1": {"name": "drone and nodrone.v2i.yolov8.zip"}
#             "src2": {"name": "11data.zip"},
#             "src3": {"name": "drone.v2i.yolov8.zip"},
#             "src4": {"name": "drone.v1i.yolov8.zip"},
#             "src5": {"name": "Drone1.v1i.yolov8.zip"},
        }
        # Сопоставление классов из паспорта датасета
        self.class_legend: dict[int, str] = {
            0: "bird",
            1: "drones",
            2: "helicopter",
            3: "plane"
        }

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dataset_base_dir, exist_ok=True)

        # Путь к файлу data.yaml, который будет создан для YOLO
        self.data_yaml_path: str = os.path.join(self.dataset_base_dir, "data.yaml")

        # Конфигурация модели
        # Используем YOLO26 nano-версию
        self.base_model_name: str = "yolo26n.pt" 

        # Используем YOLO26 small-версию
        # self.base_model_name: str = "yolo26s.pt" 
        self.saved_model_dir: str = os.path.join(project_root, "checkpoints", self.project_name)
        os.makedirs(self.saved_model_dir, exist_ok=True)
        # Путь, по которому будет храниться лучшая сохраненная модель
        ### self.best_model_path: str = os.path.join(self.saved_model_dir, "best.pt")
        self.best_model_path: str = os.path.join(self.saved_model_dir, f"best_{self.base_model_name.replace('.pt', '')}.pt")

        # Общая корневая директория для всех результатов анализа предсказаний
        self.global_results_dir: str = os.path.join(project_root, "results")
        # Путь для сохранения визуализированных результатов предсказаний (картинок)
        self.prediction_visualizations_dir: str = os.path.join(self.global_results_dir, "results_predictions")
        # Путь для сохранения изображений с пропущенными объектами (False Negatives)
        self.false_negatives_output_dir: str = os.path.join(self.global_results_dir, "False Negatives")
        # Путь для сохранения изображений с ложными срабатываниями (False Positives)
        self.false_positives_output_dir: str = os.path.join(self.global_results_dir, "False Positives")
        
        # Путь к CSV-файлу журнала для False Negatives и False Positives
        self.prediction_journal_path: str = os.path.join(self.global_results_dir, "prediction_journal.csv")

        # Пути к файлам статистики датасета
        self.dataset_stats_output_dir: str = os.path.join(self.global_results_dir, "dataset_statistics")
        self.train_stats_file: str = os.path.join(self.dataset_stats_output_dir, "train_statistics.txt")
        self.val_stats_file: str = os.path.join(self.dataset_stats_output_dir, "val_statistics.txt")
        self.test_stats_file: str = os.path.join(self.dataset_stats_output_dir, "test_statistics.txt")
        self.overall_stats_file: str = os.path.join(self.dataset_stats_output_dir, "overall_statistics.txt")
        os.makedirs(self.dataset_stats_output_dir, exist_ok=True) # Создаем директорию для статистики

        # Пути к файлам графиков статистики датасета
        self.train_class_plot: str = os.path.join(self.dataset_stats_output_dir, "train_class_distribution.png")
        self.train_size_plot: str = os.path.join(self.dataset_stats_output_dir, "train_size_distribution.png")
        self.val_class_plot: str = os.path.join(self.dataset_stats_output_dir, "val_class_distribution.png")
        self.val_size_plot: str = os.path.join(self.dataset_stats_output_dir, "val_size_distribution.png")
        self.test_class_plot: str = os.path.join(self.dataset_stats_output_dir, "test_class_distribution.png")
        self.test_size_plot: str = os.path.join(self.dataset_stats_output_dir, "test_size_distribution.png")
        self.overall_class_plot: str = os.path.join(self.dataset_stats_output_dir, "overall_class_distribution.png")
        self.overall_size_plot: str = os.path.join(self.dataset_stats_output_dir, "overall_size_distribution.png")

        # Категории размеров объектов в пикселях (по наибольшему измерению)
        self.size_categories: dict[str, tuple[int, int]] = {
            "Micro": (0, 20),
            "Tiny": (21, 40),
            "Small": (41, 80),
            "Medium": (81, 160),
            "Large": (161, 320),
            "XLarge": (321, 2000) # Максимальное значение для ширины/высоты объекта, не изображения
        }

        # Гиперпараметры обучения (примерные значения, настраивайте по необходимости)
        self.epochs: int = 200
        self.imgsz: int = 640 # Размер изображения для обучения
        self.batch_size: int = 32
        self.learning_rate: float = 0.0001       # стартовый learning_rate 0.0002 / 0.00001
        self.final_lr_factor: float = 0.01        # Коэффициент learning_rate, показывающий насколько изменится lr в финальной эпохе
        self.use_cos_lr=True                      # Способ изменения learning_rate
        # YOLO26 использует MuSGD. Если ultralytics обнажает его как прямую опцию для 'optimizer',
        # то можно изменить на "MuSGD". В противном случае, SGD является хорошей базовой опцией,
        # а MuSGD, возможно, уже интегрирован в yolo26.pt модель.
        self.optimizer: str = "MuSGD" ## SGD
        self.momentum: float = 0.937
        self.weight_decay: float = 0.0005
        self.seed: int = 42 # Для воспроизводимости

        # Конфигурация инференса
        self.conf_threshold: float = 0.25 # Порог уверенности для обнаружения
        self.iou_threshold: float = 0.7   # Порог IoU для NMS (если используется one-to-many head, хотя YOLO26 NMS-free по умолчанию)

    def save(self, path: str) -> None:
        """
        Сохраняет текущую конфигурацию в YAML-файл.

        Args:
            path: Путь к файлу, в который будет сохранена конфигурация.
        """
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def load(self, path: str) -> None:
        """
        Загружает конфигурацию из YAML-файла.

        Args:
            path: Путь к YAML-файлу, из которого будет загружена конфигурация.
        """
        with open(path, 'r') as f:
            loaded_config: dict = yaml.safe_load(f)
            self.__dict__.update(loaded_config)