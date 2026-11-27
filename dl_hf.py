from huggingface_hub import snapshot_download
import zipfile
from pathlib import Path

# --- Конфигурация ---
REPO_ID = "lgrzybowski/seraphim-drone-detection-dataset"
# Укажите путь, куда вы хотите сохранить датасет на вашем компьютере
LOCAL_DIR = Path("drone_dataset_folder") 

# --- Шаг 1: Скачивание всего репозитория ---
# Эта функция скачает все файлы из Hugging Face в указанную папку
print("Начинаю скачивание. Это может занять некоторое время...")
repo_path = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR))
print("Скачивание завершено.")

# --- Шаг 2: Распаковка всех .zip файлов ---
# Находим все скачанные .zip архивы
zip_files = list(repo_path.rglob("*.zip"))
print(f"Найдено {len(zip_files)} zip-архивов для распаковки.")

for zip_path in zip_files:
    try:
        # Распаковываем архив в ту же папку, где он лежит
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(zip_path.parent)
        print(f"✅ Распакован: {zip_path.name}")
        # Удаляем .zip файл после успешной распаковки
        zip_path.unlink()  
    except zipfile.BadZipFile:
        print(f"⚠️ Пропускаю поврежденный архив: {zip_path.name}")

print("🎉 Все архивы распакованы и удалены.")
print(f"📂 Датасет готов к использованию в папке: {repo_path.resolve()}")