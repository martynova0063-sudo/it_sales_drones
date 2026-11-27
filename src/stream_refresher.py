# src/stream_refresher.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import time
import logging
import re
import os
from src.runtime_logger import get_runtime_logger
logger = get_runtime_logger("StreamRefresher")

logger = logging.getLogger(__name__)

def get_fresh_hls_url(camera_page_url: str, referer: str) -> str | None:
    """
    Использует Selenium для открытия страницы камеры и извлечения свежего .m3u8 URL.
    
    Args:
        camera_page_url: URL страницы камеры (например, https://moidom.citylink.pro/pub/94854)
        referer: Referer для запроса
    
    Returns:
        Свежий URL .m3u8 или None, если не найден.
    """
    import json
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Включаем логирование производительности
    chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = None
    hls_url = None
    camera_page_url = camera_page_url.strip()
    referer = referer.strip()

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(2)
        driver.get(camera_page_url)

        # Ждём прогрузки видео
        time.sleep(1)

        # Получаем логи производительности
        perf_logs = driver.get_log('performance')
        for entry in perf_logs:
            try:
                msg = json.loads(entry['message'])
                method = msg.get("message", {}).get("method")
                if method == "Network.responseReceived":
                    resp = msg["message"]["params"]["response"]
                    url = resp.get("url", "")
                    if ".m3u8" in url and "moidom-stream.ru" in url:
                        hls_url = url
                        break
            except Exception:
                continue

        if hls_url:
            logger.info(f"Найден HLS URL: {hls_url}")
            return hls_url

        logger.warning("Не удалось найти .m3u8 в логах")
        return None

    except Exception as e:
        logger.error(f"Ошибка при получении HLS URL: {e}")
        return None
    finally:
        if driver:
            driver.quit()

def get_fresh_hls_url_old(camera_page_url: str, referer: str) -> str | None:
    """
    Использует Selenium для открытия страницы камеры и извлечения свежего .m3u8 URL.
    
    Args:
        camera_page_url: URL страницы камеры (например, https://moidom.citylink.pro/pub/94854)
        referer: Referer для запроса
    
    Returns:
        Свежий URL .m3u8 или None, если не найден.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Включаем логирование производительности
    chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    # Добавляем опции для использования системного прокси/VPN (если он настроен на уровне ОС)
    # Это может помочь, если Chrome не подхватывает настройки VPN по умолчанию.
    # Если VPN настроен через переменную окружения http_proxy/https_proxy:
    # proxy = os.getenv("http_proxy") or os.getenv("https_proxy")
    # if proxy:
    #     chrome_options.add_argument(f"--proxy-server={proxy}")
    #     logger.info(f"Использование прокси для Chrome: {proxy}")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(10)
        driver.get(camera_page_url)

        # Ждём загрузки контента
        time.sleep(5)

        # Ищем video элементы
        elements = driver.find_elements(By.TAG_NAME, "video")
        for el in elements:
            src = el.get_attribute("src")
            if src and ".m3u8" in src:
                logger.info(f"Найден новый URL через <video>: {src}")
                return src

        # Если не нашли — ищем в JS-коде
        script = """
        const scripts = Array.from(document.querySelectorAll('script'));
        for (let s of scripts) {
            if (s.textContent.includes('.m3u8')) {
                const match = s.textContent.match(/https:\\/\\/s\\d+\\.moidom-stream\\.ru[^"\\'\\s]+/);
                if (match) return match[0];
            }
        }
        return null;
        """
        js_result = driver.execute_script(script)
        if js_result:
            logger.info(f"Найден URL через JS: {js_result}")
            return js_result

        logger.warning("Не удалось найти .m3u8 ни в video, ни в JS")
        return None

    except Exception as e:
        logger.error(f"Ошибка при получении нового URL: {e}")
        return None
    finally:
        if driver:
            driver.quit()