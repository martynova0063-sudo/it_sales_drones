# src/event_bus.py

from typing import Callable, List, Any
import asyncio

class EventBus:
    """
    Простая шина событий для реализации паттерна "Публикатор-Подписчик".
    Позволяет подписчикам регистрировать колбэки, которые будут вызваны
    при публикации события.
    """
    def __init__(self):
        self._subscribers: List[Callable[[Any], Any]] = [] # Any - потому что могут быть sync/async, None/return
        self._websocket_subscribers: List[Callable[[Any], Any]] = [] # Отдельный список для WS

    def subscribe(self, callback: Callable[[Any], Any]) -> None:
        """
        Регистрирует функцию-подписчика.
        
        Args:
            callback: Асинхронная или синхронная функция, которая будет вызвана
                      при публикации события. Примет один аргумент - событие.
        """
        self._subscribers.append(callback)

    def subscribe_websocket(self, callback: Callable[[Any], Any]) -> None:
        """
        Регистрирует функцию-подписчика для WebSocket.
        
        Args:
            callback: Асинхронная или синхронная функция, которая будет вызвана
                      при публикации события. Примет один аргумент - событие.
        """
        self._websocket_subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Any], Any]) -> None:
        """Отменяет подписку функции."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def unsubscribe_websocket(self, callback: Callable[[Any], Any]) -> None:
        """Отменяет подписку функции для WebSocket."""
        if callback in self._websocket_subscribers:
            self._websocket_subscribers.remove(callback)


    async def publish(self, event_data: Any) -> None:
        """
        Публикует событие. Все зарегистрированные подписчики будут уведомлены.
        
        Args:
            event_data: Данные события (например, объект AlertPayload).
        """
        # Публикуем для обычных подписчиков
        for callback in self._subscribers:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback(event_data)) # Запускаем как задачу, чтобы не блокировать
            else:
                try:
                    callback(event_data)
                except Exception as e:
                    # Логируем ошибку в колбэке, чтобы не обрушить шину
                    print(f"Ошибка в синхронном колбэке EventBus: {e}") 
        
        # Публикуем для WebSocket подписчиков
        for callback in self._websocket_subscribers:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback(event_data)) # Запускаем как задачу
            else:
                try:
                    callback(event_data)
                except Exception as e:
                    print(f"Ошибка в синхронном WS колбэке EventBus: {e}")