# src/schemas.py

from typing import Optional, List, Tuple
from pydantic import BaseModel

class AlertPayload(BaseModel):
    """
    Pydantic модель для данных алерта/хартбита, отправляемых с Ноды на ПУ.
    """
    node_id: str
    timestamp: str
    type: str = "alert" # "alert" or "heartbeat"
    object_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: float = 0.0
    bbox: List[int] = [] # [x1, y1, x2, y2]
    velocity: Optional[Tuple[float, float]] = None
    prediction: Optional[List[Tuple[float, float]]] = None
    prediction_iou_score: Optional[float] = None
    image_base64: str
    camera_name: Optional[str] = "Unknown Camera"
    model_name: Optional[str] = "Unknown Model"
    source_type: Optional[str] = "Unknown Source"
    stream_url: Optional[str] = ""
    all_classes_names: Optional[List[str]] = None
    target_classes: Optional[List[str]] = None