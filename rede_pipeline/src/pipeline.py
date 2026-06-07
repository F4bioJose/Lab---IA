import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

from src.preprocessor import ImageProcessor
from src.model_io import load_tinycnn_model


class FechaduraBiometricaPipeline:
    """
    Pipeline unificado de inferência.
    Encapsula o pré-processamento (CLAHE, Haar Cascade) e a Tiny-CNN.
    """

    def __init__(
        self,
        model_path: str,
        img_size: int = 32,
        confidence_threshold: float = 0.55,
        class_map_path: str = None,
        unknown_prefix: str = "0_",
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

        self.model = load_tinycnn_model(model_path, compile_model=False)
        self.processor = ImageProcessor(img_size)
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.unknown_prefix = unknown_prefix

        self.class_names = None
        if class_map_path and Path(class_map_path).exists():
            with open(class_map_path, "r", encoding="utf-8") as f:
                class_indices = json.load(f)
            self.class_names = [None] * len(class_indices)
            for name, idx in class_indices.items():
                self.class_names[int(idx)] = name

    def predizer_imagem(self, frame: np.ndarray):
        """
        Recebe um frame BGR (ex: da webcam), processa e retorna a predição.
        """
        face_crop = self.processor.detect_and_crop(frame)

        if face_crop is None:
            return {
                "status": "Nenhuma face detectada",
                "autorizado": False,
                "classe_predita": None,
                "confianca": 0.0,
                "face_crop": None,
            }

        rosto_norm = face_crop.astype("float32") / 255.0

        rosto_input = np.expand_dims(rosto_norm, axis=(0, -1))

        pred = self.model.predict(rosto_input, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx])

        if self.class_names and idx < len(self.class_names):
            classe = self.class_names[idx]
            is_unknown = classe.startswith(self.unknown_prefix)
        else:
            classe = f"classe_{idx}"
            is_unknown = idx == 0

        autorizado = (not is_unknown) and (conf >= self.confidence_threshold)

        return {
            "status": "Sucesso",
            "autorizado": autorizado,
            "classe_predita": classe,
            "confianca": conf,
            "face_crop": face_crop
        }