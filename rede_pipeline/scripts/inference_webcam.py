import cv2
import numpy as np
import tensorflow as tf
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.preprocessor import ImageProcessor
from src.model_io import load_tinycnn_model

cfg = Config()

def iniciar_inferencia():
    # 1. Carrega o modelo final
    model_path = cfg.MODEL_PATH
    if not model_path.exists():
        print(f"Erro Crítico: Modelo não encontrado em {model_path}")
        return

    model = load_tinycnn_model(model_path, compile_model=False)
    processor = ImageProcessor(cfg.IMG_SIZE)
    class_names = None
    
    # Tenta usar o arquivo com sufixo, senao usa o generico
    class_map_file = cfg.CLASS_MAP_PATH
    if not class_map_file.exists():
        class_map_file = cfg.MODELS_DIR / "class_indices.json"

    if class_map_file.exists():
        with open(class_map_file, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        class_names = [None] * len(class_indices)
        for name, idx in class_indices.items():
            class_names[int(idx)] = name
    else:
        print(f"Aviso: Arquivo de classes não encontrado em {class_map_file}")

    confidence_threshold = 0.55

    cap = cv2.VideoCapture(0)
    print("Iniciando Webcam... Clique na janela do vídeo e pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 2. Converte para escala de cinza e aplica CLAHE (como no treino)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        # 3. Detecta múltiplos rostos com cascata adaptativa (como no treino)
        detection_configs = [
            (1.2, 5, 60),
            (1.1, 3, 40),
            (1.05, 2, 30),
        ]
        faces = []
        for sf, mn, ms in detection_configs:
            faces = processor.face_cascade.detectMultiScale(
                gray_clahe, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms)
            )
            if len(faces) > 0:
                break

        if len(faces) == 0:
            # Texto caso ninguém esteja na câmera
            cv2.putText(frame, "Procurando rosto...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for (x, y, w, h) in faces:
            # Extrair rosto com o mesmo padding de 15% usado no treinamento
            pad = int(w * 0.15)
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
            
            face_roi = gray_clahe[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            face_resized = cv2.resize(face_roi, (32, 32))
            face_normalized = face_resized.astype('float32') / 255.0
            rosto_input = np.expand_dims(face_normalized, axis=(0, -1))
            probs = model.predict(rosto_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])

            # ------------- DEBBUGING DE DECODIFICAÇÃO -------------
            if class_names and pred_idx < len(class_names) and class_names[pred_idx] is not None:
                raw_class = class_names[pred_idx]
                is_unknown = raw_class == cfg.UNKNOWN_CLASS_NAME
                
                # Remove o prefixo numérico se existir
                if "_" in raw_class:
                    parts = raw_class.split("_", 1)
                    if parts[0].isdigit():
                        display_name = parts[1]
                    else:
                        display_name = raw_class
                else:
                    display_name = raw_class
                
                # Substitui espaços por underscores
                display_name = display_name.replace(" ", "_")
            else:
                display_name = f"classe {pred_idx}"
                is_unknown = pred_idx == 0

            # Imprime o mapeamento real no terminal para quebrar a dúvida!
            print(f"Index da CNN: {pred_idx} | Classe Mapeada: {display_name} | Confiança: {pred_conf*100:.1f}%")

            # 6. Lgica de Cores e Texto
            if is_unknown or pred_conf < confidence_threshold:
                label = f"NEGADO {display_name} ({pred_conf * 100:.1f}%)"
                color = (0, 0, 255)
            else:
                label = f"LIBERADO {display_name} ({pred_conf * 100:.1f}%)"
                color = (0, 255, 0)

            # --- DESENHO DO QUADRADO E TEXTO ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Opcional: Mostra o que a rede está a ver num ecrã separado
            cv2.imshow('Visao da CNN (32x32)', cv2.resize(face_resized, (160, 160)))

        cv2.imshow('Fechadura Biometrica', frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_inferencia()