#!/usr/bin/env python3
# ==============================================================================
# Script: image_to_hex.py
# Descrição: Processa uma imagem genérica e converte em um arquivo .txt
#            contendo pixels em hexadecimal (1024 linhas), pronto para ser
#            usado pelo testbench Verilog (run_project.do).
#
# Processo:
#   1. Carrega a imagem e converte para escala de cinza.
#   2. Aplica CLAHE para equalização de iluminação.
#   3. Usa Haarcascade para encontrar e recortar o rosto (ROI quadrada).
#   4. Redimensiona para 32x32.
#   5. Normaliza/Quantiza os valores [0..255] para Q1.7 [0..127].
#   6. Salva cada pixel em formato hexadecimal (00 a 7F), um por linha.
#
# Uso:
#   python scripts/image_to_hex.py inputs/foto.jpg inputs/foto_hex.txt
# ==============================================================================

import argparse
import sys
import os
import cv2
import numpy as np

# Caminho do classificador Haarcascade (relativo à raiz do projeto)
HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "haarcascade_frontalface_default.xml"
)

def resolve_haar_cascade_path():
    candidates = [
        HAAR_CASCADE_PATH,
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None

def crop_face_square(gray_frame, face_rect):
    h_img, w_img = gray_frame.shape
    x, y, w, h = face_rect

    # A mesma lógica exata do preprocessor.py e da inferência
    pad = int(w * 0.15)
    y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
    x1, x2 = max(0, x - pad), min(w_img, x + w + pad)

    return gray_frame[y1:y2, x1:x2]

def quantize_to_q17(frame_u8: np.ndarray) -> np.ndarray:
    """Converte 0..255 (uint8) para Q1.7 positivo (0..127)."""
    scaled = np.round(frame_u8.astype(np.float32) * 127.0 / 255.0)
    return np.clip(scaled, 0, 127).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Converte imagem para formato hex 32x32 para testbench.")
    parser.add_argument("input_img", type=str, help="Caminho da imagem de entrada (ex: foto.jpg)")
    parser.add_argument("output_txt", type=str, help="Caminho do arquivo texto de saída (ex: foto_hex.txt)")
    parser.add_argument("--no-haar", action="store_true", help="Pula a detecção de rosto e usa a imagem inteira")
    args = parser.parse_args()

    if not os.path.isfile(args.input_img):
        print(f"[ERRO] Imagem não encontrada: {args.input_img}")
        sys.exit(1)

    # 1. Carrega e converte para escala de cinza
    img_bgr = cv2.imread(args.input_img)
    if img_bgr is None:
        print(f"[ERRO] Não foi possível ler a imagem {args.input_img}")
        sys.exit(1)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Equalização (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    out_frame = gray
    
    # 3. Detecção de rosto
    if not args.no_haar:
        cascade_path = resolve_haar_cascade_path()
        if cascade_path:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if not face_cascade.empty():
                detection_configs = [
                    (1.2, 5, 60),
                    (1.1, 3, 40),
                    (1.05, 2, 30),
                ]
                faces = []
                for scale_factor, min_neighbors, min_size in detection_configs:
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size)
                    )
                    if len(faces) > 0:
                        break

                if len(faces) > 0:
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    out_frame = crop_face_square(gray, largest)
                    print(f"[OK] Rosto detectado e recortado: {largest}")
                else:
                    print("[AVISO] Nenhum rosto detectado. Usando a imagem inteira.")
            else:
                print("[AVISO] Falha ao carregar Haarcascade.")
        else:
            print("[AVISO] Haarcascade não encontrado.")

    # 4. Redimensionamento 32x32
    frame_32 = cv2.resize(out_frame, (32, 32), interpolation=cv2.INTER_AREA)

    # 5. Quantização Q1.7 (Normalização)
    frame_q17 = quantize_to_q17(frame_32)
    print("[OK] Imagem normalizada para Q1.7 (0 a 127).")

    # 6. Salva no formato Hexadecimal
    os.makedirs(os.path.dirname(args.output_txt) or '.', exist_ok=True)
    pixels = frame_q17.flatten()
    
    with open(args.output_txt, "w") as f:
        for p in pixels:
            f.write(f"{p:02X}\n")

    print(f"[OK] Arquivo gerado com sucesso: {args.output_txt} ({len(pixels)} linhas)")

if __name__ == "__main__":
    main()
