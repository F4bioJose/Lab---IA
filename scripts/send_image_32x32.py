#!/usr/bin/env python3
# ==============================================================================
# Script: send_image_32x32.py
# Descrição: Envia uma imagem 32×32 grayscale via UART serial para a FPGA
#            DE2-115 para inferência pela CNN. Suporta dois modos:
#            1. Envio de arquivo estático (--file imagem.png)
#            2. Captura única da webcam com detecção de rosto Haarcascade
#
#            Diferente do send_image.py da branch terceira_entrega (64×64),
#            este script é específico para a resolução 32×32 (1024 bytes)
#            usada pelo pipeline CNN do cnn_top.
#
# Dependências: pip install opencv-python pyserial numpy
#
# Uso:
#   python send_image_32x32.py --file imagem.png          # Arquivo estático
#   python send_image_32x32.py --file inputs/teste2.txt --hex  # Arquivo hex
#   python send_image_32x32.py                            # Webcam (1 frame)
#   python send_image_32x32.py --port COM3                # Windows
#   python send_image_32x32.py --mock --file foto.jpg     # Testar sem placa
# ==============================================================================

import argparse
import sys
import time
import os

# Suprime warnings do Qt e do OpenCV
os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import numpy as np
import serial


# Caminho do classificador Haarcascade (relativo à raiz do projeto)
HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "haarcascade_frontalface_default.xml"
)


def resolve_haar_cascade_path():
    """Resolve o caminho do Haarcascade sem depender de cv2.data."""
    candidates = [
        HAAR_CASCADE_PATH,
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None

# Resolução da imagem enviada (32×32 = 1024 bytes)
IMG_SIZE = 32
FRAME_BYTES = IMG_SIZE * IMG_SIZE  # 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Envia imagem 32x32 grayscale via UART para inferência na FPGA."
    )
    parser.add_argument(
        "--port", default="/dev/ttyUSB0",
        help="Porta serial (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate (default: 115200, compatível com uart_rx do cnn_top)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Índice da câmera (default: 0)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Enviar imagem estática em vez da webcam"
    )
    parser.add_argument(
        "--hex", action="store_true",
        help="Indica que --file é um arquivo .txt com valores hex (um por linha)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Modo de simulação (não tenta abrir a porta serial real)"
    )
    parser.add_argument(
        "--no-haar", action="store_true",
        help="Desabilitar detecção de rosto (enviar frame completo)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Mostrar preview da imagem enviada"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Enviar bytes crus (sem quantizar para Q1.7)"
    )
    return parser.parse_args()


def send_frame(ser, frame: np.ndarray):
    """Envia um frame 32x32 (1024 bytes) via serial."""
    raw_bytes = frame.astype(np.uint8).tobytes()
    assert len(raw_bytes) == FRAME_BYTES, (
        f"Frame deve ter {FRAME_BYTES} bytes, tem {len(raw_bytes)}"
    )
    if ser is not None:
        ser.write(raw_bytes)
        ser.flush()
    return raw_bytes


def quantize_to_q17(frame_u8: np.ndarray) -> np.ndarray:
    """Converte 0..255 (uint8) para Q1.7 positivo (0..127)."""
    scaled = np.round(frame_u8.astype(np.float32) * 127.0 / 255.0)
    return np.clip(scaled, 0, 127).astype(np.uint8)


def load_hex_file(filepath):
    """Carrega arquivo .txt com valores hexadecimais (um por linha, sem prefixo)."""
    pixels = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                pixels.append(int(line, 16))
    if len(pixels) != FRAME_BYTES:
        print(f"[AVISO] Arquivo hex tem {len(pixels)} valores, esperado {FRAME_BYTES}.")
        if len(pixels) > FRAME_BYTES:
            pixels = pixels[:FRAME_BYTES]
        else:
            pixels.extend([0] * (FRAME_BYTES - len(pixels)))
    return np.array(pixels, dtype=np.uint8).reshape((IMG_SIZE, IMG_SIZE))


def crop_face_square(gray_frame, face_rect):
    """
    Recorta o rosto expandindo para um quadrado centrado no retângulo detectado.
    Isso evita distorção ao redimensionar para 32×32.
    """
    h_img, w_img = gray_frame.shape
    x, y, w, h = face_rect

    # Expande para quadrado usando o lado maior
    side = max(w, h)

    # Adiciona margem de 15% para incluir testa e queixo
    margin = int(side * 0.15)
    side = side + margin

    # Centraliza o quadrado no centro do retângulo original
    cx = x + w // 2
    cy = y + h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    # Clamp às bordas da imagem
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    return gray_frame[y1:y2, x1:x2]


def main():
    args = parse_args()

    # Carrega o classificador Haarcascade
    face_cascade = None
    if not args.no_haar and not args.hex:
        cascade_path = resolve_haar_cascade_path()
        if cascade_path is not None:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            print(f"[OK] Haarcascade carregado: {cascade_path}")
        else:
            print("[AVISO] Haarcascade não encontrado. Detecção de rosto desabilitada.")

        if face_cascade is not None and face_cascade.empty():
            print("[AVISO] Falha ao carregar Haarcascade. Detecção desabilitada.")
            face_cascade = None

    # Abre a porta serial
    ser = None
    if not args.mock:
        try:
            ser = serial.Serial(args.port, args.baud, timeout=1)
            print(f"[OK] Porta serial {args.port} aberta a {args.baud} baud")
        except serial.SerialException as e:
            print(f"[ERRO] Não foi possível abrir {args.port}: {e}")
            print("[INFO] Use --mock para testar sem a placa conectada.")
            sys.exit(1)
    else:
        print(f"[MOCK] Rodando em modo de simulação (sem porta serial).")

    # Aguarda a FPGA estabilizar após reset
    time.sleep(0.1)

    if args.file:
        # === Modo arquivo estático ===
        if args.hex:
            # Arquivo texto com valores hex (formato do teste2.txt)
            frame = load_hex_file(args.file)
            print(f"[OK] Arquivo hex '{args.file}' carregado ({frame.shape})")
        else:
            # Arquivo de imagem (PNG, JPG, BMP, etc.)
            img = cv2.imread(args.file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ERRO] Não foi possível ler {args.file}")
                sys.exit(1)

            out_frame = img
            # Aplica Haarcascade no arquivo estático se habilitado
            if face_cascade is not None:
                faces = face_cascade.detectMultiScale(
                    img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    out_frame = crop_face_square(img, largest)
                    print(f"[OK] Rosto detectado: {largest}")
                else:
                    print("[AVISO] Nenhum rosto detectado, enviando imagem completa.")

            frame = cv2.resize(out_frame, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_AREA)

        # Envia o frame
        if not args.raw and not args.hex:
            frame = quantize_to_q17(frame)
            print("[OK] Frame quantizado para Q1.7 (0..127)")
        t0 = time.time()
        raw = send_frame(ser, frame)
        elapsed = time.time() - t0
        print(f"[OK] Imagem enviada ({FRAME_BYTES} bytes em {elapsed:.3f}s)")

        # Preview opcional
        if args.preview:
            preview = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"Preview {IMG_SIZE}x{IMG_SIZE} -> FPGA", preview)
            print("[INFO] Pressione qualquer tecla na janela de preview para sair.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Exibe os bytes enviados (primeiros e últimos 8)
        print(f"[DEBUG] Primeiros 8 bytes: {' '.join(f'{b:02X}' for b in raw[:8])}")
        print(f"[DEBUG] Últimos  8 bytes: {' '.join(f'{b:02X}' for b in raw[-8:])}")

        if ser is not None:
            ser.close()
        return

    # === Modo webcam (captura única) ===
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera {args.camera}")
        sys.exit(1)

    print(f"[OK] Câmera {args.camera} aberta. Capturando 1 frame...")

    ret, frame_bgr = cap.read()
    if not ret:
        print("[ERRO] Falha na captura da câmera.")
        cap.release()
        sys.exit(1)

    # Converte para grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Aplica CLAHE para normalização de iluminação
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    out_frame = gray
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            out_frame = crop_face_square(gray, largest)
            print(f"[OK] Rosto detectado: {largest}")
        else:
            print("[AVISO] Nenhum rosto detectado, enviando frame completo.")

    frame_32 = cv2.resize(out_frame, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_AREA)

    # Envia o frame
    if not args.raw:
        frame_32 = quantize_to_q17(frame_32)
        print("[OK] Frame quantizado para Q1.7 (0..127)")
    t0 = time.time()
    raw = send_frame(ser, frame_32)
    elapsed = time.time() - t0
    print(f"[OK] Frame capturado e enviado ({FRAME_BYTES} bytes em {elapsed:.3f}s)")

    # Preview
    if args.preview:
        preview = cv2.resize(frame_32, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Preview {IMG_SIZE}x{IMG_SIZE} -> FPGA", preview)
        print("[INFO] Pressione qualquer tecla na janela de preview para sair.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"[DEBUG] Primeiros 8 bytes: {' '.join(f'{b:02X}' for b in raw[:8])}")
    print(f"[DEBUG] Últimos  8 bytes: {' '.join(f'{b:02X}' for b in raw[-8:])}")

    cap.release()
    if ser is not None:
        ser.close()
    print("[FIM] Inferência disparada na FPGA. Observe os LEDs LEDG.")


if __name__ == "__main__":
    main()
