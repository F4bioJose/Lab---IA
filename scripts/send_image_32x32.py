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

# Resolução da imagem de rosto enviada (32×32 = 1024 bytes)
IMG_SIZE = 32
FRAME_BYTES = IMG_SIZE * IMG_SIZE  # 1024

# Resolução da imagem de vídeo (128×128 = 16384 bytes)
VIDEO_SIZE = 128
VIDEO_FRAME_BYTES = VIDEO_SIZE * VIDEO_SIZE  # 16384

# Bytes de controle
CONTROL_NO_FACE = b'\x00'  # Nenhum rosto detectado (frame de vídeo)
CONTROL_FACE    = b'\xFF'  # Rosto detectado (frame 32x32)

# Cooldown após detecção de rosto (segundos)
FACE_COOLDOWN = 2.5   # Pausa total após enviar rosto
FACE_GRACE    = 2.5   # Período mínimo de apenas vídeo após cooldown


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


def send_control_byte(ser, control_byte: bytes):
    """Envia o byte de controle antes do frame."""
    if ser is not None:
        ser.write(control_byte)
        ser.flush()


def send_frame(ser, frame: np.ndarray, expected_size=None):
    """Envia um frame via serial. Suporta tamanhos variáveis."""
    raw_bytes = frame.astype(np.uint8).tobytes()
    if expected_size is not None:
        assert len(raw_bytes) == expected_size, (
            f"Frame deve ter {expected_size} bytes, tem {len(raw_bytes)}"
        )
    if ser is not None:
        ser.write(raw_bytes)
        ser.flush()
    return raw_bytes


def center_crop_1080(gray_frame):
    """Recorta 1080x1080 centralizado do frame (ou o maior quadrado possível)."""
    h, w = gray_frame.shape
    crop_size = min(h, w, 1080)
    y_start = (h - crop_size) // 2
    x_start = (w - crop_size) // 2
    return gray_frame[y_start:y_start + crop_size, x_start:x_start + crop_size]


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


def crop_face_rect(gray_frame, face_rect):
    """
    Recorta o rosto com padding de 15% — idêntico ao preprocessor.py do treinamento.
    Usa crop retangular (não força quadrado).
    """
    h_img, w_img = gray_frame.shape
    x, y, w, h = face_rect
    pad = int(w * 0.15)
    y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
    x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
    return gray_frame[y1:y2, x1:x2]


def detect_face_adaptive(gray_frame, face_cascade):
    """
    Detecção adaptativa em cascata — idêntica ao preprocessor.py do treinamento.
    Retorna (roi_cropped, face_rect) ou (None, None) se nenhum rosto encontrado.
    """
    detection_configs = [
        (1.2, 5, 60),
        (1.1, 3, 40),
        (1.05, 2, 30),
    ]
    for sf, mn, ms in detection_configs:
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms)
        )
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            roi = crop_face_rect(gray_frame, largest)
            return roi, largest
    return None, None


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
            img_bgr = cv2.imread(args.file)
            if img_bgr is None:
                print(f"[ERRO] Não foi possível ler {args.file}")
                sys.exit(1)

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # CLAHE — idêntico ao preprocessor.py do treinamento
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            out_frame = gray
            if face_cascade is not None:
                roi, rect = detect_face_adaptive(gray, face_cascade)
                if roi is not None:
                    out_frame = roi
                    print(f"[OK] Rosto detectado: {rect}")
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

    # === Modo webcam (captura contínua) ===
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera {args.camera}")
        sys.exit(1)

    print(f"[OK] Câmera {args.camera} aberta. Transmitindo vídeo ao vivo...")
    print("[INFO] Pressione 'q' na janela de preview ou Ctrl+C no terminal para sair.")

    # Estado da máquina de detecção
    # "detecting"  → procura rostos e envia vídeo 128x128
    # "cooldown"   → rosto enviado, pausa de FACE_COOLDOWN segundos
    # "grace"      → após cooldown, envia apenas vídeo por FACE_GRACE segundos
    state = "detecting"
    state_timer = 0.0

    try:
        frames_enviados = 0
        t_start = time.time()
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[ERRO] Falha na captura da câmera.")
                break

            now = time.time()

            # Converte para grayscale
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # CLAHE — idêntico ao preprocessor.py do treinamento
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            if state == "cooldown":
                # Aguarda FACE_COOLDOWN segundos sem enviar nada
                if now - state_timer >= FACE_COOLDOWN:
                    state = "grace"
                    state_timer = now
                    print("[INFO] Cooldown encerrado. Enviando apenas vídeo (grace period)...")
                continue

            if state == "grace":
                # Envia apenas vídeo 128x128 por FACE_GRACE segundos
                if now - state_timer >= FACE_GRACE:
                    state = "detecting"
                    print("[INFO] Grace period encerrado. Retomando detecção de rostos.")

                # Envia frame de vídeo 128x128
                cropped = center_crop_1080(gray)
                frame_video = cv2.resize(cropped, (VIDEO_SIZE, VIDEO_SIZE),
                                         interpolation=cv2.INTER_AREA)
                if not args.raw:
                    frame_video = quantize_to_q17(frame_video)
                send_control_byte(ser, CONTROL_NO_FACE)
                raw = send_frame(ser, frame_video, VIDEO_FRAME_BYTES)
                frames_enviados += 1
                preview_frame = frame_video
                preview_size = VIDEO_SIZE

            elif state == "detecting":
                face_found = False
                if face_cascade is not None:
                    roi, rect = detect_face_adaptive(gray, face_cascade)
                    if roi is not None:
                        face_found = True

                if face_found:
                    # Rosto detectado → envia 32x32
                    frame_32 = cv2.resize(roi, (IMG_SIZE, IMG_SIZE),
                                          interpolation=cv2.INTER_AREA)
                    if not args.raw:
                        frame_32 = quantize_to_q17(frame_32)
                    send_control_byte(ser, CONTROL_FACE)
                    raw = send_frame(ser, frame_32, FRAME_BYTES)
                    frames_enviados += 1
                    preview_frame = frame_32
                    preview_size = IMG_SIZE
                    print(f"[FACE] Rosto detectado e enviado: {rect}. Entrando em cooldown...")
                    state = "cooldown"
                    state_timer = now
                else:
                    # Sem rosto → envia vídeo 128x128
                    cropped = center_crop_1080(gray)
                    frame_video = cv2.resize(cropped, (VIDEO_SIZE, VIDEO_SIZE),
                                             interpolation=cv2.INTER_AREA)
                    if not args.raw:
                        frame_video = quantize_to_q17(frame_video)
                    send_control_byte(ser, CONTROL_NO_FACE)
                    raw = send_frame(ser, frame_video, VIDEO_FRAME_BYTES)
                    frames_enviados += 1
                    preview_frame = frame_video
                    preview_size = VIDEO_SIZE

            # Preview
            if args.preview:
                preview = cv2.resize(preview_frame, (256, 256), interpolation=cv2.INTER_NEAREST)
                label = "FACE 32x32" if preview_size == IMG_SIZE else "VIDEO 128x128"
                cv2.imshow(f"Preview ({label}) -> FPGA", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Dá um print esporádico para mostrar que está vivo
                if frames_enviados % 30 == 0:
                    fps = frames_enviados / (time.time() - t_start)
                    mode_str = state.upper()
                    print(f"[INFO] Frame {frames_enviados} [{mode_str}] (Média: {fps:.1f} fps)")


    except KeyboardInterrupt:
        print("\n[INFO] Captura interrompida pelo usuário.")

    cap.release()
    if args.preview:
        cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
    
    t_total = time.time() - t_start
    print(f"[FIM] Transmissão encerrada. Total de frames enviados: {frames_enviados} em {t_total:.1f}s.")


if __name__ == "__main__":
    main()
