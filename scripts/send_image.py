#!/usr/bin/env python3
# ==============================================================================
# Script: send_image.py
# Descrição: Captura frames da webcam, converte para grayscale 32×32 e envia 
#            via UART serial para a FPGA DE2-115.
#
# Dependências: pip install opencv-python pyserial numpy
#
# Uso:
#   python send_image.py                          # Porta padrão /dev/ttyUSB0
#   python send_image.py --port COM3              # Windows
#   python send_image.py --port /dev/ttyS0        # RS-232 nativo
#   python send_image.py --baud 921600            # Baud rate alternativo
#   python send_image.py --file imagem.png        # Enviar arquivo estático
# ==============================================================================

import argparse
import sys
import time

import cv2
import numpy as np
import serial


def parse_args():
    parser = argparse.ArgumentParser(
        description="Envia frames da webcam (32x32 grayscale) via UART para a FPGA."
    )
    parser.add_argument(
        "--port", default="/dev/ttyUSB0",
        help="Porta serial (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate (default: 115200)"
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
        "--once", action="store_true",
        help="Enviar apenas um frame e sair"
    )
    return parser.parse_args()


def send_frame(ser: serial.Serial, frame_32x32: np.ndarray):
    """Envia um frame 32×32 (1024 bytes) via serial."""
    raw_bytes = frame_32x32.astype(np.uint8).tobytes()
    assert len(raw_bytes) == 1024, f"Frame deve ter 1024 bytes, tem {len(raw_bytes)}"
    ser.write(raw_bytes)
    ser.flush()


def main():
    args = parse_args()

    # Abre a porta serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"[OK] Porta serial {args.port} aberta a {args.baud} baud")
    except serial.SerialException as e:
        print(f"[ERRO] Não foi possível abrir {args.port}: {e}")
        sys.exit(1)

    # Aguarda a FPGA estabilizar após reset
    time.sleep(0.1)

    if args.file:
        # === Modo arquivo estático ===
        img = cv2.imread(args.file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ERRO] Não foi possível ler {args.file}")
            sys.exit(1)
        small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        send_frame(ser, small)
        print(f"[OK] Imagem '{args.file}' enviada ({small.shape})")
        ser.close()
        return

    # === Modo webcam contínuo ===
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera {args.camera}")
        sys.exit(1)

    print(f"[OK] Câmera {args.camera} aberta. Pressione 'q' para sair.")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Falha na captura, tentando novamente...")
                continue

            # Converte para grayscale e redimensiona para 32×32
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

            # Envia via UART
            send_frame(ser, small)
            frame_count += 1

            # Exibe preview no PC
            preview = cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Enviando para FPGA (32x32)", preview)

            # Mostra FPS a cada 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Frames enviados: {frame_count} | FPS: {fps:.1f}")

            # Delay para dar tempo à UART de transmitir (1024 bytes × 10 bits / baud)
            tx_time = (1024 * 10) / args.baud
            time.sleep(max(0, tx_time - 0.001))  # Margem de 1ms

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if args.once:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário.")

    finally:
        elapsed = time.time() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n[FIM] Total: {frame_count} frames em {elapsed:.1f}s ({fps:.1f} FPS)")
        cap.release()
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
