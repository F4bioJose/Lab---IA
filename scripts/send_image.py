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
import os

# Suprime ABSOLUTAMENTE TODOS os warnings do Qt e do OpenCV no terminal do Linux
os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import numpy as np
import serial


def parse_args():
    parser = argparse.ArgumentParser(
        description="Envia frames da webcam (128x128 grayscale) via UART para a FPGA."
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
    parser.add_argument(
        "--mock", action="store_true",
        help="Modo de simulação (não tenta abrir a porta serial real)"
    )
    return parser.parse_args()


def send_frame(ser, frame: np.ndarray):
    """Envia um frame 128x128 (16384 bytes) via serial."""
    raw_bytes = frame.astype(np.uint8).tobytes()
    assert len(raw_bytes) == 16384, f"Frame deve ter 16384 bytes, tem {len(raw_bytes)}"
    if ser is not None:
        ser.write(raw_bytes)
        ser.flush()


def main():
    args = parse_args()

    # Abre a porta serial
    ser = None
    if not args.mock:
        try:
            ser = serial.Serial(args.port, args.baud, timeout=1)
            print(f"[OK] Porta serial {args.port} aberta a {args.baud} baud")
        except serial.SerialException as e:
            print(f"[ERRO] Não foi possível abrir {args.port}: {e}")
            print("[INFO] Dica: Use a flag --mock se quiser testar a câmera sem a placa conectada.")
            sys.exit(1)
    else:
        print(f"[MOCK] Rodando em modo de simulação. Os dados não serão enviados para {args.port}.")

    # Aguarda a FPGA estabilizar após reset
    time.sleep(0.1)

    if args.file:
        # === Modo arquivo estático ===
        img = cv2.imread(args.file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ERRO] Não foi possível ler {args.file}")
            sys.exit(1)
        frame = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        send_frame(ser, frame)
        print(f"[OK] Imagem '{args.file}' enviada ({frame.shape})")
        if ser is not None:
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

            # Converte para grayscale e redimensiona para 128x128
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_frame = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

            # Envia via UART o frame capturado
            send_frame(ser, out_frame)
            frame_count += 1

            # Mostra FPS a cada frame enviado
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  Frames enviados: {frame_count} | Tempo decorrido: {elapsed:.1f}s")

            # Delay para dar tempo à UART de transmitir (~1.4 segundos para 128x128)
            tx_time = (16384 * 10) / args.baud
            
            # Durante a espera da UART, mantemos a câmera gravando e exibindo para o preview ficar fluido a 30 FPS!
            t_end = time.time() + max(0, tx_time - 0.001)
            quit_req = False
            while time.time() < t_end:
                ret_live, live_frame = cap.read()
                if ret_live:
                    # Mostra o preview em tamanho maior para facilitar a visualização
                    preview_frame = cv2.resize(live_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Preview ao vivo da Webcam (aguardando envio...)", preview_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): # waitKey(1) para ~30fps
                    quit_req = True
                    break
            
            if quit_req:
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
        if ser is not None:
            ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
