#!/usr/bin/env python3
# ==============================================================================
# Script: send_image.py
# Descrição: Captura frames da webcam, detecta rostos com Haarcascade,
#            recorta e redimensiona para 64×64 grayscale e envia via UART
#            serial para a FPGA DE2-115 em velocidade máxima (~4-5 FPS).
#            Se nenhum rosto for detectado, mantém a última imagem válida.
#
# Dependências: pip install opencv-python pyserial numpy
#
# Uso:
#   python send_image.py                          # Porta padrão /dev/ttyUSB0
#   python send_image.py --port COM3              # Windows
#   python send_image.py --port /dev/ttyS0        # RS-232 nativo
#   python send_image.py --baud 230400            # Baud rate (padrão)
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


# Caminho do classificador Haarcascade (relativo à raiz do projeto)
HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "haarcascade_frontalface_default.xml"
)

# Resolução da imagem enviada (64×64 = 4096 bytes)
IMG_SIZE = 64
FRAME_BYTES = IMG_SIZE * IMG_SIZE  # 4096


def parse_args():
    parser = argparse.ArgumentParser(
        description="Envia frames da webcam (64x64 grayscale) via UART para a FPGA."
    )
    parser.add_argument(
        "--port", default="/dev/ttyUSB0",
        help="Porta serial (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baud", type=int, default=230400,
        help="Baud rate (default: 230400)"
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
    parser.add_argument(
        "--no-haar", action="store_true",
        help="Desabilitar detecção de rosto (enviar frame completo)"
    )
    return parser.parse_args()


def send_frame(ser, frame: np.ndarray):
    """Envia um frame 64x64 (4096 bytes) via serial."""
    raw_bytes = frame.astype(np.uint8).tobytes()
    assert len(raw_bytes) == FRAME_BYTES, f"Frame deve ter {FRAME_BYTES} bytes, tem {len(raw_bytes)}"
    if ser is not None:
        ser.write(raw_bytes)
        ser.flush()


def crop_face_square(gray_frame, face_rect):
    """
    Recorta o rosto expandindo para um quadrado centrado no retângulo detectado.
    Isso evita distorção ao redimensionar para 64×64.
    """
    h_img, w_img = gray_frame.shape
    x, y, w, h = face_rect

    # Expande para quadrado usando o lado maior
    side = max(w, h)

    # Adiciona margem de 30% para incluir testa e queixo
    margin = int(side * 0.3)
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
    if not args.no_haar:
        if not os.path.isfile(HAAR_CASCADE_PATH):
            print(f"[AVISO] Haarcascade não encontrado em {HAAR_CASCADE_PATH}")
            print("[AVISO] Tentando usar o caminho padrão do OpenCV...")
            alt_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.isfile(alt_path):
                face_cascade = cv2.CascadeClassifier(alt_path)
            else:
                print("[ERRO] Haarcascade não encontrado. Use --no-haar para desabilitar.")
                sys.exit(1)
        else:
            face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

        if face_cascade is not None and face_cascade.empty():
            print("[ERRO] Falha ao carregar o classificador Haarcascade.")
            sys.exit(1)

        print(f"[OK] Haarcascade carregado com sucesso")

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

        out_frame = img
        # Aplica Haarcascade no arquivo estático se habilitado
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                # Pega o maior rosto
                largest = max(faces, key=lambda f: f[2] * f[3])
                out_frame = crop_face_square(img, largest)
                print(f"[OK] Rosto detectado: {largest}")
            else:
                print("[AVISO] Nenhum rosto detectado na imagem estática, enviando imagem completa.")

        frame = cv2.resize(out_frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
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

    tx_time_est = FRAME_BYTES * 10 / args.baud
    print(f"[OK] Câmera {args.camera} aberta. Pressione 'q' para sair.")
    if face_cascade is not None:
        print(f"[OK] Detecção de rosto ATIVA. TX estimado: {tx_time_est:.3f}s (~{1/tx_time_est:.0f} FPS max)")
    else:
        print(f"[OK] Detecção de rosto DESABILITADA (--no-haar). TX estimado: {tx_time_est:.3f}s")

    frame_count = 0
    t_start = time.time()
    last_valid_face = None  # Armazena o último rosto válido detectado

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Falha na captura, tentando novamente...")
                continue

            # Converte para grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # === Detecção de rosto com Haarcascade ===
            face_detected = False
            if face_cascade is not None:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(faces) > 0:
                    # Pega o maior rosto detectado
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    face_crop = crop_face_square(gray, largest)
                    out_frame = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                    last_valid_face = out_frame.copy()
                    face_detected = True

                    # Desenha retângulo verde no preview
                    x, y, w, h = largest
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "ROSTO DETECTADO", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Nenhum rosto detectado — usa o último válido
                    if last_valid_face is not None:
                        out_frame = last_valid_face
                        cv2.putText(frame, "SEM ROSTO - usando ultimo", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # Primeira iteração sem rosto: fallback para frame completo
                        out_frame = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                        cv2.putText(frame, "SEM ROSTO - frame completo", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Haarcascade desabilitado: envia frame completo
                out_frame = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            # Envia via UART o frame processado (bloqueia ~0.18s a 230400 baud)
            t_send = time.time()
            send_frame(ser, out_frame)
            tx_elapsed = time.time() - t_send
            frame_count += 1

            # Mostra status com FPS instantâneo
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            status = "ROSTO" if face_detected else ("ULTIMO" if last_valid_face is not None else "COMPLETO")
            print(f"  Frame {frame_count} [{status}] | TX: {tx_elapsed:.3f}s | {fps:.1f} FPS")

            # Preview rápido (sem delay adicional — velocidade máxima)
            if face_cascade is not None:
                for (fx, fy, fw, fh) in (faces if len(faces) > 0 else []):
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            preview_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Preview (Haarcascade + UART)", preview_frame)

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
        if ser is not None:
            ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
