#!/usr/bin/env python3
"""
extract_sw_activations.py
=========================
Extrai as ativações intermediárias de cada camada da CNN a partir de um
modelo Keras (.h5), replicando o pré-processamento do pipeline de hardware:

  1. Carrega qualquer imagem (PNG, JPG, BMP, etc.)
  2. Aplica o Haar Cascade para detecção e recorte da ROI (face)
  3. Converte para escala de cinza e redimensiona para 32×32
  4. Normaliza para float [0, 1] e aplica a quantização INT8 equivalente
  5. Extrai ativações de conv, pool, flatten e dense
  6. Salva cada camada em um arquivo .txt compatível com o testbench Verilog

Uso:
    python scripts/extract_sw_activations.py \\
        --model tiny_cnn_final_18.h5 \\
        --image foto.jpg \\
        --cascade haarcascade_frontalface_default.xml \\
        --outdir comparacao/sw/

Formato dos arquivos de saída (mesmos do testbench Verilog):
    conv_out_sw.txt  — "f0 f1 f2 f3\\n" por linha (valores Q2.14 inteiros)
    pool_out_sw.txt  — "val\\n" por linha
    flat_out_sw.txt  — "val\\n" por linha
    dense_out_sw.txt — "s0 s1 ... s17\\n" em uma única linha
"""

import argparse
import os
import sys
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Importação condicional do TensorFlow/Keras
# ---------------------------------------------------------------------------
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("[ERRO] TensorFlow/Keras não encontrado. Instale com: pip install tensorflow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constantes de quantização (devem espelhar o hardware)
# ---------------------------------------------------------------------------
Q_BITS   = 14          # Número de bits fracionários (Q2.14)
Q_SCALE  = 2 ** Q_BITS # 16384
Q_MAX    = 32767        # INT16 máximo
Q_MIN    = -32768       # INT16 mínimo


def quantize_q2_14(x: np.ndarray) -> np.ndarray:
    """Converte float → inteiro Q2.14 saturado em INT16."""
    return np.clip(np.round(x * Q_SCALE), Q_MIN, Q_MAX).astype(np.int32)


def dequantize_q2_14(x: np.ndarray) -> np.ndarray:
    """Converte inteiro Q2.14 → float (para verificação)."""
    return x.astype(np.float32) / Q_SCALE


# ---------------------------------------------------------------------------
# Pré-processamento com Haar Cascade
# ---------------------------------------------------------------------------
def preprocess_image(image_path: str, cascade_path: str, verbose: bool = True) -> np.ndarray:
    """
    Carrega qualquer imagem, detecta a região de interesse com Haar Cascade,
    recorta, converte para cinza 32×32 e normaliza para [0, 1].

    Retorna:
        np.ndarray de shape (1, 32, 32, 1) e dtype float32, pronto para Keras.
    """
    # Carrega a imagem (qualquer formato suportado pelo OpenCV)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: '{image_path}'")

    if verbose:
        print(f"[INFO] Imagem carregada: {image_path} | shape={img_bgr.shape}")

    # Converte para escala de cinza para a detecção Haar
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Carrega o classificador Haar Cascade
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Arquivo Haar Cascade não encontrado: '{cascade_path}'")

    cascade = cv2.CascadeClassifier(cascade_path)

    # Detecção da ROI
    faces = cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(16, 16),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        print("[AVISO] Nenhuma face detectada pelo Haar Cascade.")
        print("        Usando a imagem inteira como ROI (redimensionada para 32×32).")
        roi = img_gray
    else:
        # Usa a primeira detecção (maior confiança por área)
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        roi = img_gray[y:y+h, x:x+w]
        if verbose:
            print(f"[INFO] Face detectada: x={x}, y={y}, w={w}, h={h}")

    # Redimensiona para 32×32
    roi_32 = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)

    if verbose:
        print(f"[INFO] ROI redimensionada para 32×32 | min={roi_32.min()}, max={roi_32.max()}")

    # Normaliza para [0, 1] e adiciona dimensões de batch e canal
    roi_float = roi_32.astype(np.float32) / 255.0
    img_input = roi_float.reshape(1, 32, 32, 1)

    return img_input, roi_32


# ---------------------------------------------------------------------------
# Extração de ativações por camada
# ---------------------------------------------------------------------------
def build_layer_models(model: keras.Model) -> dict:
    """
    Constrói sub-modelos para extrair ativações intermediárias.
    Identifica as camadas pelo tipo (Conv2D, MaxPooling, Flatten, Dense).

    Retorna:
        Dicionário {'conv': model, 'pool': model, 'flat': model, 'dense': model}
    """
    layer_models = {}
    layer_map    = {}

    for layer in model.layers:
        ltype = type(layer).__name__
        if ltype == "Conv2D" and "conv" not in layer_map:
            layer_map["conv"] = layer
        elif ltype == "MaxPooling2D" and "pool" not in layer_map:
            layer_map["pool"] = layer
        elif ltype == "Flatten" and "flat" not in layer_map:
            layer_map["flat"] = layer
        elif ltype == "Dense" and "dense" not in layer_map:
            layer_map["dense"] = layer

    if not layer_map:
        raise ValueError("Nenhuma camada reconhecida encontrada no modelo.")

    for key, layer in layer_map.items():
        layer_models[key] = keras.Model(
            inputs=model.input,
            outputs=layer.output,
            name=f"sub_{key}"
        )
        print(f"[INFO] Camada '{key}': {layer.name} | output shape: {layer.output_shape}")

    return layer_models


def extract_activations(model: keras.Model, img_input: np.ndarray) -> dict:
    """
    Extrai ativações de todas as camadas intermediárias de interesse.

    Retorna:
        Dicionário {'conv': array, 'pool': array, 'flat': array, 'dense': array}
    """
    layer_models = build_layer_models(model)
    activations  = {}

    for key, sub_model in layer_models.items():
        act = sub_model.predict(img_input, verbose=0)
        activations[key] = act
        print(f"[INFO] '{key}' → shape={act.shape} | "
              f"min={act.min():.4f}, max={act.max():.4f}")

    return activations


# ---------------------------------------------------------------------------
# Gravação dos arquivos de saída (formato compatível com o testbench)
# ---------------------------------------------------------------------------
def save_conv(act: np.ndarray, path: str, img_file: str) -> None:
    """
    act: shape (1, H, W, 4) → (H*W, 4) amostras
    Formato: "f0 f1 f2 f3\\n" por linha (valores Q2.14 inteiros)
    """
    data = act[0]  # (H, W, 4)
    h, w, _ = data.shape
    flat = data.reshape(h * w, 4)  # (N, 4)
    q    = quantize_q2_14(flat)

    with open(path, "w") as f:
        f.write(f"# Layer: conv | Image: {img_file} | Format: f0 f1 f2 f3 (Q2.14 signed)\n")
        for row in q:
            f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")

    print(f"[INFO] Conv salvo: {path} ({h*w} linhas)")


def save_pool(act: np.ndarray, path: str, img_file: str) -> None:
    """
    act: shape (1, H, W, 4) → serializado canal por canal (como o HW faz)
    O hardware serializa os 4 canais de cada posição spatial em sequência.
    Formato: "val\\n" por linha
    """
    data = act[0]   # (H, W, 4)
    h, w, c = data.shape
    # Reordena para (H, W, 4) e serializa: para cada posição (y,x) emite c0,c1,c2,c3
    serial = data.reshape(h * w * c)
    q = quantize_q2_14(serial)

    with open(path, "w") as f:
        f.write(f"# Layer: pool | Image: {img_file} | Format: val (Q2.14 signed, canal serializado)\n")
        for v in q:
            f.write(f"{v}\n")

    print(f"[INFO] Pool salvo: {path} ({len(q)} linhas)")


def save_flat(act: np.ndarray, path: str, img_file: str) -> None:
    """
    act: shape (1, N) → N valores
    Formato: "val\\n" por linha
    """
    data = act[0].flatten()
    q = quantize_q2_14(data)

    with open(path, "w") as f:
        f.write(f"# Layer: flat | Image: {img_file} | Format: val (Q2.14 signed)\n")
        for v in q:
            f.write(f"{v}\n")

    print(f"[INFO] Flat salvo: {path} ({len(q)} linhas)")


def save_dense(act: np.ndarray, path: str, img_file: str) -> None:
    """
    act: shape (1, 18) → 18 scores
    Formato: "s0 s1 ... s17\\n" em uma única linha
    """
    data = act[0].flatten()
    q = quantize_q2_14(data)

    with open(path, "w") as f:
        f.write(f"# Layer: dense | Image: {img_file} | Format: s0..s17 (Q2.14 signed)\n")
        f.write(" ".join(str(v) for v in q) + "\n")

    print(f"[INFO] Dense salvo: {path} ({len(q)} scores)")
    print(f"[INFO] Classe predita (SW): {np.argmax(data)} | max_score: {data.max():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extrai ativações por camada da CNN para comparação SW/HW"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Caminho para o arquivo .h5 do modelo Keras (ex: tiny_cnn_final_18.h5)"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Caminho para a imagem de entrada (qualquer formato: PNG, JPG, BMP, etc.)"
    )
    parser.add_argument(
        "--cascade",
        default="haarcascade_frontalface_default.xml",
        help="Caminho para o XML do Haar Cascade (default: haarcascade_frontalface_default.xml)"
    )
    parser.add_argument(
        "--outdir",
        default="comparacao/sw/",
        help="Diretório de saída para os arquivos .txt (default: comparacao/sw/)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Salva ativações em float em vez de Q2.14 inteiro (para depuração)"
    )
    args = parser.parse_args()

    # Cria diretório de saída se não existir
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("  CNN Layer Activation Extractor")
    print("=" * 60)
    print(f"  Modelo  : {args.model}")
    print(f"  Imagem  : {args.image}")
    print(f"  Cascade : {args.cascade}")
    print(f"  Saída   : {args.outdir}")
    print("=" * 60)

    # 1. Pré-processa a imagem
    img_input, roi_32 = preprocess_image(args.image, args.cascade)

    # Salva a ROI de 32×32 processada para inspeção visual
    roi_path = os.path.join(args.outdir, "roi_32x32.png")
    cv2.imwrite(roi_path, roi_32)
    print(f"[INFO] ROI 32×32 salva em: {roi_path}")

    # 2. Carrega o modelo
    print(f"\n[INFO] Carregando modelo: {args.model}")
    model = keras.models.load_model(args.model)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # 3. Extrai ativações
    print("\n[INFO] Extraindo ativações por camada...")
    activations = extract_activations(model, img_input)

    # 4. Salva cada camada
    print("\n[INFO] Salvando arquivos de saída...")
    img_basename = os.path.basename(args.image)

    save_conv(
        activations["conv"],
        os.path.join(args.outdir, "conv_out_sw.txt"),
        img_basename
    )
    save_pool(
        activations["pool"],
        os.path.join(args.outdir, "pool_out_sw.txt"),
        img_basename
    )
    save_flat(
        activations["flat"],
        os.path.join(args.outdir, "flat_out_sw.txt"),
        img_basename
    )
    save_dense(
        activations["dense"],
        os.path.join(args.outdir, "dense_out_sw.txt"),
        img_basename
    )

    print("\n[OK] Extração concluída com sucesso.")
    print(f"     Arquivos salvos em: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
