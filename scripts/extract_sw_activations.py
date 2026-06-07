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

Mapeamento de Classes (19 classes — ordem Keras, string-sort):
  0 = Desconhecido (classe nativa da rede, Softmax)
  1 = Igor        | 2 = Joao      | 3 = Jose Henrique | 4 = Julia
  5 = Lucio       | 6 = Naira     | 7 = Rafael    | 8 = Samuel
  9 = Yuri        | 10 = Anna Carol | 11 = Bruno  | 12 = Diego
  13 = Eduardo    | 14 = Fabio    | 15 = Felipe   | 16 = Gabriel
  17 = Horacio    | 18 = Hugo

Uso:
    python scripts/extract_sw_activations.py \\
        --model tiny_cnn_multiclasse.h5 \\
        --image foto.jpg \\
        --cascade haarcascade_frontalface_default.xml \\
        --outdir comparacao/sw/

Formato dos arquivos de saída (mesmos do testbench Verilog):
    conv_out_sw.txt  — "f0 f1 f2 f3\\n" por linha (valores Q2.14 inteiros)
    pool_out_sw.txt  — "val\\n" por linha
    flat_out_sw.txt  — "val\\n" por linha
    dense_out_sw.txt — "s0 s1 ... s18\\n" em uma única linha
"""

import argparse
import os
import sys
from datetime import datetime
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
# Registro de classes customizadas do modelo (necessário para load_model)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notebooks_rede"))
    from quantization import Q17ClipConstraint
except ImportError:
    # Define uma versão stub caso o arquivo não seja encontrado
    print("[AVISO] quantization.py não encontrado — usando stub para Q17ClipConstraint.")
    @tf.keras.utils.register_keras_serializable(package="Lab1")
    class Q17ClipConstraint(tf.keras.constraints.Constraint):
        def __init__(self, frac_bits=7): self.frac_bits = frac_bits
        def __call__(self, w): return tf.clip_by_value(w, -1.0, (2**self.frac_bits-1)/(2**self.frac_bits))
        def get_config(self): return {"frac_bits": self.frac_bits}

# Mapeamento ID → nome da classe
CLASS_NAMES = [
    "Desconhecido",  # 0
    "Igor",          # 1
    "Joao",          # 2
    "Jose Henrique", # 3
    "Julia",         # 4
    "Lucio",         # 5
    "Naira",         # 6
    "Rafael",        # 7
    "Samuel",        # 8
    "Yuri",          # 9
    "Anna Carol",    # 10
    "Bruno",         # 11
    "Diego",         # 12
    "Eduardo",       # 13
    "Fabio",         # 14
    "Felipe",        # 15
    "Gabriel",       # 16
    "Horacio",       # 17
    "Hugo",          # 18
]


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

def quantize_q6_10(x: np.ndarray) -> np.ndarray:
    """Converte float → inteiro Q6.10 saturado em INT16."""
    return np.clip(np.round(x * (2 ** 10)), Q_MIN, Q_MAX).astype(np.int32)


# ---------------------------------------------------------------------------
# Pré-processamento com Haar Cascade
# ---------------------------------------------------------------------------
def preprocess_image(image_path: str, cascade_path: str, verbose: bool = True) -> np.ndarray:
    """
    Carrega imagem jpg, aplica CLAHE, detecta com adaptações, pad 15% (não força quadrado),
    redimensiona para 32x32 e normaliza para [0, 1].
    Retorna a imagem normalizada para Keras e a imagem crua (uint8).
    """
    if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
        raise ValueError("extract_sw_activations.py agora processa exclusivamente imagens (e.g. .jpg).")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: '{image_path}'")

    if verbose:
        print(f"[INFO] Imagem carregada: {image_path} | shape={img_bgr.shape}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Aplica CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2. Carrega cascade
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Arquivo Haar Cascade não encontrado: '{cascade_path}'")
    cascade = cv2.CascadeClassifier(cascade_path)

    # 3. Detecção Adaptativa
    detection_configs = [
        (1.2, 5, 60),
        (1.1, 3, 40),
        (1.05, 2, 30),
    ]

    faces = []
    for scale_factor, min_neighbors, min_size in detection_configs:
        faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size))
        if len(faces) > 0:
            break

    if len(faces) == 0:
        print("[AVISO] Nenhuma face detectada. Usando a imagem inteira.")
        roi = gray
    else:
        # Usa a primeira detecção (maior face)
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        # Pad idêntico ao preprocessor.py
        pad = int(w * 0.15)
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        roi = gray[y1:y2, x1:x2]
        if verbose:
            print(f"[INFO] Face detectada e cortada com 15% pad.")

    # 4. Redimensiona para 32×32
    roi_32 = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)

    if verbose:
        print(f"[INFO] ROI redimensionada para 32×32 | min={roi_32.min()}, max={roi_32.max()}")

    # 5. Normaliza para float32 [0, 1] e adiciona dims para Keras
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
            inputs=model.inputs,
            outputs=layer.output,
            name=f"sub_{key}"
        )
        print(f"[INFO] Camada '{key}': {layer.name} | output shape: {layer.output.shape}")

    return layer_models


def extract_activations(model: keras.Model, img_input: np.ndarray) -> dict:
    """
    Extrai ativações de todas as camadas intermediárias de interesse.

    Retorna:
        Dicionário {'conv': array, 'pool': array, 'flat': array, 'dense': array}
    """
    layer_models = build_layer_models(model)
    acts = {}

    for key, l_model in layer_models.items():
        if key == "dense":
            continue  # Faremos manualmente para pegar os logits antes do Softmax
            
        out = l_model.predict(img_input, verbose=0)
        acts[key] = out
        print(f"[INFO] '{key}' → shape={out.shape} | min={out.min():.4f}, max={out.max():.4f}")

    # Computa logits do Dense manualmente para ignorar o Softmax
    dense_layer = [l for l in model.layers if type(l).__name__ == "Dense"][-1]
    W, b = dense_layer.get_weights()  # W: (900, 19), b: (19,)
    flat_out = acts["flat"]           # (1, 900)
    logits = np.dot(flat_out, W) + b  # (1, 19)
    acts["dense"] = logits
    print(f"[INFO] 'dense' (logits) → shape={logits.shape} | min={logits.min():.4f}, max={logits.max():.4f}")

    return acts


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
    act: shape (1, N_CLASSES) → N_CLASSES scores
    Formato: "s0 s1 ... sN\n" em uma única linha
    Desconhecido = classe 0 (nativo da rede, Softmax).
    """
    data = act[0].flatten()
    n_classes = len(data)
    q = quantize_q6_10(data)

    with open(path, "w") as f:
        f.write(f"# Layer: dense | Image: {img_file} | "
                f"Format: s0..s{n_classes-1} (Q6.10 signed) | "
                f"Classe 0=Desconhecido, 1..{n_classes-1}=Pessoas\n")
        f.write(" ".join(str(v) for v in q) + "\n")

    pred_idx  = int(np.argmax(data))
    pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Classe_{pred_idx}"
    is_unknown = (pred_idx == 0)

    print(f"[INFO] Dense salvo: {path} ({n_classes} scores)")
    print(f"[INFO] Classe predita (SW): {pred_idx} = '{pred_name}' | "
          f"max_score: {data.max():.4f} ({'DESCONHECIDO' if is_unknown else 'IDENTIFICADO'})")


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
        default=None,
        help="Diretório de saída para os arquivos .txt (default: auto por imagem)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Salva ativações em float em vez de Q2.14 inteiro (para depuração)"
    )
    args = parser.parse_args()

    # Resolução do diretório de saída: subpasta automática por imagem
    img_stem = os.path.splitext(os.path.basename(args.image))[0]
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join("comparacao", "sw", img_stem)

    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("  CNN Layer Activation Extractor")
    print("=" * 60)
    print(f"  Modelo  : {args.model}")
    print(f"  Imagem  : {args.image}")
    print(f"  Cascade : {args.cascade}")
    print(f"  Saída   : {outdir}")
    print("=" * 60)

    # 1. Pré-processa a imagem
    img_input, roi_32 = preprocess_image(args.image, args.cascade)

    # Salva a ROI de 32×32 processada para inspeção visual
    roi_path = os.path.join(outdir, "roi_32x32.png")
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
        os.path.join(outdir, "conv_out_sw.txt"),
        img_basename
    )
    save_pool(
        activations["pool"],
        os.path.join(outdir, "pool_out_sw.txt"),
        img_basename
    )
    save_flat(
        activations["flat"],
        os.path.join(outdir, "flat_out_sw.txt"),
        img_basename
    )
    save_dense(
        activations["dense"],
        os.path.join(outdir, "dense_out_sw.txt"),
        img_basename
    )

    print("\n[OK] Extração concluída com sucesso.")
    print(f"     Arquivos salvos em: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
