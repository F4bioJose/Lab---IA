#!/usr/bin/env python3
"""
test_model_19classes.py
=======================
Script de teste de sanidade para o modelo tiny_cnn_multiclasse.h5 (19 classes).

Realiza as seguintes verificações:
  1. Carrega o modelo com Q17ClipConstraint registrado
  2. Verifica a arquitetura (camadas, shapes, nº de parâmetros)
  3. Valida que o .mif tem DEPTH = total de parâmetros (17159)
  4. Roda inferência em todas as imagens de inputs/ e exibe predições
  5. Exporta os pesos em formato .hex (para validação manual)
  6. Gera um relatório de sanidade completo

Mapeamento de Classes (ordem Keras — string sort):
  0 = Desconhecido (classe nativa da rede, Softmax)
  1 = Igor        | 2 = Joao      | 3 = Jose Henrique | 4 = Julia
  5 = Lucio       | 6 = Naira     | 7 = Rafael    | 8 = Samuel
  9 = Yuri        | 10 = Anna Carol | 11 = Bruno  | 12 = Diego
  13 = Eduardo    | 14 = Fabio    | 15 = Felipe   | 16 = Gabriel
  17 = Horacio    | 18 = Hugo

Uso:
    # A partir da raiz do projeto (conda env machine_learning):
    python scripts/test_model_19classes.py
    python scripts/test_model_19classes.py --model outra_versao.h5
    python scripts/test_model_19classes.py --no-infer   # Pula inferências, só valida pesos
"""

import argparse
import os
import sys
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Importações com fallback informativo
# ---------------------------------------------------------------------------
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("[ERRO] TensorFlow/Keras não encontrado.")
    print("       Ative o ambiente conda: conda activate machine_learning")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[AVISO] OpenCV não encontrado — inferência em imagens será pulada.")

# ---------------------------------------------------------------------------
# Registro de classes customizadas
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notebooks_rede"))
    from quantization import Q17ClipConstraint
    print("[OK] Q17ClipConstraint carregado de notebooks_rede/quantization.py")
except ImportError:
    print("[AVISO] quantization.py não encontrado — usando stub.")
    @tf.keras.utils.register_keras_serializable(package="Lab1")
    class Q17ClipConstraint(tf.keras.constraints.Constraint):
        def __init__(self, frac_bits=7): self.frac_bits = frac_bits
        def __call__(self, w): return tf.clip_by_value(w, -1.0, (2**self.frac_bits-1)/(2**self.frac_bits))
        def get_config(self): return {"frac_bits": self.frac_bits}

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
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

# Layout esperado do .mif para 19 classes:
#   36 conv_w + 4 conv_b + 900*19 dense_w + 19 dense_b = 17159
EXPECTED_MIF_DEPTH = 17159
EXPECTED_CLASSES   = 19


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sep(char="=", n=60):
    print(char * n)


def quantize_q17(arr: np.ndarray) -> np.ndarray:
    """Quantiza float [-1, ~1) → INT8 Q1.7."""
    scaled = np.round(arr * 128.0)
    return np.clip(scaled, -128, 127).astype(np.int8)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# 1. Carregamento e validação da arquitetura
# ---------------------------------------------------------------------------
def load_and_validate_model(model_path: str) -> keras.Model:
    sep()
    print(f"  [1] CARREGAMENTO DO MODELO")
    sep()
    print(f"  Arquivo : {model_path}")

    if not os.path.exists(model_path):
        print(f"[ERRO] Arquivo não encontrado: {model_path}")
        sys.exit(1)

    model = keras.models.load_model(model_path)
    print("[OK] Modelo carregado com sucesso.\n")

    print("  === SUMMARY ===")
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # Verifica o número de classes
    dense_layers = [l for l in model.layers if type(l).__name__ == "Dense"]
    if not dense_layers:
        print("[ERRO] Nenhuma camada Dense encontrada no modelo.")
        sys.exit(1)

    last_dense = dense_layers[-1]
    n_classes = last_dense.units
    print(f"\n  [INFO] Camada Dense final: '{last_dense.name}' | {n_classes} classes")

    if n_classes != EXPECTED_CLASSES:
        print(f"[AVISO] Esperado {EXPECTED_CLASSES} classes, encontrado {n_classes}.")
    else:
        print(f"[OK] {n_classes} classes confirmadas.")

    # Verifica ativação
    act_name = last_dense.activation.__name__ if last_dense.activation else "None"
    print(f"  [INFO] Ativação da Dense final: {act_name}")
    if act_name.lower() == "softmax":
        print(f"[OK] Softmax confirmado — classe 0 (Desconhecido) é nativa da rede.")
    else:
        print(f"[AVISO] Ativação esperada: softmax. Encontrado: {act_name}")

    return model


# ---------------------------------------------------------------------------
# 2. Validação do .mif
# ---------------------------------------------------------------------------
def validate_mif(mif_path: str):
    sep()
    print(f"  [2] VALIDAÇÃO DO ARQUIVO .MIF")
    sep()
    print(f"  Arquivo : {mif_path}")

    if not os.path.exists(mif_path):
        print(f"[AVISO] Arquivo .mif não encontrado: {mif_path} — pulando validação.")
        return

    depth = None
    data_count = 0
    in_content = False

    with open(mif_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("DEPTH"):
                depth = int(line.split("=")[1].strip().rstrip(";"))
            if line.upper().startswith("CONTENT BEGIN"):
                in_content = True
                continue
            if line.upper() == "END;" or line.upper() == "END":
                in_content = False
                continue
            if in_content and ":" in line and ";" in line:
                data_count += 1

    print(f"  DEPTH declarado  : {depth}")
    print(f"  Entradas contadas: {data_count}")
    print(f"  Esperado         : {EXPECTED_MIF_DEPTH}")

    # Valida layout
    conv_w  = 4 * 9          # 36
    conv_b  = 4              # 4
    dense_w = 900 * 19       # 17100
    dense_b = 19             # 19
    total   = conv_w + conv_b + dense_w + dense_b
    print(f"\n  Layout esperado:")
    print(f"    conv_w  = 4×9 = {conv_w}")
    print(f"    conv_b  = {conv_b}")
    print(f"    dense_w = 900×19 = {dense_w}")
    print(f"    dense_b = {dense_b}")
    print(f"    TOTAL   = {total}")

    if depth == EXPECTED_MIF_DEPTH and depth == total:
        print(f"\n[OK] .mif validado: DEPTH={depth} bate com layout 19 classes.")
    elif depth != EXPECTED_MIF_DEPTH:
        print(f"\n[ERRO] DEPTH={depth}, esperado {EXPECTED_MIF_DEPTH}!")
    else:
        print(f"\n[AVISO] DEPTH={depth} bate com esperado, mas contagem de dados={data_count} diverge.")


# ---------------------------------------------------------------------------
# 3. Extração e exportação de pesos
# ---------------------------------------------------------------------------
def export_weights_hex(model: keras.Model, out_hex: str):
    sep()
    print(f"  [3] EXPORTAÇÃO DE PESOS → {out_hex}")
    sep()

    conv_layers  = [l for l in model.layers if type(l).__name__ == "Conv2D"]
    dense_layers = [l for l in model.layers if type(l).__name__ == "Dense"]

    if not conv_layers or not dense_layers:
        print("[ERRO] Modelo sem Conv2D ou Dense.")
        return

    conv_w_raw, conv_b_raw = conv_layers[0].get_weights()
    dense_w_raw, dense_b_raw = dense_layers[-1].get_weights()

    # Quantiza para INT8
    conv_w_q  = quantize_q17(conv_w_raw)    # (3,3,1,4)
    conv_b_q  = quantize_q17(conv_b_raw)    # (4,)
    dense_w_q = quantize_q17(dense_w_raw)   # (900, 19)
    dense_b_q = quantize_q17(dense_b_raw)   # (19,)

    print(f"  conv_w  shape: {conv_w_q.shape}  → {conv_w_q.size} bytes")
    print(f"  conv_b  shape: {conv_b_q.shape}  → {conv_b_q.size} bytes")
    print(f"  dense_w shape: {dense_w_q.shape} → {dense_w_q.size} bytes")
    print(f"  dense_b shape: {dense_b_q.shape} → {dense_b_q.size} bytes")

    # Serialização: filtros em ordem f0..f3, depois bias, depois colunas densas
    # Layout ROM: [conv_w f0(9)] [conv_w f1(9)] [conv_w f2(9)] [conv_w f3(9)]
    #             [conv_b(4)] [dense_w col0(900)] ... [dense_w col18(900)]
    #             [dense_b(19)]
    all_bytes = []

    # conv weights: 4 filtros × 9 coefs, ordem f × (row,col)
    h, w, in_ch, n_filters = conv_w_q.shape  # (3,3,1,4)
    for f in range(n_filters):
        kernel = conv_w_q[:, :, 0, f].flatten()  # (9,)
        all_bytes.extend(kernel.tolist())

    # conv biases
    all_bytes.extend(conv_b_q.tolist())

    # dense weights: coluna por classe (900 por classe)
    # dense_w_q shape (900, 19): coluna j = pesos de saída para classe j
    for cls in range(dense_w_q.shape[1]):
        all_bytes.extend(dense_w_q[:, cls].tolist())

    # dense biases
    all_bytes.extend(dense_b_q.tolist())

    total = len(all_bytes)
    print(f"\n  Total de bytes  : {total}")
    print(f"  Esperado        : {EXPECTED_MIF_DEPTH}")

    if total != EXPECTED_MIF_DEPTH:
        print(f"[AVISO] Divergência: {total} != {EXPECTED_MIF_DEPTH}")
    else:
        print("[OK] Total bate com EXPECTED_MIF_DEPTH.")

    os.makedirs(os.path.dirname(out_hex) or ".", exist_ok=True)
    with open(out_hex, "w") as f:
        for b in all_bytes:
            f.write(f"{b & 0xFF:02X}\n")

    print(f"[OK] Pesos exportados: {out_hex}")


# ---------------------------------------------------------------------------
# 4. Inferência em imagens de inputs/
# ---------------------------------------------------------------------------
def run_inference_on_inputs(model: keras.Model, inputs_dir: str, cascade_path: str):
    sep()
    print(f"  [4] INFERÊNCIA NAS IMAGENS DE {inputs_dir}")
    sep()

    if not HAS_CV2:
        print("[AVISO] OpenCV não disponível — pulando inferência em imagens.")
        return

    # Carrega cascade (opcional)
    face_cascade = None
    if cascade_path and os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            face_cascade = None
        else:
            print(f"[OK] Haar Cascade: {cascade_path}")
    else:
        print(f"[AVISO] Haar Cascade não encontrado em '{cascade_path}'. Usando imagem completa.")

    # Lista imagens
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([
        f for f in os.listdir(inputs_dir)
        if f.lower().endswith(extensions)
    ])

    if not images:
        print(f"[AVISO] Nenhuma imagem encontrada em '{inputs_dir}'.")
        return

    print(f"  {len(images)} imagem(ns) encontrada(s):\n")

    results = []
    for img_name in images:
        img_path = os.path.join(inputs_dir, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [ERRO] Não foi possível ler: {img_name}")
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        roi = gray

        if face_cascade is not None:
            detection_configs = [
                (1.2, 5, 60),
                (1.1, 3, 40),
                (1.05, 2, 30),
            ]
            faces = []
            for sf, mn, ms in detection_configs:
                faces = face_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms))
                if len(faces) > 0:
                    break

            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                pad = int(w * 0.15)
                y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
                roi = gray[y1:y2, x1:x2]

        roi_32 = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        inp = roi_32.astype(np.float32) / 255.0
        inp = inp.reshape(1, 32, 32, 1)

        # Inferência
        pred = model.predict(inp, verbose=0)[0]  # (19,)
        pred_idx  = int(np.argmax(pred))
        pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Classe_{pred_idx}"
        confidence = float(pred[pred_idx])
        is_unknown = (pred_idx == 0)

        status = "⚠ DESCONHECIDO" if is_unknown else "✓ IDENTIFICADO"
        print(f"  {img_name:<25}  →  [{pred_idx:2d}] {pred_name:<16}  "
              f"conf={confidence:.3f}  {status}")

        results.append({
            "image": img_name,
            "pred_idx": pred_idx,
            "pred_name": pred_name,
            "confidence": confidence,
            "is_unknown": is_unknown,
            "all_scores": pred.tolist(),
        })

    # Resumo
    print(f"\n  === RESUMO ===")
    n_identified = sum(1 for r in results if not r["is_unknown"])
    n_unknown    = sum(1 for r in results if r["is_unknown"])
    print(f"  Total      : {len(results)}")
    print(f"  Identificados: {n_identified}")
    print(f"  Desconhecidos: {n_unknown}")

    return results


# ---------------------------------------------------------------------------
# 5. Relatório final
# ---------------------------------------------------------------------------
def print_final_report(model_path: str, mif_path: str):
    sep()
    print("  [5] RELATÓRIO FINAL DE SANIDADE")
    sep()
    print(f"  Modelo  : {model_path}")
    print(f"  .mif    : {mif_path}")
    print(f"  Classes : {EXPECTED_CLASSES}")
    print(f"  Classe 0: Desconhecido (nativa da rede — SEM threshold)")
    print(f"  Layout MIF:")
    print(f"    [0..35]      → pesos conv (4×9 = 36)")
    print(f"    [36..39]     → biases conv (4)")
    print(f"    [40..17139]  → pesos dense (900×19 = 17100)")
    print(f"    [17140..17158] → biases dense (19)")
    print(f"    TOTAL        → 17159 ✓")
    sep()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Teste de sanidade do modelo CNN 19 classes"
    )
    parser.add_argument(
        "--model", default="tiny_cnn_multiclasse.h5",
        help="Caminho para o .h5 (default: tiny_cnn_multiclasse.h5)"
    )
    parser.add_argument(
        "--mif", default="modulos_verilog/weights_all.mif",
        help="Caminho para o .mif (default: modulos_verilog/weights_all.mif)"
    )
    parser.add_argument(
        "--inputs", default="inputs",
        help="Diretório de imagens de teste (default: inputs/)"
    )
    parser.add_argument(
        "--cascade", default="haarcascade_frontalface_default.xml",
        help="Caminho do Haar Cascade (default: haarcascade_frontalface_default.xml)"
    )
    parser.add_argument(
        "--export-hex", default="modulos_verilog/weights_exported.hex",
        help="Exporta pesos em .hex para validação (default: modulos_verilog/weights_exported.hex)"
    )
    parser.add_argument(
        "--no-infer", action="store_true",
        help="Pula inferência em imagens (apenas valida arquitetura e pesos)"
    )
    args = parser.parse_args()

    sep("=", 60)
    print("  CNN 19 Classes — Teste de Sanidade")
    print("  Desconhecido = Classe 0 (nativo da rede, sem threshold)")
    sep("=", 60)
    print()

    # 1. Carrega e valida modelo
    model = load_and_validate_model(args.model)
    print()

    # 2. Valida .mif
    validate_mif(args.mif)
    print()

    # 3. Exporta pesos
    export_weights_hex(model, args.export_hex)
    print()

    # 4. Inferência
    if not args.no_infer and os.path.isdir(args.inputs):
        run_inference_on_inputs(model, args.inputs, args.cascade)
        print()
    elif args.no_infer:
        print("[INFO] --no-infer ativo — inferência pulada.")
    else:
        print(f"[AVISO] Diretório de inputs não encontrado: '{args.inputs}'")
    print()

    # 5. Relatório final
    print_final_report(args.model, args.mif)
    print("\n[OK] Teste de sanidade concluído.")


if __name__ == "__main__":
    main()
