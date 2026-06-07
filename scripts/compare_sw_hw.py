#!/usr/bin/env python3
"""
compare_sw_hw.py
================
Compara as ativações intermediárias da CNN entre a implementação em software
(Python/Keras, arquivos *_sw.txt) e a implementação em hardware (Verilog/ModelSim,
arquivos sem sufixo _sw).

Gera métricas quantitativas e os seguintes gráficos:
  1. Histogramas de erro (hw - sw) por camada
  2. Scatter plots SW × HW por camada (com linha ideal y=x)
  3. Barplot dos 19 scores finais SW × HW (classe 0 = Desconhecido destacado)
  4. Mapa de calor de correlação dos canais da camada conv

Uso:
    python scripts/compare_sw_hw.py \\
        --sw-dir comparacao/sw/ \\
        --hw-dir comparacao/hw/ \\
        --outdir comparacao/plots/

Saídas (em --outdir):
    01_hist_errors.png     — Histogramas de erro por camada
    02_scatter_layers.png  — Scatter SW × HW por camada
    03_barplot_scores.png  — Comparação de scores finais (19 classes)
    04_heatmap_conv.png    — Correlação dos canais conv
    report.txt             — Resumo textual das métricas
"""

import argparse
import os
import sys
import textwrap
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # Sem interface gráfica (roda em servidores)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("[ERRO] matplotlib não encontrado. Instale com: pip install matplotlib")
    sys.exit(1)

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[AVISO] scipy não encontrado — correlação de Pearson desativada.")

# ---------------------------------------------------------------------------
# Configuração visual
# ---------------------------------------------------------------------------
COLORS = {
    "sw"       : "#4C72B0",   # Azul suave
    "hw"       : "#DD8452",   # Laranja suave
    "unknown"  : "#C44E52",   # Vermelho (classe 0 = Desconhecido)
    "error"    : "#55A868",   # Verde
    "ideal"    : "#808080",   # Cinza
}
FIGSIZE_WIDE  = (18, 4)
FIGSIZE_SQUARE = (12, 10)
DPI = 150

Q_SCALE = 2 ** 14  # 16384 (fator Q2.14)

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
# Helpers de leitura
# ---------------------------------------------------------------------------
def read_matrix(path: str) -> np.ndarray:
    """
    Lê um arquivo .txt gerado pelo testbench ou pelo extract_sw_activations.py.
    Ignora linhas de comentário (iniciadas com '#').
    Retorna np.ndarray 2D (linhas × colunas).
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([int(v) for v in line.split()])
    return np.array(rows, dtype=np.int32)


def load_layer(sw_dir: str, hw_dir: str, sw_name: str, hw_name: str, label: str):
    """
    Carrega e valida os dados SW e HW de uma camada.
    Retorna (sw_flat, hw_flat) como arrays 1D, ou (None, None) se falhar.
    """
    sw_path = os.path.join(sw_dir, sw_name)
    hw_path = os.path.join(hw_dir, hw_name)

    for p in [sw_path, hw_path]:
        if not os.path.exists(p):
            print(f"[AVISO] Arquivo não encontrado: {p} → camada '{label}' ignorada.")
            return None, None

    sw = read_matrix(sw_path).flatten()
    hw = read_matrix(hw_path).flatten()

    if sw.shape != hw.shape:
        print(f"[AVISO] Dimensão diferente em '{label}': SW={sw.shape}, HW={hw.shape}")
        min_len = min(len(sw), len(hw))
        sw, hw = sw[:min_len], hw[:min_len]

    print(f"[INFO] '{label}': {len(sw)} amostras carregadas.")
    return sw, hw


# ---------------------------------------------------------------------------
# Cálculo de métricas
# ---------------------------------------------------------------------------
def compute_metrics(sw: np.ndarray, hw: np.ndarray, label: str) -> dict:
    """Calcula e retorna métricas de diferença entre SW e HW."""
    diff  = hw.astype(np.float64) - sw.astype(np.float64)
    mae   = np.mean(np.abs(diff))
    mse   = np.mean(diff ** 2)
    rmse  = np.sqrt(mse)
    max_e = np.max(np.abs(diff))
    exact = np.sum(diff == 0) / len(diff) * 100.0

    corr = None
    if HAS_SCIPY and len(sw) > 1:
        corr, _ = pearsonr(sw.astype(np.float64), hw.astype(np.float64))

    metrics = {
        "label"  : label,
        "n"      : len(sw),
        "mae"    : mae,
        "mse"    : mse,
        "rmse"   : rmse,
        "max_err": max_e,
        "exact%"  : exact,
        "pearson": corr,
    }

    print(f"  [{label}] MAE={mae:.2f} | RMSE={rmse:.2f} | MaxErr={max_e:.0f} "
          f"| Exact={exact:.1f}%" +
          (f" | r={corr:.4f}" if corr is not None else ""))
    return metrics


# ---------------------------------------------------------------------------
# Plot 1: Histogramas de erro por camada
# ---------------------------------------------------------------------------
def plot_histograms(layers_data: dict, out_path: str) -> None:
    n = len(layers_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), dpi=DPI)
    if n == 1:
        axes = [axes]

    for ax, (label, (sw, hw)) in zip(axes, layers_data.items()):
        diff = hw.astype(np.float64) - sw.astype(np.float64)
        ax.hist(diff, bins=60, color=COLORS["error"], edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Erro = 0")
        ax.set_title(f"Camada: {label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Erro  HW − SW  (unidades quantizadas)", fontsize=10)
        ax.set_ylabel("Contagem", fontsize=10)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Histogramas de Erro por Camada  (HW − SW)", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Histogramas salvos: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Scatter SW × HW por camada
# ---------------------------------------------------------------------------
def plot_scatters(layers_data: dict, out_path: str) -> None:
    n = len(layers_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=DPI)
    if n == 1:
        axes = [axes]

    for ax, (label, (sw, hw)) in zip(axes, layers_data.items()):
        # Limita o número de pontos para legibilidade
        max_pts = 5000
        if len(sw) > max_pts:
            idx = np.random.choice(len(sw), max_pts, replace=False)
            sw_p, hw_p = sw[idx], hw[idx]
        else:
            sw_p, hw_p = sw, hw

        ax.scatter(sw_p, hw_p, s=3, alpha=0.35, color=COLORS["sw"], label="Amostra")

        # Linha ideal y = x
        lim = [min(sw_p.min(), hw_p.min()), max(sw_p.max(), hw_p.max())]
        ax.plot(lim, lim, color=COLORS["ideal"], linestyle="--",
                linewidth=1.5, label="Ideal (y = x)")

        ax.set_title(f"Camada: {label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("SW (Quantizado)", fontsize=10)
        ax.set_ylabel("HW (Quantizado)", fontsize=10)
        ax.legend(fontsize=9, markerscale=3)
        ax.grid(alpha=0.3)

    fig.suptitle("Scatter SW × HW por Camada", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Scatter plots salvos: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Barplot dos 19 scores finais SW × HW
# Classe 0 (Desconhecido) destacada em vermelho
# ---------------------------------------------------------------------------
def plot_scores(sw_scores: np.ndarray, hw_scores: np.ndarray, out_path: str) -> None:
    n_classes = max(len(sw_scores), len(hw_scores))
    x = np.arange(n_classes)
    width = 0.38

    fig, ax = plt.subplots(figsize=(16, 5), dpi=DPI)

    # Cores: classe 0 (Desconhecido) em vermelho, demais em azul/laranja
    sw_colors = [COLORS["unknown"] if i == 0 else COLORS["sw"] for i in range(n_classes)]
    hw_colors = [COLORS["unknown"] if i == 0 else COLORS["hw"] for i in range(n_classes)]

    bars_sw = ax.bar(x - width / 2, sw_scores[:n_classes], width,
                     label="SW (Keras)", color=sw_colors, alpha=0.85, zorder=3)
    bars_hw = ax.bar(x + width / 2, hw_scores[:n_classes], width,
                     label="HW (Verilog)", color=hw_colors, alpha=0.85, zorder=3)

    # Linha zero para referência
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, zorder=4)

    # Labels curtos para o eixo X
    short_labels = ["Desc"] + [f"{i}" for i in range(1, n_classes)]
    ax.set_xlabel("Classe", fontsize=11)
    ax.set_ylabel("Score (Q6.10)", fontsize=11)
    ax.set_title("Scores Finais — SW vs. HW (19 Classes | Classe 0 = Desconhecido)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Anotação da classe predita
    sw_pred = int(np.argmax(sw_scores))
    hw_pred = int(np.argmax(hw_scores))
    sw_label = CLASS_NAMES[sw_pred] if sw_pred < len(CLASS_NAMES) else f"C{sw_pred}"
    hw_label = CLASS_NAMES[hw_pred] if hw_pred < len(CLASS_NAMES) else f"C{hw_pred}"
    ax.annotate(f"SW→{sw_label}", xy=(sw_pred - width/2, sw_scores[sw_pred]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=COLORS["sw"], fontweight="bold")
    ax.annotate(f"HW→{hw_label}", xy=(hw_pred + width/2, hw_scores[hw_pred]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=COLORS["hw"], fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Barplot de scores salvo: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Mapa de calor de correlação dos canais conv
# ---------------------------------------------------------------------------
def plot_conv_heatmap(sw_conv: np.ndarray, hw_conv: np.ndarray, out_path: str) -> None:
    """
    Correlação entre os 4 canais de sw e os 4 canais de hw.
    sw_conv e hw_conv são 1D serializados (N × 4 colunas aplanadas).
    Tenta reconstruir para (N, 4) — se N não for múltiplo de 4, pula.
    """
    try:
        n_total = len(sw_conv)
        if n_total % 4 != 0:
            print("[AVISO] Heatmap conv: total de amostras não é múltiplo de 4 → ignorado.")
            return

        n_pixels = n_total // 4
        sw_ch = sw_conv.reshape(n_pixels, 4)
        hw_ch = hw_conv.reshape(n_pixels, 4)

        # Matriz de correlação 8×8 (sw_c0..sw_c3, hw_c0..hw_c3)
        combined = np.hstack([sw_ch, hw_ch])  # (N, 8)
        labels = [f"SW_f{i}" for i in range(4)] + [f"HW_f{i}" for i in range(4)]
        corr_matrix = np.corrcoef(combined.T)

        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=DPI)
        im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Correlação de Pearson")

        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title("Mapa de Correlação — Canais Conv SW × HW", fontsize=13,
                     fontweight="bold")

        # Anota os valores na célula
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=8, color="black" if abs(corr_matrix[i,j]) < 0.7 else "white")

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] Heatmap de correlação conv salvo: {out_path}")
    except Exception as e:
        print(f"[AVISO] Não foi possível gerar heatmap conv: {e}")


# ---------------------------------------------------------------------------
# Relatório textual
# ---------------------------------------------------------------------------
def save_report(metrics_list: list, out_path: str,
                sw_pred: int, hw_pred: int, agreement: bool) -> None:
    sw_name = CLASS_NAMES[sw_pred] if sw_pred < len(CLASS_NAMES) else f"Classe_{sw_pred}"
    hw_name = CLASS_NAMES[hw_pred] if hw_pred < len(CLASS_NAMES) else f"Classe_{hw_pred}"
    sw_status = "DESCONHECIDO" if sw_pred == 0 else "IDENTIFICADO"
    hw_status = "DESCONHECIDO" if hw_pred == 0 else "IDENTIFICADO"
    lines = [
        "=" * 60,
        "  RELATÓRIO DE COMPARAÇÃO SW vs. HW — CNN 19 Classes",
        "=" * 60,
        f"  Sem threshold — Desconhecido = classe 0 (nativo da rede)",
        f"  Predição SW : classe {sw_pred} = '{sw_name}' [{sw_status}]",
        f"  Predição HW : classe {hw_pred} = '{hw_name}' [{hw_status}]",
        f"  Concordância: {'✓ SIM' if agreement else '✗ NÃO'}",
        "-" * 60,
        f"  {'Camada':<10} {'N':>7} {'MAE':>8} {'RMSE':>8} {'MaxErr':>8} {'Exato%':>8} {'Pearson':>9}",
        "-" * 60,
    ]
    for m in metrics_list:
        pearson_str = f"{m['pearson']:.4f}" if m["pearson"] is not None else "  N/A"
        lines.append(
            f"  {m['label']:<10} {m['n']:>7} {m['mae']:>8.2f} {m['rmse']:>8.2f} "
            f"{m['max_err']:>8.0f} {m['exact%']:>7.1f}% {pearson_str:>9}"
        )
    lines += ["=" * 60, ""]

    report = "\n".join(lines)
    print("\n" + report)

    with open(out_path, "w") as f:
        f.write(report)
    print(f"[OK] Relatório salvo: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compara ativações SW (Keras) vs. HW (Verilog) por camada"
    )
    parser.add_argument("--image", default=None,
                        help="Caminho para a imagem, para inferir diretórios automaticamente")
    parser.add_argument("--sw-dir",    default=None,
                        help="Diretório com arquivos *_sw.txt")
    parser.add_argument("--hw-dir",    default=None,
                        help="Diretório com arquivos do testbench Verilog")
    parser.add_argument("--outdir",    default=None,
                        help="Diretório de saída para gráficos e relatório")
    args = parser.parse_args()

    if args.image:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        if not args.sw_dir:
            args.sw_dir = os.path.join("comparacao", "sw", base_name)
        if not args.hw_dir:
            args.hw_dir = os.path.join("comparacao", "hw", base_name)
        if not args.outdir:
            args.outdir = os.path.join("comparacao", "plots", base_name)
    else:
        if not args.sw_dir: args.sw_dir = "comparacao/sw/"
        if not args.hw_dir: args.hw_dir = "comparacao/hw/"
        if not args.outdir: args.outdir = "comparacao/plots/"

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("  CNN SW vs. HW Comparator")
    print("=" * 60)
    print(f"  SW dir    : {args.sw_dir}")
    print(f"  HW dir    : {args.hw_dir}")
    print(f"  Output    : {args.outdir}")

    print("=" * 60)

    # -----------------------------------------------------------------------
    # Carrega todas as camadas
    # -----------------------------------------------------------------------
    layers_config = [
        ("conv",  "conv_out_sw.txt",   "conv_out.txt"),
        ("pool",  "pool_out_sw.txt",   "pool_out.txt"),
        ("flat",  "flat_out_sw.txt",   "flat_out.txt"),
    ]

    layers_data    = {}  # Para plots de hist/scatter
    metrics_list   = []

    print("\n[INFO] Carregando dados por camada...")
    for label, sw_name, hw_name in layers_config:
        sw, hw = load_layer(args.sw_dir, args.hw_dir, sw_name, hw_name, label)
        if sw is not None:
            layers_data[label] = (sw, hw)
            metrics_list.append(compute_metrics(sw, hw, label))

    # Camada densa (tratada separadamente para o barplot)
    sw_dense, hw_dense = load_layer(
        args.sw_dir, args.hw_dir, "dense_out_sw.txt", "dense_out.txt", "dense"
    )
    if sw_dense is not None:
        metrics_list.append(compute_metrics(sw_dense, hw_dense, "dense"))

    if not layers_data and sw_dense is None:
        print("[ERRO] Nenhum dado encontrado. Verifique os diretórios.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Gráficos
    # -----------------------------------------------------------------------
    np.random.seed(42)

    if layers_data:
        plot_histograms(layers_data,
                        os.path.join(args.outdir, "01_hist_errors.png"))
        plot_scatters(layers_data,
                      os.path.join(args.outdir, "02_scatter_layers.png"))

    if sw_dense is not None and hw_dense is not None:
        plot_scores(sw_dense, hw_dense,
                    os.path.join(args.outdir, "03_barplot_scores.png"))

    if "conv" in layers_data:
        sw_c, hw_c = layers_data["conv"]
        plot_conv_heatmap(sw_c, hw_c,
                          os.path.join(args.outdir, "04_heatmap_conv.png"))

    # -----------------------------------------------------------------------
    # Relatório textual
    # -----------------------------------------------------------------------
    sw_pred = int(np.argmax(sw_dense)) if sw_dense is not None else -1
    hw_pred = int(np.argmax(hw_dense)) if hw_dense is not None else -1
    agreement = (sw_pred == hw_pred)

    save_report(metrics_list,
                os.path.join(args.outdir, "report.txt"),
                sw_pred, hw_pred, agreement)

    print("\n[OK] Comparação concluída.")
    print(f"     Gráficos e relatório em: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
