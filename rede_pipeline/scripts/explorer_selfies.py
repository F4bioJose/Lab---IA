import sys
import tarfile
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random
from src.config import Config

cfg = Config()


def extrair_ucf_selfies(limite=7000):
    tar_path = cfg.RAW_DIR / "Selfie-dataset.tar.gz"
    destino_raw = cfg.RAW_DIR / "selfies"

    if not tar_path.exists():
        print(f"ERRO: Arquivo não encontrado em {tar_path}")
        print("Certifique-se de que o arquivo se chama exatamente 'Selfie-dataset.tar.gz'")
        return []

    destino_raw.mkdir(parents=True, exist_ok=True)
    print(f"Lendo o arquivo {tar_path.name} e extraindo para {destino_raw.name}...")

    salvas = 0
    imagens_extraidas = []

    with tarfile.open(tar_path, "r:gz") as tar:
        for membro in tar:
            if membro.isfile() and membro.name.lower().endswith('.jpg'):
                nome_arquivo = Path(membro.name).name
                caminho_salvar = destino_raw / nome_arquivo
                if not caminho_salvar.exists():
                    f = tar.extractfile(membro)
                    if f is not None:
                        with open(caminho_salvar, "wb") as out_file:
                            out_file.write(f.read())

                imagens_extraidas.append(caminho_salvar)
                salvas += 1

                if salvas % 1000 == 0:
                    print(f"  -> {salvas} selfies extraídas...")

                if salvas >= limite:
                    break

    print(f"\nSucesso! {salvas} selfies prontas.")
    return imagens_extraidas


def visualizar_amostra(imagens_paths, num_amostras=5):
    """Gera um plot comparando a selfie real com a visão da FPGA."""
    print("\nGerando visualização (Original vs Pré-processada para a FPGA)...")
    plt.figure(figsize=(12, 5))

    amostras = random.sample(imagens_paths, min(num_amostras, len(imagens_paths)))

    for i, img_path in enumerate(amostras):
        # Lê a imagem original
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (cfg.IMG_SIZE, cfg.IMG_SIZE))

        plt.subplot(2, num_amostras, i + 1)
        plt.imshow(img_rgb)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, num_amostras, i + 1 + num_amostras)
        plt.imshow(resized, cmap='gray')
        plt.title(f"FPGA ({cfg.IMG_SIZE}x{cfg.IMG_SIZE})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    caminhos = extrair_ucf_selfies(limite=7000)
    if caminhos:
        visualizar_amostra(caminhos)