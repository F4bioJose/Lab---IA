import cv2
import numpy as np
import random
from src.config import Config

cfg = Config()


def contar_autorizados():
    """Conta quantas fotos a equipe possui no total para basear o equilíbrio."""
    if not cfg.INTERIM_AUTORIZADO_DIR.exists():
        return 0
    return len(list(cfg.INTERIM_AUTORIZADO_DIR.rglob("*.jpg")))


def gerar_paredes_sinteticas(interim_dir, quantidade=300):
    """Gera imagens de ruído/parede para a rede aprender a classe 'vazia'."""
    print(f"Gerando {quantidade} imagens de fundo (paredes/ruído)...")
    for i in range(quantidade):
        cor_base = random.randint(40, 230)
        imagem = np.full((cfg.IMG_SIZE, cfg.IMG_SIZE), cor_base, dtype=np.float32)

        tipo_gradiente = random.choice(['horizontal', 'vertical', 'nenhum'])
        if tipo_gradiente != 'nenhum':
            intensidade_luz = random.uniform(-40, 40)
            gradiente = np.linspace(0, intensidade_luz, cfg.IMG_SIZE)
            if tipo_gradiente == 'horizontal':
                imagem = imagem + gradiente
            else:
                imagem = imagem + gradiente[:, np.newaxis]

        ruido = np.random.normal(0, random.uniform(2.0, 15.0), (cfg.IMG_SIZE, cfg.IMG_SIZE))
        imagem = np.clip(imagem + ruido, 0, 255).astype(np.uint8)

        cv2.imwrite(str(interim_dir / f"fundo_sintetico_{i:04d}.jpg"), imagem)


def preprocess_unknown(proporcao_classe_0=2.0, num_fundos=300):
    # 1. Configuração de Alvos e Detectores
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    total_autorizados = contar_autorizados()

    if total_autorizados == 0:
        print("ERRO: Nenhuma imagem de Autorizados encontrada em 'interim/1_autorizado'.")
        return

    alvo_total_negados = int(total_autorizados * proporcao_classe_0)
    meta_rostos = alvo_total_negados - num_fundos

    print(f"\n--- Estratégia de Dataset Robusta (Zoom + Balanceamento) ---")
    print(f"Total Autorizados (Classe 1) : {total_autorizados}")
    print(f"Meta Desconhecidos (Classe 0) : {alvo_total_negados}")
    print(f"  -> Alvo de Rostos (Selfies) : {meta_rostos}")

    interim_dir = cfg.NEGADOS_INTERIM_DIR
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 2. Varredura das Selfies UCF (data/raw/selfies)
    src_dir = cfg.RAW_DIR / "selfies"
    image_paths = [p for p in src_dir.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    random.shuffle(image_paths)

    salvas = 0
    print(f"Processando selfies com detecção de face para consistência...")

    for img_path in image_paths:
        if salvas >= meta_rostos:
            break

        try:
            img_array = np.fromfile(str(img_path), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None: continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                # Aplica o mesmo enquadramento (zoom de 15% padding)
                pad = int(w * 0.15)
                y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

                rosto_crop = gray[y1:y2, x1:x2]
                resized = cv2.resize(rosto_crop, (cfg.IMG_SIZE, cfg.IMG_SIZE))

                dst_path = interim_dir / f"selfie_ucf_{salvas:05d}.jpg"
                cv2.imwrite(str(dst_path), resized)
                salvas += 1
                break

        except Exception:
            continue

    print(f"Rostos de estranhos processados: {salvas}")

    # 3. Gerar Fundos
    gerar_paredes_sinteticas(interim_dir, quantidade=num_fundos)

    print(f"\nProcessamento concluído! Total Classe 0: {salvas + num_fundos} imagens.")


if __name__ == "__main__":
    random.seed(42)
    preprocess_unknown(proporcao_classe_0=2.0, num_fundos=300)