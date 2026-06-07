import cv2
import random
import unicodedata
import re
from src.config import Config

cfg = Config()


def sanitize_name(name):
    # Remove acentos transformando "á" em "a" + "´" e descartando o acento
    nfd_form = unicodedata.normalize('NFD', name)
    without_accents = ''.join([c for c in nfd_form if not unicodedata.combining(c)])

    # Substitui espaços por underlines
    without_spaces = without_accents.replace(' ', '_')

    # Remove qualquer outro caractere especial (mantém letras, números, underlines e pontos)
    clean_name = re.sub(r'[^a-zA-Z0-9_.]', '', without_spaces)
    return clean_name


def aplicar_augmentation(imagem):
    img_aug = imagem.copy()

    if random.random() > 0.5:
        img_aug = cv2.flip(img_aug, 1)

    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-10, 10)
    img_aug = cv2.convertScaleAbs(img_aug, alpha=alpha, beta=beta)

    return img_aug


def processar_dados_autorizados(frames_pular=1, max_fotos_por_pessoa=400):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # SANITIZAÇÃO DOS NOMES NA PASTA RAW ---
    itens_raw_iniciais = list(cfg.RAW_AUTORIZADO_DIR.iterdir())
    for item in itens_raw_iniciais:
        nome_limpo = sanitize_name(item.name)
        if nome_limpo != item.name:
            novo_caminho = item.with_name(nome_limpo)
            item.rename(novo_caminho)
            print(f"Aviso: Renomeado '{item.name}' para '{nome_limpo}' para evitar erros.")

    itens_raw = list(cfg.RAW_AUTORIZADO_DIR.iterdir())

    if not itens_raw:
        print(f"Nenhum arquivo ou pasta encontrado em:\n{cfg.RAW_AUTORIZADO_DIR}")
        return

    for item in itens_raw:
        rostos_salvos = 0
        imagens_extraidas = []

        if item.is_file() and item.suffix.lower() == '.mp4':
            nome_pessoa = item.stem
            tipo_processamento = "video"
        elif item.is_dir():
            nome_pessoa = item.name
            tipo_processamento = "pasta"
        else:
            continue

        output_dir = cfg.INTERIM_AUTORIZADO_DIR / nome_pessoa
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Extraindo dados de: {nome_pessoa} (Origem: {tipo_processamento})")

        if tipo_processamento == "video":
            cap = cv2.VideoCapture(str(item))
            if not cap.isOpened():
                print(f"ERRO: Não foi possível abrir o vídeo {item.name}.")
                continue

            frames_lidos = 0
            while True:
                ret, frame = cap.read()
                if not ret or rostos_salvos >= max_fotos_por_pessoa:
                    break

                frames_lidos += 1
                if frames_lidos % frames_pular != 0:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

                for (x, y, w, h) in faces:
                    pad = int(w * 0.15)
                    y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
                    x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

                    rosto_crop = gray[y1:y2, x1:x2]
                    rosto_final = cv2.resize(rosto_crop, (cfg.IMG_SIZE, cfg.IMG_SIZE))

                    caminho_salvar = output_dir / f"{nome_pessoa}_original_{rostos_salvos:04d}.jpg"
                    cv2.imwrite(str(caminho_salvar), rosto_final)

                    imagens_extraidas.append(rosto_final)
                    rostos_salvos += 1
                    break

            cap.release()

        elif tipo_processamento == "pasta":
            fotos_raw = [p for p in item.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            random.shuffle(fotos_raw)

            for foto_path in fotos_raw:
                if rostos_salvos >= max_fotos_por_pessoa:
                    break

                img = cv2.imread(str(foto_path))
                if img is None: continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

                for (x, y, w, h) in faces:
                    pad = int(w * 0.15)
                    y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
                    x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

                    rosto_crop = gray[y1:y2, x1:x2]
                    rosto_final = cv2.resize(rosto_crop, (cfg.IMG_SIZE, cfg.IMG_SIZE))

                    caminho_salvar = output_dir / f"{nome_pessoa}_original_{rostos_salvos:04d}.jpg"
                    cv2.imwrite(str(caminho_salvar), rosto_final)
                    imagens_extraidas.append(rosto_final)
                    rostos_salvos += 1
                    break

        print(f"    Extraídos {rostos_salvos} rostos originais.")

        if rostos_salvos < max_fotos_por_pessoa and len(imagens_extraidas) > 0:
            faltam = max_fotos_por_pessoa - rostos_salvos
            print(f"    Faltam {faltam} fotos para a meta. Completando de forma sintética...")

            for i in range(faltam):
                img_base = random.choice(imagens_extraidas)
                img_aug = aplicar_augmentation(img_base)

                caminho_aug = output_dir / f"{nome_pessoa}_aug_{i:04d}.jpg"
                cv2.imwrite(str(caminho_aug), img_aug)
                rostos_salvos += 1

        print(f"    Concluído! Total de {rostos_salvos} fotos finalizadas na pasta {output_dir.name}")


if __name__ == "__main__":
    processar_dados_autorizados(frames_pular=1, max_fotos_por_pessoa=400)
    print("\nProcessamento finalizado com sucesso!")