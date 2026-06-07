import cv2
from src.config import Config
from src.data_utils import DataExtractor

def gravar_membro_equipe():
    cfg = Config()
    extractor = DataExtractor()

    nome_input = input("Digite o nome do integrante da equipe: ")
    nome_sanitizado = extractor.sanitize_name(nome_input)

    output_path = cfg.RAW_AUTORIZADO_DIR / f"{nome_sanitizado}.mp4"
    cfg.RAW_AUTORIZADO_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (640, 480))

    print(f"\nGravando para: {output_path}")
    print("DICAS: Mova a cabeça levemente para os lados, para cima e para baixo.")
    print("Pressione 'q' para PARAR a gravação.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        out.write(frame)

        cv2.imshow('Gravando Dados de Treino', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVídeo salvo com sucesso! Pronto para ser processado pela Main.")

if __name__ == "__main__":
    gravar_membro_equipe()