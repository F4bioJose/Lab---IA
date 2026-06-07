import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import Config
from src.model_io import load_tinycnn_model

cfg = Config()


def avaliar_modelo():
    print("Carregando o modelo final...")
    model_path = cfg.MODEL_PATH
    model = load_tinycnn_model(model_path, compile_model=False)

    # Prepara os dados de TESTE
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_gen = test_datagen.flow_from_directory(
        cfg.PROCESSED_DIR / "test",
        target_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=32,
        shuffle=False
    )

    # Predições
    print("Gerando predições para o conjunto de teste...")
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\n--- Relatório de Classificação ---")
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito ')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')

    img_path = cfg.PROJECT_ROOT / "reports" / "matriz_confusao.png"
    img_path.parent.mkdir(exist_ok=True)
    plt.savefig(str(img_path))
    print(f"\nGráfico da Matriz de Confusão salvo em: {img_path}")
    plt.show()


if __name__ == "__main__":
    avaliar_modelo()