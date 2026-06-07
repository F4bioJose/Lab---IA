import os
import random
import kagglehub
import shutil
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

import tensorflow as tf

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

from src.config import Config
from src.data_utils import DataExtractor
from src.preprocessor import ImageProcessor, DataPreprocessor
from src.dataset_manager import DatasetManager
from src.engine import ModelEngine
from src.evaluator import ModelEvaluator
from src.model import build_tiny_cnn
from src.export_mif import export_model_to_mif


def main():
    cfg = Config()
    cfg.validate()
    cfg.setup_directories()
    extractor = DataExtractor()
    img_processor = ImageProcessor(cfg.IMG_SIZE)
    data_preprocessor = DataPreprocessor(cfg, img_processor, extractor)
    ds_manager = DatasetManager(cfg)
    evaluator = ModelEvaluator(cfg)
    random.seed(42)

    print("\n" + "=" * 50)
    print("         INICIANDO PIPELINE BIOMÉTRICA (ARTEFATO 1)      ")
    print("=" * 50)

    # 1. GESTÃO DE DATASETS da Classe 0
    print("\n[PASSO 1] Gerenciando fontes de Desconhecidos...")

    # Extração das Selfies locais (.tar.gz)
    tar_selfies = cfg.RAW_DIR / "Selfie-dataset.tar.gz"
    if tar_selfies.exists():
        extractor.extract_tar(tar_selfies, cfg.RAW_DIR / "selfies", limit=3000)

    lfw_raw_folder = cfg.RAW_DIR / "lfw_extracted"
    if not lfw_raw_folder.exists():
        print("  -> Baixando e extraindo LFW (conexão necessária)...")
        lfw_download_path = kagglehub.dataset_download("atulanandjha/lfwpeople")
        shutil.copytree(lfw_download_path, lfw_raw_folder)

    # 2. PRÉ-PROCESSAMENTO (Raw -> Interim)
    data_preprocessor.clear_interim()
    data_preprocessor.process_authorized()
    data_preprocessor.process_unknowns(ratio=cfg.UNKNOWN_RATIO_ACTIVE)

    # 3. ORGANIZACAO DO DATASET (Interim -> Processed)
    print("\n[PASSO 3] Organizando Dataset (Split Treino/Validao/Teste)...")
    ds_manager.clean_processed()

    total_classes = 0
    desconhecidos = list(cfg.NEGADOS_INTERIM_DIR.glob("*.jpg"))
    if desconhecidos:
        ds_manager.split_data(desconhecidos, cfg.UNKNOWN_CLASS_NAME)
        total_classes += 1

    alunos_dirs = sorted([p for p in cfg.INTERIM_AUTORIZADO_DIR.iterdir() if p.is_dir()])
    if cfg.is_binary_mode:
        autorizados = []
        for aluno_dir in alunos_dirs:
            autorizados.extend(list(aluno_dir.rglob("*.jpg")))
        if autorizados:
            ds_manager.split_data(autorizados, cfg.BINARY_AUTHORIZED_CLASS_NAME)
            total_classes += 1
        print(f" -> Dataset binario pronto com {total_classes} classes.")
    else:
        for idx, aluno_dir in enumerate(alunos_dirs, start=1):
            class_name = f"{idx}_{aluno_dir.name}"
            aluno_imgs = list(aluno_dir.rglob("*.jpg"))
            if aluno_imgs:
                ds_manager.split_data(aluno_imgs, class_name)
                total_classes += 1
        print(f" -> Dataset multiclasse pronto com {total_classes} classes.")

    # 4. TREINAMENTO E TUNING
    modo = "Binario" if cfg.is_binary_mode else "Multiclasse"
    print(f"\n[PASSO 4] Iniciando Treinamento Tiny-CNN ({modo})...")
    engine = ModelEngine(cfg, build_tiny_cnn)
    history, model = engine.train()

    # 5. AVALIAÇÃO E EXPORTAÇÃO
    print("\n[PASSO 5] Gerando relatórios e arquivos para FPGA...")
    evaluator.plot_training_history(history)
    evaluator.evaluate_on_test_set()

    # 5.1 VALIDAÇÃO DE QUANTIZAÇÃO (NÍVEL 1)
    print("\n[PASSO 5.1] Validando integridade de quantização Q1.7...")
    evaluator.validate_quantization_degradation()

    # Exportação para arquivos .mif quantizados em Q1.7
    export_model_to_mif()

    print("\n" + "=" * 50)
    print("      PIPELINE CONCLUÍDA COM SUCESSO (ARTEFATO 1)      ")
    print("=" * 50)


if __name__ == "__main__":
    main()