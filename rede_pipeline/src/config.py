import json
from pathlib import Path


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._init_paths()
        return cls._instance

    def _init_paths(self):
        # Definindo paths
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.INTERIM_DIR = self.DATA_DIR / "interim"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.REPORTS_DIR = self.PROJECT_ROOT / "reports"
        self.EXPORT_DIR = self.PROJECT_ROOT / "export"
        self.LFW_TAR = self.RAW_DIR / "lfw.tgz"
        self.LFW_RAW_DIR = self.RAW_DIR / "lfw_extracted"
        self.RAW_AUTORIZADO_DIR = self.RAW_DIR / "1_autorizado"
        self.INTERIM_AUTORIZADO_DIR = self.INTERIM_DIR / "1_autorizado"
        self.NEGADOS_INTERIM_DIR = self.INTERIM_DIR / "0_desconhecido"
        self.TUNER_LOGS_DIR = self.PROJECT_ROOT / "tuner_logs"

        self.IMG_SIZE = 32 # não alterar
        self.CHANNELS = 1
        self.CLASSIFICATION_MODE = "multiclass"
        self.UNKNOWN_CLASS_NAME = "0_desconhecido"
        self.BINARY_AUTHORIZED_CLASS_NAME = "1_autorizado"

        self.RUN_HYPERPARAMETER_SEARCH = True
        self.TUNER_OVERWRITE = True
        self.TUNER_MAX_TRIALS = 30
        self.SEARCH_EPOCHS = 45
        self.SEARCH_EARLY_STOP_PATIENCE = 10

        self.TRAIN_EPOCHS = 200
        self.EARLY_STOP_PATIENCE = 25

        # Hiperparametros padrao do modelo
        self.DEFAULT_DROPOUT = 0.1
        self.DEFAULT_OPTIMIZER = "adam"
        self.DEFAULT_LEARNING_RATE = 0.002
        self.DEFAULT_PESO_CLASSE_0 = 1.0
        # Proporção classe desconhecida
        self.UNKNOWN_RATIO_BINARY = 1.0
        self.UNKNOWN_RATIO_MULTICLASS = 0.2


        # Meta automatica para balanceamento das classes autorizadas.
        # Estratégia max para obter o máximo de exemplos e multiplier alto.
        self.AUTHORIZED_TARGET_STRATEGY = "max"
        # Limita a quantidade de imagens artificiais por classe para evitar dominação por augmentations
        self.MAX_AUG_MULTIPLIER = 3.0  # Permite até 3x de augmentation para atingir os números de amostras do alvo
        self.AUTHORIZED_TARGET_QUANTILE = 0.25
        self.AUTHORIZED_TARGET_MIN = 400
        self.AUTHORIZED_TARGET_MAX = 1500
        self.AUTHORIZED_TARGET_FALLBACK = 500

        # Espacos de busca por modo
        self.BINARY_SEARCH_SPACE = {
            "dropout": [0.05, 0.1, 0.15, 0.2],
            "optimizer": ["adam", "rmsprop"],
            "learning_rate": [3e-4, 5e-4, 8e-4, 1e-3, 2e-3],
            "peso_classe_0": [1.0, 1.25, 1.5, 2.0],
            "max_trials": 16,
            "search_epochs": 50,
        }
        self.MULTICLASS_SEARCH_SPACE = {
            "dropout": [0.05, 0.1, 0.15],
            "optimizer": ["adam"],
            "learning_rate": [1e-3, 2e-3, 3e-3],
            "peso_classe_0": [1.0],
            "max_trials": 5,
            "search_epochs": 45,
        }

        # Restricoes de quantizacao para alinhamento com FPGA (Q1.7).
        self.QUANT_BITS = 8
        self.QUANT_FRAC_BITS = 7
        self.ENABLE_QAT_WEIGHT_SIMULATION = True
        # QAT começa cedo na busca para que a seleção de hiperparâmetros já enxergue o regime quantizado.
        self.QAT_START_EPOCH_SEARCH = 5
        # No treino final, dá um warmup um pouco maior para estabilizar a convergência antes da quantização.
        self.QAT_START_EPOCH_FINAL = 20
        # Alias de compatibilidade com o resto do código e checkpoints antigos.
        self.QAT_START_EPOCH = self.QAT_START_EPOCH_FINAL
        self.ENABLE_HARD_WEIGHT_CONSTRAINT_DURING_TRAIN = True  # ✅ QAT Ativado

        # Limite de armazenamento de pesos na SRAM externa (512 KB).
        self.MAX_MIF_BYTES = 512 * 1024

        self._refresh_mode_dependent_paths()

    @property
    def is_binary_mode(self):
        return self.CLASSIFICATION_MODE == "binary"

    @property
    def is_multiclass_mode(self):
        return self.CLASSIFICATION_MODE == "multiclass"

    def _refresh_mode_dependent_paths(self):
        mode_suffix = "binario" if self.is_binary_mode else "multiclasse"
        self.TUNER_PROJECT_NAME = f"tiny_cnn_{mode_suffix}_search"
        self.MODEL_FILENAME = f"tiny_cnn_{mode_suffix}.h5"
        self.MODEL_PATH = self.MODELS_DIR / self.MODEL_FILENAME
        self.CLASS_MAP_PATH = self.MODELS_DIR / f"class_indices_{mode_suffix}.json"
        self.BEST_HPS_PATH = self.MODELS_DIR / f"best_hyperparameters_{mode_suffix}.json"
        self.TRAINING_METRICS_PATH = self.REPORTS_DIR / f"training_metrics_{mode_suffix}.json"

        if self.is_binary_mode:
            active_space = self.BINARY_SEARCH_SPACE
            self.DEFAULT_PESO_CLASSE_0 = 1.0
            self.UNKNOWN_RATIO_ACTIVE = self.UNKNOWN_RATIO_BINARY
        else:
            active_space = self.MULTICLASS_SEARCH_SPACE
            self.DEFAULT_PESO_CLASSE_0 = 1.0
            self.UNKNOWN_RATIO_ACTIVE = self.UNKNOWN_RATIO_MULTICLASS

        self.SEARCH_DROPOUT_VALUES = active_space["dropout"]
        self.SEARCH_OPTIMIZERS = active_space["optimizer"]
        self.SEARCH_LR_VALUES = active_space["learning_rate"]
        self.SEARCH_PESO_C0_VALUES = active_space["peso_classe_0"]
        self.TUNER_MAX_TRIALS = active_space["max_trials"]
        self.SEARCH_EPOCHS = active_space["search_epochs"]

    def validate(self):
        valid_modes = {"binary", "multiclass"}
        if self.CLASSIFICATION_MODE not in valid_modes:
            raise ValueError(
                f"CLASSIFICATION_MODE invalido: {self.CLASSIFICATION_MODE}. Use 'binary' ou 'multiclass'."
            )

    def set_classification_mode(self, mode):
        self.CLASSIFICATION_MODE = mode
        self.validate()
        self._refresh_mode_dependent_paths()

    def has_saved_best_hps(self):
        return self.BEST_HPS_PATH.exists()

    def should_run_hyperparameter_search(self):
        if self.RUN_HYPERPARAMETER_SEARCH is not None:
            return bool(self.RUN_HYPERPARAMETER_SEARCH)

        # Politica automatica:
        # - multiclasse: busca quando ainda nao existe melhor configuracao salva desse modo
        # - binario: sem busca por padrao
        if self.is_multiclass_mode:
            return not self.has_saved_best_hps()
        return False

    def load_saved_best_hps(self):
        if not self.has_saved_best_hps():
            return None
        try:
            with open(self.BEST_HPS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def setup_directories(self):
        dirs = [
            self.RAW_DIR, self.INTERIM_DIR, self.PROCESSED_DIR,
            self.RAW_AUTORIZADO_DIR, self.INTERIM_AUTORIZADO_DIR, self.NEGADOS_INTERIM_DIR,
            self.MODELS_DIR, self.REPORTS_DIR, self.EXPORT_DIR, self.TUNER_LOGS_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
