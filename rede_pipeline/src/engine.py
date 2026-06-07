import mlflow
import mlflow.keras
import keras_tuner as kt
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score
from src.augmentation import create_augmentation_pipeline
from src.quantization import Q17WeightQuantizationCallback

class TinyCNNHyperModel(kt.HyperModel):
    def __init__(self, model_builder, base_class_weight, num_classes, cfg, unknown_class_idx):
        super().__init__()
        self.model_builder = model_builder
        self.base_class_weight = base_class_weight
        self.num_classes = num_classes
        self.cfg = cfg
        self.unknown_class_idx = unknown_class_idx

    def build(self, hp):
        return self.model_builder(hp, self.num_classes, self.cfg)

    def fit(self, hp, model, *args, **kwargs):
        pena_invasor = hp.Choice('peso_classe_0', values=self.cfg.SEARCH_PESO_C0_VALUES)
        cw_tunado = dict(self.base_class_weight)
        cw_tunado[self.unknown_class_idx] = cw_tunado[self.unknown_class_idx] * pena_invasor
        kwargs['class_weight'] = cw_tunado
        return model.fit(*args, **kwargs)


class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """Calcula metricas de validacao focadas no desafio da fechadura biometrica."""

    def __init__(self, val_gen, unknown_class_idx):
        super().__init__()
        self.val_gen = val_gen
        self.unknown_class_idx = unknown_class_idx

    def __deepcopy__(self, memo):
        # Keras Tuner exige callbacks deep-copyable entre trials.
        # Mantemos a mesma referencia do generator para evitar tentar serializar locks internos.
        return ValidationMetricsCallback(self.val_gen, self.unknown_class_idx)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        probs = self.model.predict(self.val_gen, verbose=0)
        y_true = self.val_gen.classes
        y_pred = np.argmax(probs, axis=1)

        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        unk_true = (y_true == self.unknown_class_idx).astype(int)
        unk_pred = (y_pred == self.unknown_class_idx).astype(int)
        unknown_recall = recall_score(unk_true, unk_pred, zero_division=0)

        auth_labels = [i for i in np.unique(y_true) if i != self.unknown_class_idx]
        if auth_labels:
            authorized_recall = recall_score(
                y_true,
                y_pred,
                labels=auth_labels,
                average='macro',
                zero_division=0
            )
        else:
            authorized_recall = 0.0

        logs['val_macro_f1'] = float(macro_f1)
        logs['val_balanced_acc'] = float(bal_acc)
        logs['val_unknown_recall'] = float(unknown_recall)
        logs['val_authorized_recall'] = float(authorized_recall)



class ModelEngine:
    def __init__(self, config, model_builder):
        self.cfg = config
        self.model_builder = model_builder
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mode_name = "Binario" if self.cfg.is_binary_mode else "Multiclasse"
        mlflow.set_experiment(f"Fechadura_Biometrica_{mode_name}_Q17")

    @staticmethod
    def _to_int_key_dict(d):
        return {int(k): float(v) for k, v in d.items()}

    def _save_class_map(self, class_indices):
        self.cfg.CLASS_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.CLASS_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(class_indices, f, indent=2, ensure_ascii=True)

    def _save_best_hps(self, best_hps):
        values = dict(best_hps.values)
        self.cfg.BEST_HPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.BEST_HPS_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2, ensure_ascii=True)

    def _build_fixed_hps(self):
        hps = kt.HyperParameters()
        saved_hps = self.cfg.load_saved_best_hps()

        if saved_hps:
            hps.Fixed('dropout', float(saved_hps.get('dropout', self.cfg.DEFAULT_DROPOUT)))
            hps.Fixed('learning_rate', float(saved_hps.get('learning_rate', self.cfg.DEFAULT_LEARNING_RATE)))
            hps.Fixed('optimizer', str(saved_hps.get('optimizer', self.cfg.DEFAULT_OPTIMIZER)))
            hps.Fixed('peso_classe_0', float(saved_hps.get('peso_classe_0', self.cfg.DEFAULT_PESO_CLASSE_0)))
        else:
            hps.Fixed('dropout', self.cfg.DEFAULT_DROPOUT)
            hps.Fixed('learning_rate', self.cfg.DEFAULT_LEARNING_RATE)
            hps.Fixed('optimizer', self.cfg.DEFAULT_OPTIMIZER)
            hps.Fixed('peso_classe_0', self.cfg.DEFAULT_PESO_CLASSE_0)
        return hps

    @staticmethod
    def _build_class_weight(base_cw, unknown_idx, peso_c0):
        cw_final = dict(base_cw)
        cw_final[unknown_idx] = cw_final[unknown_idx] * peso_c0
        return cw_final

    def _build_callbacks(self, val_gen, is_search=False):
        unknown_idx = val_gen.class_indices.get(self.cfg.UNKNOWN_CLASS_NAME, 0)
        patience = self.cfg.SEARCH_EARLY_STOP_PATIENCE if is_search else self.cfg.EARLY_STOP_PATIENCE
        qat_start_epoch = self.cfg.QAT_START_EPOCH_SEARCH if is_search else self.cfg.QAT_START_EPOCH_FINAL
        qat_frequency = 3 if is_search else 1
        callbacks = [
            ValidationMetricsCallback(val_gen, unknown_idx),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_macro_f1',
                mode='max',
                patience=patience,
                restore_best_weights=True
            )
        ]

        if self.cfg.ENABLE_QAT_WEIGHT_SIMULATION:
            callbacks.append(
                Q17WeightQuantizationCallback(
                    frac_bits=self.cfg.QUANT_FRAC_BITS,
                    start_epoch=qat_start_epoch,
                    frequency=qat_frequency,
                )
            )

        return callbacks

    def get_generators(self, progressive_augmentation=False):
        """Prepara geradores em grayscale 32x32 para modo binario ou multiclasse."""
        if progressive_augmentation:
            train_gen = create_augmentation_pipeline(intensity=0.3)
        else:
            train_gen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=5,
                brightness_range=[0.8, 1.2],
                horizontal_flip=False
            )

        # A validação deve permanecer limpa para refletir a performance real do modelo.
        val_gen = ImageDataGenerator(rescale=1. / 255)

        train = train_gen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'train',
            target_size=(self.cfg.IMG_SIZE, self.cfg.IMG_SIZE),
            color_mode="grayscale",
            class_mode="sparse",
            batch_size=32,
            shuffle=True
        )

        val = val_gen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'val',
            target_size=(self.cfg.IMG_SIZE, self.cfg.IMG_SIZE),
            color_mode="grayscale",
            class_mode="sparse",
            batch_size=32,
            shuffle=False
        )
        return train, val

    def train(self):
        from sklearn.utils import class_weight
        run_search = self.cfg.should_run_hyperparameter_search()
        train_gen, val_gen = self.get_generators(progressive_augmentation=run_search)

        if train_gen.num_classes < 2:
            raise RuntimeError("Dataset invalido: e necessario ao menos 2 classes para treino.")

        if self.cfg.is_binary_mode and train_gen.num_classes != 2:
            raise RuntimeError(
                f"Modo binario requer 2 classes, mas o dataset possui {train_gen.num_classes}."
            )

        if self.cfg.is_multiclass_mode and train_gen.num_classes <= 2:
            raise RuntimeError(
                f"Modo multiclasse requer mais de 2 classes, mas o dataset possui {train_gen.num_classes}."
            )

        self._save_class_map(train_gen.class_indices)

        unknown_idx = train_gen.class_indices.get(self.cfg.UNKNOWN_CLASS_NAME)
        if unknown_idx is None:
            raise RuntimeError(f"Classe obrigatoria '{self.cfg.UNKNOWN_CLASS_NAME}' nao encontrada no dataset processado.")

        labels = train_gen.classes
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        base_cw = dict(enumerate(weights))
        base_cw = self._to_int_key_dict(base_cw)

        callbacks = self._build_callbacks(val_gen, is_search=run_search)
        mlflow.keras.autolog(log_models=True)


        if run_search:
            print("\n[HPS] Configuracao ativa da busca:")
            print(f"  - dropout: {self.cfg.SEARCH_DROPOUT_VALUES}")
            print(f"  - optimizer: {self.cfg.SEARCH_OPTIMIZERS}")
            print(f"  - learning_rate: {self.cfg.SEARCH_LR_VALUES}")
            print(f"  - peso_classe_0: {self.cfg.SEARCH_PESO_C0_VALUES}")
            print(f"  - max_trials: {self.cfg.TUNER_MAX_TRIALS}")
            print(f"  - search_epochs: {self.cfg.SEARCH_EPOCHS}")
            print(f"  - overwrite: {getattr(self.cfg, 'TUNER_OVERWRITE', False)}")

            hypermodel_wrapper = TinyCNNHyperModel(
                self.model_builder,
                base_cw,
                train_gen.num_classes,
                self.cfg,
                unknown_idx
            )

            tuner = kt.RandomSearch(
                hypermodel_wrapper,
                objective=kt.Objective('val_macro_f1', direction='max'),
                max_trials=self.cfg.TUNER_MAX_TRIALS,
                directory=str(self.cfg.TUNER_LOGS_DIR),
                project_name=self.cfg.TUNER_PROJECT_NAME,
                overwrite=bool(getattr(self.cfg, 'TUNER_OVERWRITE', False))
            )

            mode_name = "Binario" if self.cfg.is_binary_mode else "Multiclasse"
            with mlflow.start_run(run_name=f"HPS_{mode_name}_MacroF1"):
                mode_name = "binario" if self.cfg.is_binary_mode else "multiclasse"
                print(f"\n[PASSO 4.1] Iniciando busca ({mode_name}, objetivo: val_macro_f1)...")
                tuner.search(
                    train_gen,
                    validation_data=val_gen,
                    epochs=self.cfg.SEARCH_EPOCHS,
                    callbacks=callbacks
                )

                best_hps = tuner.get_best_hyperparameters()[0]
                self._save_best_hps(best_hps)
                model = tuner.hypermodel.build(best_hps)

                peso_escolhido = best_hps.get('peso_classe_0')
                cw_final = self._build_class_weight(base_cw, unknown_idx, peso_escolhido)
                final_callbacks = self._build_callbacks(val_gen, is_search=False)

                print("\n[PASSO 4.2] Iniciando ajuste fino do modelo final usando a melhor arquitetura encontrada...")
                history = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=self.cfg.TRAIN_EPOCHS,
                    class_weight=cw_final,
                    callbacks=final_callbacks
                )

                model.save(str(self.cfg.MODEL_PATH))
                print(f" -> Modelo final guardado em: {self.cfg.MODEL_PATH}")

            return history, model
        else:
            if self.cfg.has_saved_best_hps():
                print("\n[PASSO 4.1] Busca desativada. Reutilizando melhores hiperparametros salvos para este modo...")
            else:
                print("\n[PASSO 4.1] Busca desativada. Treinando com hiperparametros finais padrao...")
            best_hps = self._build_fixed_hps()
            self._save_best_hps(best_hps)

            model = self.model_builder(best_hps, train_gen.num_classes, self.cfg)
            peso_escolhido = best_hps.get('peso_classe_0')
            cw_final = self._build_class_weight(base_cw, unknown_idx, peso_escolhido)
            final_callbacks = self._build_callbacks(val_gen, is_search=False)

            mode_name = "Binario" if self.cfg.is_binary_mode else "Multiclasse"
            with mlflow.start_run(run_name=f"Training_{mode_name}_Final"):
                mode_name = "binario" if self.cfg.is_binary_mode else "multiclasse"
                print(f"\n[PASSO 4.2] Iniciando treinamento do modelo final ({mode_name})...")
                history = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=self.cfg.TRAIN_EPOCHS,
                    class_weight=cw_final,
                    callbacks=final_callbacks
                )

                model.save(str(self.cfg.MODEL_PATH))
                print(f" -> Modelo final guardado em: {self.cfg.MODEL_PATH}")

            return history, model
