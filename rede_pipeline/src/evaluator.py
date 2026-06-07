import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model_io import load_tinycnn_model

class ModelEvaluator:
    def __init__(self, config):
        self.cfg = config

    def plot_training_history(self, history):
        if isinstance(history, tuple):
            history = history[0]

        stats = history.history if hasattr(history, 'history') else history

        loss = stats.get('loss')
        val_loss = stats.get('val_loss')

        train_acc = stats.get('accuracy')
        val_acc = stats.get('val_accuracy')
        val_macro_f1 = stats.get('val_macro_f1')

        if loss is None:
            print("Erro: Chaves não encontradas no histórico.")
            return

        epochs_range = range(len(loss))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, label='Treino')
        plt.plot(epochs_range, val_loss, label='Validação')
        plt.title('Loss')
        plt.legend()

        if train_acc is not None:
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, train_acc, label='Treino')
            plt.plot(epochs_range, val_acc, label='Validação')
            if val_macro_f1 is not None:
                plt.plot(epochs_range, val_macro_f1, label='Val Macro-F1')
            plt.title('Acuracia / Macro-F1')
            plt.legend()

        plt.tight_layout()

        img_path = self.cfg.PROJECT_ROOT / "reports" / "historico_treinamento.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(img_path))
        print(f"\nGráficos de treinamento salvos em: {img_path}")
        plt.close()

    def evaluate_on_test_set(self):
        print("\nCarregando o modelo final para avaliação...")
        model = load_tinycnn_model(self.cfg.MODEL_PATH, compile_model=False)

        test_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            self.cfg.PROCESSED_DIR / "test",
            target_size=(self.cfg.IMG_SIZE, self.cfg.IMG_SIZE),
            color_mode="grayscale",
            class_mode="sparse",
            batch_size=32,
            shuffle=False
        )

        print("\nGerando predições para o conjunto de teste...")
        y_pred_prob = model.predict(test_gen, verbose=0)
        y_true = test_gen.classes
        class_names = list(test_gen.class_indices.keys())
        y_pred = np.argmax(y_pred_prob, axis=1)

        if self.cfg.is_binary_mode:
            test_auc_ovr = roc_auc_score(y_true, y_pred_prob[:, 1])
        else:
            test_auc_ovr = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')
        test_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        test_balanced_acc = balanced_accuracy_score(y_true, y_pred)

        unknown_idx = test_gen.class_indices.get(self.cfg.UNKNOWN_CLASS_NAME, 0)
        unk_true = (y_true == unknown_idx).astype(int)
        unk_pred = (y_pred == unknown_idx).astype(int)
        unknown_recall = recall_score(unk_true, unk_pred, zero_division=0)

        auth_labels = [i for i in np.unique(y_true) if i != unknown_idx]
        authorized_recall = recall_score(
            y_true,
            y_pred,
            labels=auth_labels,
            average='macro',
            zero_division=0
        ) if auth_labels else 0.0

        modo = "Binario" if self.cfg.is_binary_mode else "Multiclasse"
        print(f"\n--- Metricas de Teste ({modo}) ---")
        if self.cfg.is_binary_mode:
            print(f"AUC: {test_auc_ovr:.4f}")
        else:
            print(f"AUC OvR (macro): {test_auc_ovr:.4f}")
        print(f"Macro-F1: {test_macro_f1:.4f}")
        print(f"Balanced Accuracy: {test_balanced_acc:.4f}")
        print(f"Recall da classe desconhecido (negar invasor): {unknown_recall:.4f}")
        print(f"Recall macro dos alunos autorizados: {authorized_recall:.4f}")

        print("\n--- Relatório de Classificação ---")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusao - {modo}')

        img_path_cm = self.cfg.REPORTS_DIR / "matriz_confusao.png"
        plt.savefig(str(img_path_cm))
        plt.close()
        print(f"Matriz de confusão salva em: {img_path_cm}")

        metrics_payload = {
            "auc_ovr_macro": float(test_auc_ovr),
            "macro_f1": float(test_macro_f1),
            "balanced_accuracy": float(test_balanced_acc),
            "unknown_recall": float(unknown_recall),
            "authorized_recall_macro": float(authorized_recall),
        }

        self.cfg.TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        print(f"Metricas consolidadas salvas em: {self.cfg.TRAINING_METRICS_PATH}")

    def validate_quantization_degradation(self):
        """
        Valida se a quantização Q1.7 preserva a qualidade do modelo.
        Compara predições FP32 vs Q1.7 no conjunto de teste.
        """
        print("\n" + "="*70)
        print("[VALIDAÇÃO] Quantização Q1.7 - Integridade de Pesos")
        print("="*70)
        
        # 1. Carregar modelo original (FP32)
        print("\n[1/5] Carregando modelo FP32...")
        model_fp32 = load_tinycnn_model(self.cfg.MODEL_PATH, compile_model=False)
        
        # 2. Carregar test set
        print("[2/5] Carregando test set...")
        test_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            self.cfg.PROCESSED_DIR / "test",
            target_size=(self.cfg.IMG_SIZE, self.cfg.IMG_SIZE),
            color_mode="grayscale",
            class_mode="sparse",
            batch_size=32,
            shuffle=False
        )
        
        # 3. Predições FP32
        print("[3/5] Executando predições FP32...")
        predictions_fp32 = model_fp32.predict(test_gen, verbose=0)
        
        # 4. Simular quantização Q1.7 nos pesos
        print("[4/5] Simulando quantização Q1.7 nos pesos...")
        frac_bits = self.cfg.QUANT_FRAC_BITS
        scale = float(2 ** frac_bits)
        min_val = -1.0
        max_val = (2 ** frac_bits - 1) / (2 ** frac_bits)
        
        # Criar cópia do modelo para quantizar
        model_q17 = load_tinycnn_model(self.cfg.MODEL_PATH, compile_model=False)
        
        # Quantizar todos os pesos
        for layer in model_q17.layers:
            for weight in layer.weights:
                # Simular quantização Q1.7
                clipped = tf.clip_by_value(weight, min_val, max_val)
                quantized = tf.round(clipped * scale) / scale
                weight.assign(quantized)
        
        # 5. Predições Q1.7
        print("[5/5] Executando predições Q1.7...")
        predictions_q17 = model_q17.predict(test_gen, verbose=0)
        
        # ============ ANÁLISE DETALHADA ============
        y_true = test_gen.classes
        class_names = list(test_gen.class_indices.keys())
        
        y_pred_fp32 = np.argmax(predictions_fp32, axis=1)
        y_pred_q17 = np.argmax(predictions_q17, axis=1)
        
        # Métrica 1: Diferença média de predições
        pred_diff = np.abs(predictions_fp32 - predictions_q17)
        mean_diff = np.mean(pred_diff)
        max_diff = np.max(pred_diff)
        
        # Métrica 2: Taxa de mudança de predição
        changed_predictions = (y_pred_fp32 != y_pred_q17).sum()
        changed_rate = changed_predictions / len(y_true)
        
        # Métrica 3: Acurácia antes e depois
        acc_fp32 = np.mean(y_pred_fp32 == y_true)
        acc_q17 = np.mean(y_pred_q17 == y_true)
        acc_degradation = acc_fp32 - acc_q17
        
        # Métrica 4: Métrica F1 antes e depois
        f1_fp32 = f1_score(y_true, y_pred_fp32, average='macro', zero_division=0)
        f1_q17 = f1_score(y_true, y_pred_q17, average='macro', zero_division=0)
        f1_degradation = f1_fp32 - f1_q17
        
        # Métrica 5: AUC antes e depois
        if self.cfg.is_binary_mode:
            auc_fp32 = roc_auc_score(y_true, predictions_fp32[:, 1])
            auc_q17 = roc_auc_score(y_true, predictions_q17[:, 1])
        else:
            auc_fp32 = roc_auc_score(y_true, predictions_fp32, average='macro', multi_class='ovr')
            auc_q17 = roc_auc_score(y_true, predictions_q17, average='macro', multi_class='ovr')
        auc_degradation = auc_fp32 - auc_q17
        
        # ============ RELATÓRIO ============
        print("\n" + "-"*70)
        print("📊 RELATÓRIO DE VALIDAÇÃO DE QUANTIZAÇÃO Q1.7")
        print("-"*70)
        
        print(f"\n🔹 DIFERENÇAS DE PREDIÇÃO:")
        print(f"   • Diferença média (L1):     {mean_diff:.6f}")
        print(f"   • Diferença máxima (L∞):    {max_diff:.6f}")
        print(f"   • Amostras com mudança:     {changed_predictions}/{len(y_true)} ({changed_rate*100:.2f}%)")
        
        print(f"\n🔹 ACURÁCIA:")
        print(f"   • FP32:                     {acc_fp32:.4f}")
        print(f"   • Q1.7:                     {acc_q17:.4f}")
        print(f"   • Degradação:               {acc_degradation:.4f} ({acc_degradation*100:.2f}%)")
        
        print(f"\n🔹 MACRO-F1:")
        print(f"   • FP32:                     {f1_fp32:.4f}")
        print(f"   • Q1.7:                     {f1_q17:.4f}")
        print(f"   • Degradação:               {f1_degradation:.4f} ({f1_degradation*100:.2f}%)")
        
        print(f"\n🔹 AUC OVR (MACRO):")
        print(f"   • FP32:                     {auc_fp32:.4f}")
        print(f"   • Q1.7:                     {auc_q17:.4f}")
        print(f"   • Degradação:               {auc_degradation:.4f} ({auc_degradation*100:.2f}%)")
        
        # ============ VALIDAÇÃO E ALERTAS ============
        print("\n" + "="*70)
        print("✅ CRITÉRIOS DE VALIDAÇÃO:")
        print("="*70)
        
        THRESHOLD_DIFF = 0.02  # 2%
        THRESHOLD_CHANGED = 0.05  # 5% de mudança
        THRESHOLD_DEGRADATION = 0.01  # 1% de degradação
        
        all_passed = True
        
        # Verificação 1: Diferença média
        check1 = mean_diff < THRESHOLD_DIFF
        status1 = "✅ PASSOU" if check1 else "❌ FALHOU"
        print(f"\n1. Diferença média < {THRESHOLD_DIFF:.2f}:")
        print(f"   {status1} (atual: {mean_diff:.6f})")
        all_passed = all_passed and check1
        
        # Verificação 2: Taxa de mudança
        check2 = changed_rate < THRESHOLD_CHANGED
        status2 = "✅ PASSOU" if check2 else "❌ FALHOU"
        print(f"\n2. Mudança de predição < {THRESHOLD_CHANGED*100:.1f}%:")
        print(f"   {status2} (atual: {changed_rate*100:.2f}%)")
        all_passed = all_passed and check2
        
        # Verificação 3: Degradação de acurácia
        check3 = acc_degradation < THRESHOLD_DEGRADATION
        status3 = "✅ PASSOU" if check3 else "❌ FALHOU"
        print(f"\n3. Degradação de acurácia < {THRESHOLD_DEGRADATION*100:.1f}%:")
        print(f"   {status3} (atual: {acc_degradation*100:.2f}%)")
        all_passed = all_passed and check3
        
        # Verificação 4: Degradação de F1
        check4 = f1_degradation < THRESHOLD_DEGRADATION
        status4 = "✅ PASSOU" if check4 else "❌ FALHOU"
        print(f"\n4. Degradação de Macro-F1 < {THRESHOLD_DEGRADATION*100:.1f}%:")
        print(f"   {status4} (atual: {f1_degradation*100:.2f}%)")
        all_passed = all_passed and check4
        
        # Verificação 5: Degradação de AUC
        check5 = auc_degradation < THRESHOLD_DEGRADATION
        status5 = "✅ PASSOU" if check5 else "❌ FALHOU"
        print(f"\n5. Degradação de AUC < {THRESHOLD_DEGRADATION*100:.1f}%:")
        print(f"   {status5} (atual: {auc_degradation*100:.2f}%)")
        all_passed = all_passed and check5
        
        # ============ CONCLUSÃO ============
        print("\n" + "="*70)
        if all_passed:
            print("🎉 RESULTADO FINAL: ✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
            print("\nOs pesos Q1.7 preservam a qualidade do modelo.")
            print("Seguro para exportação para FPGA!")
        else:
            print("⚠️  RESULTADO FINAL: ⚠️  ATENÇÃO NECESSÁRIA!")
            print("\nA degradação em quantização está acima dos limites.")
            print("Recomendações:")
            if not check1:
                print("  • Aumentar precisão de quantização (considerar mais bits)")
            if not check2:
                print("  • Revisar distribuição de pesos durante treinamento")
            if not check3 or not check4:
                print("  • Considerar Quantization-Aware Training (QAT) mais agressivo")
            if not check5:
                print("  • Validar calibração de quantização")
        
        print("="*70 + "\n")
        
        # ============ SALVAR RELATÓRIO ============
        validation_report = {
            "timestamp": str(np.datetime64('today')),
            "mode": "binario" if self.cfg.is_binary_mode else "multiclasse",
            "fp32_accuracy": float(acc_fp32),
            "q17_accuracy": float(acc_q17),
            "accuracy_degradation": float(acc_degradation),
            "fp32_f1": float(f1_fp32),
            "q17_f1": float(f1_q17),
            "f1_degradation": float(f1_degradation),
            "fp32_auc": float(auc_fp32),
            "q17_auc": float(auc_q17),
            "auc_degradation": float(auc_degradation),
            "mean_prediction_difference": float(mean_diff),
            "max_prediction_difference": float(max_diff),
            "changed_predictions": int(changed_predictions),
            "changed_predictions_rate": float(changed_rate),
            "validation_passed": bool(all_passed),
            "thresholds": {
                "max_mean_diff": float(THRESHOLD_DIFF),
                "max_changed_rate": float(THRESHOLD_CHANGED),
                "max_degradation": float(THRESHOLD_DEGRADATION)
            }
        }
        
        report_path = self.cfg.REPORTS_DIR / "quantization_validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"📄 Relatório salvo em: {report_path}\n")
        
        return validation_report

