import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from src.config import Config
from tabulate import tabulate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TrainingAnalyzer:
    def __init__(self):
        self.cfg = Config()
        self.project_root = self.cfg.PROJECT_ROOT
        self.reports_dir = self.cfg.REPORTS_DIR
        self.models_dir = self.cfg.MODELS_DIR
        self.tuner_logs_dir = self.cfg.TUNER_LOGS_DIR
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10

    def generate_hyperparameters_history(self):
        """Lê os resultados diretamente do Keras Tuner e gera o DataFrame atualizado."""
        tuner_dir = self.tuner_logs_dir / self.cfg.TUNER_PROJECT_NAME
        
        if not tuner_dir.exists():
            print(f"⚠️ Diretório de tuner não encontrado: {tuner_dir}")
            return pd.DataFrame()

        trials_data = []
        for trial_dir in tuner_dir.glob("trial_*"):
            trial_json = trial_dir / "trial.json"
            if trial_json.exists():
                with open(trial_json, "r") as f:
                    data = json.load(f)

                hiperparametros = data.get("hyperparameters", {}).get("values", {})
                metrics = data.get("metrics", {}).get("metrics", {})
                
                try:
                    val_macro_f1 = metrics.get("val_macro_f1", {}).get("observations", [{}])[0].get("value", [None])[0]
                    val_balanced_acc = metrics.get("val_balanced_acc", {}).get("observations", [{}])[0].get("value", [None])[0]
                    val_unknown_recall = metrics.get("val_unknown_recall", {}).get("observations", [{}])[0].get("value", [None])[0]
                    val_auc = metrics.get("val_auc", {}).get("observations", [{}])[0].get("value", [None])[0]
                    val_precision = metrics.get("val_precision", {}).get("observations", [{}])[0].get("value", [None])[0]
                    val_loss = metrics.get("val_loss", {}).get("observations", [{}])[0].get("value", [None])[0]
                except (IndexError, KeyError):
                    val_macro_f1 = None
                    val_balanced_acc = None
                    val_unknown_recall = None
                    val_auc = None
                    val_precision = None
                    val_loss = None

                status = data.get("status", "UNKNOWN")
                ranking_metric = val_macro_f1 if val_macro_f1 is not None else val_auc
                if status == "COMPLETED" and ranking_metric is not None:
                    record = {
                        "Trial ID": data.get("trial_id", ""),
                        "L2 Reg": hiperparametros.get("l2_reg"),
                        "Dropout": hiperparametros.get("dropout"),
                        "Otimizador": hiperparametros.get("optimizer"),
                        "Learn Rate": hiperparametros.get("learning_rate"),
                        "Peso C0": hiperparametros.get("peso_classe_0"),
                        "Val Macro-F1": val_macro_f1,
                        "Val Bal Acc": val_balanced_acc,
                        "Val Unknown Recall": val_unknown_recall,
                        "Val AUC": val_auc,
                        "Val Precision": val_precision,
                        "Val Loss": val_loss
                    }
                    trials_data.append(record)

        df = pd.DataFrame(trials_data)
        if not df.empty:
            if "Val Macro-F1" in df.columns and df["Val Macro-F1"].notna().any():
                df = df.sort_values(by="Val Macro-F1", ascending=False)
            else:
                if "Val AUC" in df.columns:
                    df = df.sort_values(by="Val AUC", ascending=False)
            df = df.reset_index(drop=True)
            
            # Exibir tabela assim como o analyzer.py fazia
            print("\n" + "="*80)
            print("      🏆 RANKING DAS ARQUITETURAS EXPLORADAS (KERAS TUNER)      ")
            print("="*80)
            print(tabulate(df.head(15), headers='keys', tablefmt='fancy_grid', showindex=False))
            
            # Salva num arquivo
            report_csv = self.reports_dir / "historico_hiperparametros.csv"
            df.to_csv(report_csv, index=False)
            
        return df

    def load_hyperparameters_history(self):
        """Carrega o histórico de hiperparâmetros da busca."""
        df = self.generate_hyperparameters_history()
        if df.empty:
            csv_path = self.reports_dir / "historico_hiperparametros.csv"
            if not csv_path.exists():
                print(f"❌ Histórico não encontrado e nenhum trial concluído em logs.")
                return None
            df = pd.read_csv(csv_path)

        # Remover linhas vazias
        df = df.dropna(how='all')
        return df
    
    def load_training_metrics(self):
        """Carrega as métricas finais de treinamento."""
        metrics_path = self.cfg.TRAINING_METRICS_PATH
        if not metrics_path.exists():
            print(f"❌ Arquivo não encontrado: {metrics_path}")
            return None
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def load_best_hyperparameters(self):
        """Carrega os melhores hiperparâmetros encontrados."""
        hp_path = self.cfg.BEST_HPS_PATH
        if not hp_path.exists():
            print(f"❌ Arquivo não encontrado: {hp_path}")
            return None
        
        with open(hp_path, 'r') as f:
            return json.load(f)
    
    def plot_hyperparameter_comparison(self, df):
        """Gera gráfico de comparação de hiperparâmetros."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Análise de Hiperparâmetros - Busca Multitrial', fontsize=16, fontweight='bold')
        
        target_metric = 'Val Macro-F1' if 'Val Macro-F1' in df.columns and not df['Val Macro-F1'].isna().all() else 'Val AUC'
        
        # 1. Val target por Dropout
        ax = axes[0, 0]
        dropout_auc = df.groupby('Dropout')[target_metric].agg(['mean', 'std', 'min', 'max'])
        dropout_auc.plot(kind='bar', ax=ax, color=['green', 'orange', 'red', 'blue'])
        ax.set_title(f'{target_metric} por Dropout', fontweight='bold')
        ax.set_ylabel(target_metric)
        ax.set_xlabel('Dropout')
        ax.grid(True, alpha=0.3)
        ax.legend(['Média', 'Std', 'Min', 'Max'])
        
        # 2. Val target por Learning Rate
        ax = axes[0, 1]
        lr_auc = df.groupby('Learn Rate')[target_metric].agg(['mean', 'std', 'min', 'max'])
        lr_auc.plot(kind='bar', ax=ax, color=['purple', 'cyan', 'magenta', 'yellow'])
        ax.set_title(f'{target_metric} por Learning Rate', fontweight='bold')
        ax.set_ylabel(target_metric)
        ax.set_xlabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.legend(['Média', 'Std', 'Min', 'Max'])
        
        # 3. Val target por Peso Classe 0
        ax = axes[1, 0]
        peso_auc = df.groupby('Peso C0')[target_metric].agg(['mean', 'std', 'min', 'max'])
        peso_auc.plot(kind='bar', ax=ax, color=['brown', 'pink', 'gray', 'olive'])
        ax.set_title(f'{target_metric} por Peso Classe 0', fontweight='bold')
        ax.set_ylabel(target_metric)
        ax.set_xlabel('Peso Classe 0')
        ax.grid(True, alpha=0.3)
        ax.legend(['Média', 'Std', 'Min', 'Max'])
        
        # 4. Val target vs Val Loss (Scatter)
        ax = axes[1, 1]
        scatter = ax.scatter(df['Val Loss'], df[target_metric], 
                            c=df['Dropout'].astype('category').cat.codes, s=100, cmap='viridis', alpha=0.6)
        ax.set_title(f'{target_metric} vs Val Loss', fontweight='bold')
        ax.set_xlabel('Val Loss')
        ax.set_ylabel(target_metric)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Dropout Codes')
        
        plt.tight_layout()
        output_path = self.reports_dir / "hyperparameter_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo: {output_path}")
        plt.close()
    
    def plot_trial_rankings(self, df):
        """Gera ranking de trials por métrica."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Ranking de Trials - Top 10', fontsize=16, fontweight='bold')
        
        target_metric = 'Val Macro-F1' if 'Val Macro-F1' in df.columns and not df['Val Macro-F1'].isna().all() else 'Val AUC'
        secondary_metric = 'Val Bal Acc' if 'Val Bal Acc' in df.columns and not df['Val Bal Acc'].isna().all() else 'Val Loss'
        
        # Top 10 por target
        ax = axes[0]
        top_auc = df.nlargest(10, target_metric)[['Trial ID', target_metric]].sort_values(target_metric)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_auc)))
        ax.barh(range(len(top_auc)), top_auc[target_metric].values, color=colors)
        ax.set_yticks(range(len(top_auc)))
        ax.set_yticklabels(top_auc['Trial ID'].values)
        ax.set_xlabel(target_metric)
        ax.set_title(f'Top 10 Trials por {target_metric}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Top 10 por secondary
        ax = axes[1]
        try:
            top_prec = df.nsmallest(10, secondary_metric) if secondary_metric == 'Val Loss' else df.nlargest(10, secondary_metric)
            top_prec = top_prec[['Trial ID', secondary_metric]].sort_values(secondary_metric)
            ax.barh(range(len(top_prec)), top_prec[secondary_metric].values, color=colors)
            ax.set_yticks(range(len(top_prec)))
            ax.set_yticklabels(top_prec['Trial ID'].values)
            ax.set_xlabel(secondary_metric)
            ax.set_title(f'Top 10 Trials por {secondary_metric}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        except Exception as e:
            ax.set_title(f"Erro ao plotar {secondary_metric}: {e}")
        
        plt.tight_layout()
        output_path = self.reports_dir / "trial_rankings.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo: {output_path}")
        plt.close()
    
    def plot_distribution_analysis(self, df):
        """Analisa distribuição de métricas."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribuição de Métricas', fontsize=16, fontweight='bold')
        
        target_metric = 'Val Macro-F1' if 'Val Macro-F1' in df.columns and not df['Val Macro-F1'].isna().all() else 'Val AUC'
        secondary_metric = 'Val Bal Acc' if 'Val Bal Acc' in df.columns and not df['Val Bal Acc'].isna().all() else 'Val Precision'
        
        # Val target Distribution
        ax = axes[0, 0]
        ax.hist(df[target_metric].dropna(), bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(df[target_metric].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[target_metric].mean():.4f}')
        ax.set_title(f'Distribuição {target_metric}', fontweight='bold')
        ax.set_xlabel(target_metric)
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Val secondary Distribution
        ax = axes[0, 1]
        ax.hist(df[secondary_metric].dropna(), bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(df[secondary_metric].mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {df[secondary_metric].mean():.4f}')
        ax.set_title(f'Distribuição {secondary_metric}', fontweight='bold')
        ax.set_xlabel(secondary_metric)
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Val Loss Distribution
        ax = axes[1, 0]
        ax.hist(df['Val Loss'].dropna(), bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(df['Val Loss'].mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {df["Val Loss"].mean():.4f}')
        ax.set_title('Distribuição Val Loss', fontweight='bold')
        ax.set_xlabel('Val Loss')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Boxplot de todas as métricas
        ax = axes[1, 1]
        data_to_plot = [
            df[target_metric].dropna().values,
            df[secondary_metric].dropna().values,
            (df['Val Loss'].dropna().values / df['Val Loss'].max())  # Normalizar para visualização
        ]
        bp = ax.boxplot(data_to_plot, tick_labels=[target_metric, secondary_metric, 'Val Loss (norm)'])
        ax.set_title('Boxplot de Métricas', fontweight='bold')
        ax.set_ylabel('Valor')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.reports_dir / "distribution_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo: {output_path}")
        plt.close()
    
    def generate_summary_report(self, df, metrics, best_hps):
        """Gera relatório resumido em texto."""
        output_path = self.reports_dir / "training_analysis_summary.txt"
        
        target_metric = 'Val Macro-F1' if 'Val Macro-F1' in df.columns and not df['Val Macro-F1'].isna().all() else 'Val AUC'
        secondary_metric = 'Val Bal Acc' if 'Val Bal Acc' in df.columns and not df['Val Bal Acc'].isna().all() else 'Val Precision'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE TREINAMENTO E HIPERPARÂMETROS\n")
            f.write("=" * 80 + "\n\n")
            
            # Seção 1: Estatísticas Gerais
            f.write("1. ESTATÍSTICAS GERAIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total de Trials: {len(df)}\n")
            if target_metric in df.columns:
                f.write(f"{target_metric} - Mínimo: {df[target_metric].min():.4f}\n")
                f.write(f"{target_metric} - Máximo: {df[target_metric].max():.4f}\n")
                f.write(f"{target_metric} - Média: {df[target_metric].mean():.4f}\n")
                f.write(f"{target_metric} - Std Dev: {df[target_metric].std():.4f}\n")
            if secondary_metric in df.columns:
                f.write(f"{secondary_metric} - Média: {df[secondary_metric].mean():.4f}\n")
            f.write(f"Val Loss - Mínimo: {df['Val Loss'].min():.4f}\n\n")
            
            # Seção 2: Top 5 Trials
            f.write(f"2. TOP 5 TRIALS (por {target_metric})\n")
            f.write("-" * 80 + "\n")
            if target_metric in df.columns:
                top_5 = df.nlargest(5, target_metric)
                for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                    f.write(f"\nRank #{idx}: Trial {row['Trial ID']}\n")
                    f.write(f"  {target_metric}:       {row[target_metric]:.4f}\n")
                    if secondary_metric in row:
                        f.write(f"  {secondary_metric}: {row[secondary_metric]:.4f}\n")
                    f.write(f"  Val Loss:      {row['Val Loss']:.4f}\n")
                    f.write(f"  Dropout:       {row['Dropout']}\n")
                    f.write(f"  Learning Rate: {row['Learn Rate']}\n")
                    f.write(f"  Peso Classe 0: {row['Peso C0']}\n")
            
            f.write("\n\n")
            
            # Seção 3: Análise por Hiperparâmetro
            f.write("3. ANÁLISE POR HIPERPARÂMETRO\n")
            f.write("-" * 80 + "\n\n")
            if target_metric in df.columns:
                f.write("Dropout:\n")
                dropout_stats = df.groupby('Dropout')[target_metric].agg(['count', 'mean', 'std', 'min', 'max'])
                f.write(dropout_stats.to_string())
                f.write("\n\n")
                
                f.write("Learning Rate:\n")
                lr_stats = df.groupby('Learn Rate')[target_metric].agg(['count', 'mean', 'std', 'min', 'max'])
                f.write(lr_stats.to_string())
                f.write("\n\n")
                
                f.write("Peso Classe 0:\n")
                peso_stats = df.groupby('Peso C0')[target_metric].agg(['count', 'mean', 'std', 'min', 'max'])
                f.write(peso_stats.to_string())
                f.write("\n\n")
            
            # Seção 4: Métricas Finais
            f.write("4. MÉTRICAS FINAIS DE TESTE\n")
            f.write("-" * 80 + "\n")
            if metrics:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            # Seção 5: Melhores Hiperparâmetros
            f.write("5. MELHORES HIPERPARÂMETROS ENCONTRADOS\n")
            f.write("-" * 80 + "\n")
            if best_hps:
                for key, value in best_hps.items():
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Seção 6: Recomendações
            f.write("6. RECOMENDAÇÕES DE MELHORIA\n")
            f.write("-" * 80 + "\n")
            f.write("""
✓ RECOMENDAÇÃO 1: Expandir espaço de otimizadores
  - Adicionar RMSprop e SGD ao lado do Adam
  - Esperado: +1-2% de AUC
  - Esforço: Médio

✓ RECOMENDAÇÃO 2: Ativar Quantization-Aware Training (QAT)
  - ENABLE_HARD_WEIGHT_CONSTRAINT_DURING_TRAIN = True
  - Esperado: Preservar +2-5% pós-exportação FPGA
  - Esforço: Baixo

✓ RECOMENDAÇÃO 3: Aumentar épocas de busca
  - Aumentar de 100 para 150 épocas
  - Esperado: Melhor convergência +0.5-1%
  - Esforço: Alto (tempo)

✓ RECOMENDAÇÃO 4: Implementar Learning Rate Schedule
  - Usar ReduceLROnPlateau
  - Esperado: +1-2% de estabilidade
  - Esforço: Baixo

✓ RECOMENDAÇÃO 5: Adicionar L2 Regularization
  - Expandir espaço para [1e-5, 1e-4, 1e-3]
  - Esperado: Melhor generalização +0.5%
  - Esforço: Médio
""")
        
        print(f"✅ Relatório salvo: {output_path}")
        return output_path
    
    def run_full_analysis(self):
        """Executa análise completa."""
        print("\n" + "=" * 80)
        print("INICIANDO ANÁLISE DE RESULTADOS DE TREINAMENTO")
        print("=" * 80 + "\n")
        
        # Carregar dados
        print("📊 Carregando dados...")
        df = self.load_hyperparameters_history()
        metrics = self.load_training_metrics()
        best_hps = self.load_best_hyperparameters()
        
        if df is None or metrics is None or best_hps is None:
            print("❌ Erro ao carregar dados. Abortando análise.")
            return
        
        print("✅ Dados carregados com sucesso!\n")
        
        # Gerar gráficos
        print("📈 Gerando gráficos...")
        self.plot_hyperparameter_comparison(df)
        self.plot_trial_rankings(df)
        self.plot_distribution_analysis(df)
        
        # Gerar relatório
        print("📝 Gerando relatório...")
        self.generate_summary_report(df, metrics, best_hps)
        
        print("\n" + "=" * 80)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print(f"\n📁 Arquivos gerados em: {self.reports_dir}")
        print("\nArquivos gerados:")
        print("  • hyperparameter_analysis.png")
        print("  • trial_rankings.png")
        print("  • distribution_analysis.png")
        print("  • training_analysis_summary.txt")


if __name__ == "__main__":
    analyzer = TrainingAnalyzer()
    analyzer.run_full_analysis()

