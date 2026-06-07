# Relatório Técnico: Reconhecimento Facial para Fechadura Biométrica (Artefato 1)

## 1. Visão Geral
Este documento detalha o desenvolvimento do sistema de inteligência artificial para o projeto de uma fechadura biométrica utilizando a placa FPGA DE2-115. O foco desta etapa foi a criação de um pipeline robusto para aquisição de dados, treinamento de uma **Tiny-CNN** otimizada e a exportação dos parâmetros quantizados para o hardware.

## 2. Desenvolvimento do Dataset
O dataset foi projetado para um problema de classificação binária focado em segurança: **Classe 1 (Autorizados)** e **Classe 0 (Desconhecidos)**.

### Classe 1: Autorizados (Equipe)
* **Origem e Processamento**: Imagens extraídas de vídeos `.mp4` ou pastas, utilizando o algoritmo *Haar Cascade* com `scaleFactor=1.2` para detecção facial.
* **Normalização de Iluminação**: Implementação do **CLAHE** (Contrast Limited Adaptive Histogram Equalization) para garantir o reconhecimento sob diferentes condições de luz e sombras.
* **Aumento de Dados (Augmentation)**: Aplicação de técnicas de aumento de dados incluindo rotação leve (-15° a 15°), variações de zoom (0.9x a 1.1x), espelhamento horizontal e ajustes de brilho para atingir a meta de 400 imagens por integrante.

### Classe 0: Desconhecidos (Unknowns)
Para minimizar falsos positivos, a Classe 0 foi expandida para garantir maior diversidade:
* **Datasets Externos**: Integração do **LFW (Labeled Faces in the Wild)** via `kagglehub` e do **UCF Selfie-dataset**.
* **Estratégia de Proporção**: Utilização de uma proporção de **5:1** (Desconhecidos:Autorizados) para cobrir a variabilidade do rosto humano.
* **Paridade de Augmentation**: Aplicação das mesmas técnicas de aumento de dados (rotação/zoom) na Classe 0 para garantir a invariância de identidade.
* **Imagens de Fundo**: Inclusão de **300 imagens sintéticas** (paredes e ruído) para evitar autorizações baseadas em texturas de fundo.

## 3. Arquitetura e Treinamento Otimizado
A rede foi projetada para ser leve (**Tiny-CNN**) visando a implementação em blocos de memória M9K da FPGA.

* **Arquitetura**: Entrada 32x32 -> Conv2D (4 filtros 3x3) -> ReLU -> Max-Pooling 2x2 -> Dropout -> Dense -> Sigmoid.
* **Busca de Hiperparâmetros**: Uso do **Keras Tuner** (Hyperband) para otimizar a taxa de **Dropout (0.2 a 0.5)**, neurônios da camada oculta, **Learning Rate** e regularização **L2**.
* **Estabilidade**: Inclusão de regularização **L2** para manter os pesos pequenos, facilitando a quantização posterior.
* **Custom Loss Weighting**: Aplicação de pesos manuais (**1.0 para Classe 0 e 4.0 para Classe 1**) para priorizar o reconhecimento correto da equipe.
* **Convergência**: Treinamento com `EarlyStopping` e **paciência de 15 épocas** para garantir estabilização em condições difíceis.

## 4. Inferência e Validação
O sistema utiliza um script de validação (`inference_webcam.py`) sincronizado com o pipeline de hardware:
* **Sincronização**: Aplicação do mesmo pré-processamento (CLAHE, Padding de 15%, Redimensionamento) utilizado no treino.
* **Debug Visual**: Exibição da "Visão da CNN" (32x32) para validação da qualidade da entrada em tempo real.

## 5. Quantização e Exportação (.MIF)
Os pesos são convertidos de ponto flutuante para **Ponto Fixo (Q1.7)** para execução em hardware de 8 bits.
* **Lógica**: Multiplicação por 128 ($2^7$) e arredondamento para o intervalo de -128 a 127.
* **Arquivos**: Geração de ficheiros `.mif` formatados para os blocos de memória M9K do Intel Quartus Prime.

---
**Disciplina:** ENGG52 - LABORATÓRIO INTEGRADO I-A (UFBA)  
**Docentes:** Wagner Oliveira e Edmar de Souza