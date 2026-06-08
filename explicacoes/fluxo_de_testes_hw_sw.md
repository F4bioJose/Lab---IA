# Fluxo de Testagem: CNN Hardware vs. Software

Este documento descreve o fluxo oficial passo a passo para testar e validar o alinhamento entre o modelo Keras (Software) e a implementação na FPGA (Hardware).

O processo garante que ambos os sistemas processem a mesma imagem visual, mas cada um respeitando estritamente os seus domínios de representação (Ponto Flutuante vs. Ponto Fixo Q1.7/Q2.14/Q6.10).

> **Atalho:** Para executar todo o fluxo automaticamente em um único comando, use o script [`run_pipeline.sh`](../scripts/run_pipeline.sh):
> ```bash
> ./scripts/run_pipeline.sh anna.jpg
> ```
> Ele encadeia os passos 2, 3 e 4 abaixo e salva os resultados em `comparacao/plots/<nome>/`.

---

## 1. Entrada: A Imagem Bruta
Todo o fluxo começa a partir de um arquivo de imagem convencional (ex: `.jpg`, `.png`).
**Exemplo:** `inputs/frame0.jpg`

---

## 2. Ramo de Teste do Hardware (FPGA)

O hardware não processa imagens `.jpg` nativamente, e também opera com precisão inteira quantizada (formato Q1.7, onde cada pixel é um número inteiro de `-128` a `127`).

### Passo 2.1: Conversão e Pré-processamento
Para simular a FPGA, a imagem `.jpg` deve ser convertida para um arquivo de texto hexadecimal (`.txt`).
Durante essa conversão, a imagem sofre os mesmos filtros que a rede aprendeu no treinamento:
- Conversão para escala de cinza.
- Normalização de Iluminação (CLAHE).
- Detecção da face (Haarcascade adaptativo).
- Recorte (crop) com *padding* de 15%.
- Redimensionamento para 32x32.
- Quantização para inteiros Q1.7.

**Comando:**
```bash
python scripts/image_to_hex.py inputs/frame0.jpg
```
**Saída:** `inputs/imgs_hex/frame0_hex.txt` (Contém 1024 linhas com valores hexadecimais, simulando a RAM de vídeo).

> **Nota:** O segundo argumento (caminho de saída) é opcional. Se omitido, o script salva automaticamente em `inputs/imgs_hex/<nome>_hex.txt`.

### Passo 2.2: Simulação Verilog (ModelSim)
O arquivo hexadecimal gerado é então empurrado para o *testbench* Verilog, que injeta os pixels na simulação. A rede calcula as camadas (Convolucional, MaxPool, Flatten, Dense) usando acumuladores Q2.14 (16-bits).
Os resultados de cada camada são salvos em arquivos `.txt`.

**Comando:**
```bash
vsim -c -do "do scripts/run_project.do inputs/frame0_hex.txt comparacao/frame0_teste/hw/"
```
**Saída:** Arquivos de ativações na subpasta informada (ex: `comparacao/frame0_teste/hw/dense_out.txt`).

---

## 3. Ramo de Teste do Software (Keras / Python)

Para manter a testagem justa, o script de Software processa exatamente a mesma imagem bruta, mas simula as ativações internas da rede, quantizando os resultados de saída para coincidir com a FPGA.

### Passo 3.1: Extração de Ativações Keras
O script Keras consome o arquivo `.jpg` diretamente. Ele aplica a mesmíssima pipeline de processamento visual via OpenCV (CLAHE, padding 15%). Em seguida, os *Logits* de saída de todas as camadas intermediárias são calculados. 
Para permitir a comparação exata contra o Hardware, as saídas são convertidas (clippadas) para o teto do **Q2.14** (saturando nos limites do registrador de 16-bits da placa).

**Comando:**
```bash
python scripts/extract_sw_activations.py --model tiny_cnn_multiclasse.h5 --image inputs/frame0.jpg --outdir comparacao/frame0_teste/sw/
```
**Saída:** Arquivos de ativações na subpasta (ex: `comparacao/sw/frame0/dense_out_sw.txt`).

> **Nota:** Se omitir o `--outdir`, o script criará automaticamente uma subpasta baseada no nome da imagem em `comparacao/sw/<nome>/`.

---

## 4. Comparação Final e Validação

Com as duas coleções de arquivos `.txt` isoladas e no mesmo formato inteiro de 16-bits, o comparador cruza os dados camada por camada.
Ele avalia MAE, RMSE, MaxErr, Exato%, e a Correlação de Pearson (`r`).

A forma mais simples de rodar a comparação é passar a imagem original — o script infere automaticamente os diretórios de SW e HW:

**Comando (recomendado):**
```bash
python scripts/compare_sw_hw.py --image inputs/frame0.jpg
```

**Comando (explícito, se os diretórios não seguirem a convenção padrão):**
```bash
python scripts/compare_sw_hw.py --sw-dir comparacao/sw/frame0/ --hw-dir comparacao/hw/frame0/ --outdir comparacao/plots/frame0/
```

**Saída:** Relatório final em `comparacao/plots/frame0/report.txt` e gráficos na mesma pasta.

Para uma explicação detalhada de cada métrica e gráfico gerado, consulte [`metricas_e_graficos.md`](metricas_e_graficos.md).
