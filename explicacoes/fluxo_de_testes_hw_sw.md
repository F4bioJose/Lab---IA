# Fluxo de Testagem: CNN Hardware vs. Software

Este documento descreve o fluxo oficial passo a passo para testar e validar o alinhamento perfeito entre o modelo Keras (Software) e a implementação na FPGA (Hardware).

O processo garante que ambos os sistemas processem a mesma imagem visual, mas cada um respeitando estritamente os seus domínios de representação (Ponto Flutuante vs. Ponto Fixo Q1.7/Q2.14).

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
python scripts/image_to_hex.py inputs/frame0.jpg inputs/frame0_hex.txt
```
**Saída:** `inputs/frame0_hex.txt` (Contém 1024 linhas com valores hexadecimais, simulando a RAM de vídeo).

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
**Saída:** Arquivos de ativações na subpasta (ex: `comparacao/frame0_teste/sw/dense_out_sw.txt`).
*(Dica: Se omitir o `--outdir`, o script criará automaticamente uma pasta com timestamp, e.g., `comparacao/frame0_20260607_143500/sw/`).*

---

## 4. Comparação Final e Validação (Pearson)

Com as duas coleções de arquivos `.txt` isoladas e no mesmo formato inteiro de 16-bits, o comparador cruza os dados camada por camada. 
Ele avalia o RMSE (Erro Quadrático Médio) e a Correlação de Pearson (`r`).

- **Correlação nas camadas iniciais:** Deverá atingir níveis estratosféricos (ex: `r = 0.9998`), com microvariações surgindo unicamente porque o SW opera em precisão flutuante infinita contra os limites do ponto fixo Q1.7 original.
- **Camada Densa (Final):** Como os limites de ativação ultrapassam o que um registrador de 16-bits comporta (`[-2.0, +1.999]`), o *Logit* correto saturará no limite exato de `32767`, enquanto as demais classes erradas saturarão em `-32768`. Isso gera uma Correlação de Pearson cravada de **1.000**.

**Comando:**
```bash
python scripts/compare_sw_hw.py --sw-dir comparacao/frame0_teste/sw/ --hw-dir comparacao/frame0_teste/hw/ --outdir comparacao/frame0_teste/plots/
```
**Saída:** Relatório final detalhado em `comparacao/frame0_teste/plots/report.txt` e gráficos anexos na mesma pasta.
