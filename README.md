# Tiny-CNN FPGA — Fechadura Biométrica Inteligente

Implementação em hardware de uma Rede Neural Convolucional (CNN) sintetizada na FPGA **DE2-115 (Cyclone IV EP4CE115F29C7)** para classificação biométrica facial em tempo real. O sistema recebe uma imagem via UART, executa a inferência completa em pipeline de hardware dedicado e exibe o resultado nos LEDs verdes da placa.

---

## Visão Geral

```
PC (Python)  ──[UART 115200]──►  FPGA DE2-115
                                      │
                               uart_rx  →  Framebuffer 32×32
                                               │ (auto-start ao receber 1024 bytes)
                                      Line Buffer  →  Convolução 3×3 (4 filtros, ReLU)
                                                              │
                                                      Max Pooling 2×2
                                                              │
                                                         Flatten (900)
                                                              │
                                                    Dense 900×19 (19 classes)
                                                              │
                                                     Argmax Puro (sem threshold)
                                                              │
                                          class_id[4:0] → LEDG[4:0]
                                          unknown       → LEDG[7]
                                          done          → LEDG[6]
```

**Capacidade:** 19 classes — classe 0 = Desconhecido (nativa da rede via Softmax), classes 1–18 = membros autorizados.

### Mapeamento de Classes (Ordem Keras — String Sort)

| ID | Nome | ID | Nome | ID | Nome |
|----|------|----|------|----|------|
| 0 | Desconhecido | 7 | Rafael | 14 | Fabio |
| 1 | Igor | 8 | Samuel | 15 | Felipe |
| 2 | Joao | 9 | Yuri | 16 | Gabriel |
| 3 | Jose Henrique | 10 | Anna Carol | 17 | Horacio |
| 4 | Julia | 11 | Bruno | 18 | Hugo |
| 5 | Lucio | 12 | Diego | | |
| 6 | Naira | 13 | Eduardo | | |

> **Nota:** A ordem dos índices é definida automaticamente pelo Keras (`ImageDataGenerator`), que ordena as pastas do dataset por string sort. Pastas nomeadas `1_`, `10_`, `11_`... são agrupadas antes de `2_`, resultando nesta sequência específica.

---

## Estrutura do Repositório

```
Lab---IA/
├── modulos_verilog/              # Módulos Verilog do pipeline CNN
│   ├── fpga_top_de2115.v         # Top-level sintetizável (DE2-115)
│   ├── cnn_top.v                 # Orquestrador FSM + instâncias
│   ├── uart_rx.v                 # Receptor serial UART
│   ├── framebuffer_32x32.v       # Buffer de imagem 32×32
│   ├── line_buffer_32x32.v       # Janelas 3×3 por linha buffer
│   ├── convolucao_mac.v          # Convolução 3×3 (4 filtros + ReLU)
│   ├── max_pooling_design.v      # Max Pooling 2×2
│   ├── flatten.v                 # Serialização 2D → 1D (900 elementos)
│   ├── dense_900x19.v            # Camada densa (900 × 19 classes)
│   ├── argmax_19.v               # Decisão: argmax puro (sem threshold)
│   ├── weights_shared_rom.v      # ROM de pesos (arquivo único .hex)
│   ├── weights_all.mif           # Pesos pré-treinados (fonte Quartus)
│   ├── weights_all.hex           # Pesos convertidos (simulação)
│   └── tb_cnn_layer_capture.v    # Testbench: captura saída por camada
│
├── quartus_cnn/                  # Projeto Quartus Prime
│   ├── cnn_inference.qpf         # Arquivo de projeto
│   ├── cnn_inference.qsf         # Pin assignments e configurações
│   └── cnn_inference.sdc         # Timing constraints (50 MHz)
│
├── scripts/                      # Automação e testagem
│   ├── run_project.do            # Compilação + simulação (Questa/ModelSim)
│   ├── extract_sw_activations.py # Extração SW de ativações por camada
│   ├── compare_sw_hw.py          # Comparação gráfica SW vs. HW
│   ├── image_to_hex.py           # Converte .jpg → .txt hex para testbench
│   ├── send_image_32x32.py       # Envio de imagem para a FPGA via UART
│   └── test_model_19classes.py   # Teste de sanidade da rede Keras
│
├── rede_pipeline/                # Pipeline de treinamento da rede
│   ├── src/                      # Código-fonte do treinamento
│   │   ├── preprocessor.py       # Pré-processamento oficial (CLAHE + Haar)
│   │   ├── model.py              # Arquitetura TinyCNN
│   │   ├── engine.py             # Motor de treino e tuning
│   │   ├── export_mif.py         # Exportação de pesos → .mif/.hex
│   │   └── ...
│   └── scripts/                  # Scripts auxiliares do pipeline
│       ├── inference_webcam.py   # Inferência ao vivo via webcam
│       └── ...
│
├── inputs/                       # Imagens de teste
├── comparacao/                   # Saídas de comparação SW vs. HW
├── explicacoes/                  # Documentação técnica
│   ├── pipeline.md               # Arquitetura completa e fluxo de dados
│   ├── FSM.md                    # Máquinas de estado (diagramas + ciclos)
│   └── fluxo_de_testes_hw_sw.md  # Guia do fluxo de testagem
│
├── notebooks_rede/               # Notebooks de experimentação
├── haarcascade_frontalface_default.xml
├── tiny_cnn_multiclasse.h5       # Modelo treinado (19 classes)
└── README.md
```

---

## Fluxo de Trabalho

### 1 · Pré-processamento Padrão

Todos os scripts seguem rigorosamente o pipeline definido em `rede_pipeline/src/preprocessor.py`:

1. Conversão para escala de cinza
2. CLAHE (`clipLimit=2.0, tileGridSize=(8,8)`)
3. Detecção facial adaptativa em cascata: `[(1.2,5,60), (1.1,3,40), (1.05,2,30)]`
4. Crop retangular com 15% de padding
5. Resize para 32×32 com `INTER_AREA`
6. Normalização: `/255.0` (SW) ou quantização Q1.7 (HW)

### 2 · Simulação (Questa/ModelSim)

O script `run_project.do` unifica compilação e simulação num fluxo único:

```tcl
# Simula com a imagem padrão
do scripts/run_project.do

# Simula com outra imagem
do scripts/run_project.do inputs/frame0_hex.txt

# Simula e salva saídas em pasta específica
do scripts/run_project.do inputs/anna_hex.txt comparacao/anna/hw/
```

**Saídas geradas:**

| Arquivo | Conteúdo |
|---|---|
| `conv_out.txt` | Ativações conv pós-ReLU: `f0 f1 f2 f3` por linha (Q2.14) |
| `pool_out.txt` | Saídas do Max Pooling serializadas (Q2.14) |
| `flat_out.txt` | Vetor Flatten de 900 elementos (Q2.14) |
| `dense_out.txt` | 19 scores finais em uma linha (Q2.14) |

### 3 · Comparação SW vs. HW

```bash
# Extrai ativações do modelo Keras (cria subpasta automática)
python scripts/extract_sw_activations.py \
    --model tiny_cnn_multiclasse.h5 \
    --image inputs/anna.jpg

# Com diretório explícito
python scripts/extract_sw_activations.py \
    --model tiny_cnn_multiclasse.h5 \
    --image inputs/anna.jpg \
    --outdir comparacao/anna/sw/

# Gera gráficos de comparação
python scripts/compare_sw_hw.py \
    --sw-dir comparacao/anna/sw/ \
    --hw-dir comparacao/anna/hw/ \
    --outdir comparacao/anna/plots/
```

### 4 · Enviar imagem para a FPGA

```bash
# Imagem .jpg com pré-processamento completo
python scripts/send_image_32x32.py --file inputs/anna.jpg --port /dev/ttyUSB0

# Arquivo hexadecimal já processado
python scripts/send_image_32x32.py --file inputs/anna_hex.txt --hex --port /dev/ttyUSB0

# Webcam (captura ao vivo)
python scripts/send_image_32x32.py --port /dev/ttyUSB0 --preview
```

### 5 · Síntese e programação

1. Abra o Quartus Prime: **File → Open Project → `quartus_cnn/cnn_inference.qpf`**
2. Compile: **Processing → Start Compilation** (`Ctrl+L`)
3. Programe via JTAG: **Tools → Programmer → `output_files/cnn_inference.sof` → Start**
4. Pressione **KEY[0]** para reset

---

## LEDs da DE2-115

| LED | Sinal | Significado |
|-----|-------|-------------|
| `LEDG[4:0]` | `class_id` | Classe predita em binário (0 = Desconhecido; 1–18 = membro) |
| `LEDG[5]` | `debug_frame_nonzero` | Frame recebido com pixels não-nulos |
| `LEDG[6]` | `done` | Aceso = inferência concluída |
| `LEDG[7]` | `unknown` | Aceso = classe 0 (Desconhecido) predita |

Os LEDs mantêm o último resultado até reset (**KEY[0]**) ou nova inferência.

---

## Dependências Python

```bash
pip install tensorflow opencv-python numpy matplotlib scipy pyserial
```

---

## Documentação Técnica

- **[Arquitetura e Pipeline](explicacoes/pipeline.md)** — Fluxo de dados completo, detalhamento de cada módulo, formatos numéricos (Q1.7 / Q2.14 / Q3.21)
- **[Máquinas de Estado](explicacoes/FSM.md)** — FSM de inferência e lógica de captura dos LEDs, diagramas Mermaid
- **[Fluxo de Testagem](explicacoes/fluxo_de_testes_hw_sw.md)** — Guia passo a passo para validação SW vs. HW