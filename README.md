# Tiny-CNN FPGA — Fechadura Biométrica Inteligente

Implementação em hardware de uma Rede Neural Convolucional (CNN) sintetizada na FPGA **DE2-115 (Cyclone IV EP4CE115F29C7)** para classificação biométrica facial em tempo real. O sistema recebe qualquer imagem via UART, executa a inferência completa em pipeline de hardware dedicado e exibe o resultado nos LEDs verdes da placa.

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
                                                    Dense 900×18 (18 classes)
                                                              │
                                               Argmax + Threshold 95%
                                                              │
                                          class_id[4:0] → LEDG[4:0]
                                          unknown       → LEDG[7]
                                          done          → LEDG[6]
```

**Capacidade:** 18 classes de usuários + rejeição automática por limiar de confiança (95%).

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
│   ├── dense_900x18.v            # Camada densa (900 × 18 classes)
│   ├── argmax_threshold_18.v     # Decisão: argmax + threshold 95%
│   ├── weights_shared_rom.v      # ROM de pesos (arquivo único .hex)
│   ├── weights_all.mif           # Pesos pré-treinados (fonte)
│   ├── weights_all.hex           # Pesos convertidos (gerado pelo compile.do)
│   └── tb_cnn_layer_capture.v    # Testbench: captura saída por camada
│
├── quartus_cnn/                  # Projeto Quartus Prime
│   ├── cnn_inference.qpf         # Arquivo de projeto
│   ├── cnn_inference.qsf         # Pin assignments e configurações
│   └── cnn_inference.sdc         # Timing constraints (50 MHz)
│
├── scripts/                      # Automação
│   ├── compile_project.do        # Compilação Questa (converte .mif + vlog)
│   ├── run_project.do            # Simulação com imagem configurável
│   ├── extract_sw_activations.py # Extração SW de ativações por camada
│   ├── compare_sw_hw.py          # Comparação gráfica SW vs. HW
│   └── send_image_32x32.py       # Envio de imagem para a FPGA via UART
│
├── inputs/                       # Imagens de teste
│   └── teste2.txt                # Imagem 32×32 em hexadecimal (1024 bytes)
│
├── explicacoes/                  # Documentação técnica
│   ├── pipeline.md               # Arquitetura completa e fluxo de dados
│   └── FSM.md                    # Máquinas de estado (diagramas + ciclos)
│
├── notebooks_rede/               # Treinamento do modelo
│   └── pipeline_fechadura.ipynb  # Notebook Keras (treino + exportação .h5)
│
├── imagens/                      # Dataset de rostos por usuário
├── haarcascade_frontalface_default.xml
├── tiny_cnn_final.h5             # Modelo treinado atual (7 classes — legado)
└── README.md
```

---

## Fluxo de Trabalho

### 1 · Quando chegar um novo modelo treinado

```bash
# Substitua weights_all.mif pelo novo arquivo fornecido
# A conversão para .hex ocorre automaticamente na compilação
```

### 2 · Simulação (Questa FSE)

Abra o Questa/ModelSim **com o terminal na raiz do repositório**.

```tcl
# Passo único: converte .mif → .hex + compila todos os módulos
do scripts/compile_project.do

# Simula com a imagem padrão (inputs/teste2.txt)
do scripts/run_project.do

# Simula com outra imagem
do scripts/run_project.do inputs/frame0.txt

# Simula e salva saídas em pasta específica
do scripts/run_project.do inputs/frame0.txt comparacao/hw/frame0/
```

**Saídas geradas** em `comparacao/hw/` (ou na pasta especificada):

| Arquivo | Conteúdo |
|---|---|
| `conv_out.txt` | Ativações conv pós-ReLU: `f0 f1 f2 f3` por linha (Q2.14) |
| `pool_out.txt` | Saídas do Max Pooling serializadas (Q2.14) |
| `flat_out.txt` | Vetor Flatten de 900 elementos (Q2.14) |
| `dense_out.txt` | 18 scores finais em uma linha (Q2.14) |

### 3 · Comparação SW vs. HW

```bash
# Extrai ativações do modelo Keras para qualquer imagem
python scripts/extract_sw_activations.py \
    --model tiny_cnn_final.h5 \
    --image inputs/aprovado.jpg \
    --outdir comparacao/sw/

# Gera gráficos de comparação (histogramas, scatter, barplot, heatmap)
python scripts/compare_sw_hw.py \
    --sw-dir comparacao/sw/ \
    --hw-dir comparacao/hw/
```

### 4 · Enviar imagem para a FPGA

```bash
# Imagem .jpg com detecção de rosto automática
python scripts/send_image_32x32.py --file inputs/aprovado.jpg --port /dev/ttyUSB0

# Arquivo hexadecimal de teste
python scripts/send_image_32x32.py --file inputs/teste2.txt --hex --port /dev/ttyUSB0

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
| `LEDG[4:0]` | `class_id` | Classe predita em binário (0–17 = membro; 18 = negado) |
| `LEDG[5]` | `debug_frame_nonzero` | Frame recebido com pixels não-nulos |
| `LEDG[6]` | `done` | Aceso = inferência concluída |
| `LEDG[7]` | `unknown` | Aceso = score abaixo do limiar de 95% |

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