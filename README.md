# Tiny-CNN FPGA: Fechadura Biométrica Inteligente

Implementação em hardware de uma Rede Neural Convolucional (CNN) na FPGA DE2-115 (Cyclone IV EP4CE115F29C7) para classificação biométrica facial. O sistema recebe uma imagem 32×32 em escala de cinza via comunicação serial UART, executa a inferência completa em pipeline de hardware dedicado e exibe o resultado da classificação nos LEDs verdes da placa.

## Status Atual

* **Pipeline Funcional em Hardware:** O fluxo completo de inferência está implementado e sintetizável — desde a recepção serial da imagem até a decisão final exibida nos LEDs.
* **Pesos Sintéticos:** O arquivo de pesos (`weights_all.mif`) carrega dados gerados sinteticamente para validação estrutural. Para classificação real, deve ser substituído por pesos exportados do modelo TensorFlow/Keras treinado no notebook `pipeline_fechadura.ipynb`.
* **Imagem de Teste:** O arquivo `inputs/teste2.txt` contém uma imagem hexadecimal para testes. Imagens reais podem ser enviadas via o script Python incluído.

## Arquitetura do Sistema

```
PC (send_image_32x32.py)
  │  1024 bytes via RS-232 (115200 baud)
  ▼
UART_RXD ──► uart_rx ──► Framebuffer 32×32
                              │
                    FSM auto-start (1024 bytes recebidos)
                              │
                              ▼
                    Line Buffer ──► Conv 3×3 (4 filtros, ReLU)
                                          │
                                    Max Pooling 2×2
                                          │
                                      Flatten (900)
                                          │
                                   Dense 900×7 (7 classes)
                                          │
                                  Argmax + Threshold (0.7)
                                          │
                                  class_id ──► LEDG[2:0]
                                  unknown  ──► LEDG[7]
                                  done     ──► LEDG[6]
```

## Estrutura de Diretórios

```
Lab---IA/
├── modulos_verilog/           # Módulos Verilog do pipeline CNN
│   ├── fpga_top_de2115.v      # Wrapper top-level para síntese na DE2-115
│   ├── cnn_top.v              # Orquestrador do pipeline (FSM + instâncias)
│   ├── uart_rx.v              # Receptor serial UART (115200 baud)
│   ├── framebuffer_32x32.v    # Buffer de imagem 32×32 (RAM dual-port)
│   ├── line_buffer_32x32.v    # Buffer de linhas para janelas 3×3
│   ├── convolucao_mac.v       # Convolução 3×3 (4 filtros + ReLU)
│   ├── max_pooling_design.v   # Max Pooling 2×2 com FIFO
│   ├── flatten.v              # Serialização de mapas 2D → vetor 1D
│   ├── dense_900x7.v          # Camada densa (900 entradas × 7 classes)
│   ├── argmax_threshold_7.v   # Decisão: argmax + limiar de confiança
│   ├── weights_shared_rom.v   # ROM de pesos (6347 parâmetros Q1.7)
│   ├── weights_all.mif        # Arquivo de inicialização dos pesos
│   └── tb_fpga_top.v          # Testbench (simula envio UART completo)
│
├── quartus_cnn/               # Projeto Quartus Prime para síntese
│   ├── cnn_inference.qpf      # Arquivo de projeto
│   ├── cnn_inference.qsf      # Pin assignments e configurações
│   └── cnn_inference.sdc      # Timing constraints (50 MHz)
│
├── scripts/                   # Scripts de automação
│   ├── compile_project.do     # Compilação ModelSim (todos os módulos)
│   ├── run_project.do         # Execução do testbench no ModelSim
│   └── send_image_32x32.py    # Envio de imagem via UART (Python)
│
├── inputs/                    # Imagens de teste
│   └── teste2.txt             # Imagem 32×32 em formato hexadecimal
│
├── explicacoes/               # Documentação técnica detalhada
│   ├── pipeline.txt           # Descrição completa do pipeline e módulos
│   └── FSM.md                 # Diagramas e explicação das FSMs
│
├── pipeline_fechadura.ipynb   # Notebook de treino do modelo (TensorFlow/Keras)
└── README.md                  # Este arquivo
```

## Como Simular (ModelSim)

Pré-requisito: ModelSim instalado e imagem de teste em `inputs/teste2.txt`.

1. Abra o terminal do ModelSim com o diretório atual na **raiz do repositório**.
2. Compile todos os módulos:
   ```tcl
   do scripts/compile_project.do
   ```
3. Execute a simulação:
   ```tcl
   do scripts/run_project.do
   ```

O testbench `tb_fpga_top` simulará:
- Reset do sistema.
- Envio serial de 1024 bytes via protocolo UART (baud acelerado para simulação).
- Inferência completa pelo pipeline CNN.
- Exibição no console: transições da FSM, scores das 7 classes e resultado final (class_id, unknown).

## Como Sintetizar e Programar o FPGA

1. Abra o **Quartus Prime** e carregue `quartus_cnn/cnn_inference.qpf`.
2. Compile: **Processing → Start Compilation** (ou `Ctrl+L`).
3. Programe a FPGA via JTAG: **Tools → Programmer** → selecione `output_files/cnn_inference.sof` → **Start**.
4. Pressione **KEY[0]** para realizar o reset.

## Como Enviar Imagem para a FPGA

Pré-requisitos: Python 3, pacotes `opencv-python`, `pyserial`, `numpy`.

```bash
# Instalar dependências
pip install opencv-python pyserial numpy

# Enviar arquivo hexadecimal de teste
python3 scripts/send_image_32x32.py --file inputs/teste2.txt --hex --port /dev/ttyUSB0

# Enviar uma fotografia (com detecção de rosto automática)
python3 scripts/send_image_32x32.py --file foto.jpg --port /dev/ttyUSB0 --preview

# Capturar da webcam e enviar
python3 scripts/send_image_32x32.py --port /dev/ttyUSB0 --preview

# Testar sem a placa (modo simulação)
python3 scripts/send_image_32x32.py --mock --file inputs/teste2.txt --hex
```

## Leitura do Resultado nos LEDs

| LED | Significado |
|-----|-------------|
| `LEDG[2:0]` | Classe predita em binário (0–6 = membros, 7 = desconhecido) |
| `LEDG[6]` | Aceso = inferência concluída |
| `LEDG[7]` | Aceso = score abaixo do limiar (desconhecido) |
| `LEDG[5:3]` | Não utilizados (apagados) |

Os LEDs permanecem acesos com o último resultado até que um novo reset (`KEY[0]`) ou uma nova imagem seja enviada.

## Documentação Estendida

Para compreensão detalhada de cada módulo, do formato de ponto fixo (Q1.7, Q2.14, Q3.21), do fluxo de dados completo e dos procedimentos de integração futura:

* **[Pipeline e Arquitetura](explicacoes/pipeline.txt)** — Descrição exaustiva de todos os módulos, barramentos e formatos numéricos.
* **[Máquinas de Estados (FSMs)](explicacoes/FSM.md)** — Diagramas Mermaid e explicação ciclo-a-ciclo da FSM de inferência e da lógica de captura dos LEDs.