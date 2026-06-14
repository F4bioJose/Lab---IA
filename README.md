# Sistema de Classificação Facial com CNN em FPGA

Implementação em hardware de uma Rede Neural Convolucional (CNN) com 19 classes, sintetizada na FPGA **DE2-115 (Cyclone IV EP4CE115F29C7)**. O sistema recebe imagens de uma webcam via UART, exibe a imagem no monitor VGA em tempo real e executa a inferência em lógica digital dedicada. O resultado da classificação é exibido na tela como um sprite de texto e sinalizado nos LEDs da placa.

---

## Arquitetura

O projeto é composto por dois pipelines que compartilham o mesmo framebuffer de imagem:

- **Pipeline CNN (50 MHz):** Recebe a imagem, processa as camadas da rede neural (convolução, max pooling, camada densa) e identifica a classe.
- **Pipeline VGA (25 MHz):** Lê a mesma imagem do framebuffer e a renderiza no monitor, junto com um sprite de texto contendo o nome da classe predita.

```
PC (Python + OpenCV)  ──[UART 2 Mbaud]──►  FPGA DE2-115
                                                │
                                          Byte de controle
                                           ┌────┴────┐
                                           ▼         ▼
                                     Rosto 32×32   Vídeo 128×128
                                           │         │
                                    framebuffer   framebuffer
                                      32×32        128×128
                                           │         │
                   ┌───────────────────────┴─┐       │
                   ▼                         ▼       ▼
          Pipeline CNN (50 MHz)     Pipeline VGA (25 MHz)
       Line Buffer → Conv 3×3      vga_sync → pixel_x, pixel_y
                  │                              │
           Max Pooling 2×2          Escalonamento (12× ou 3×)
                  │                              │
            Flatten (900)            ROM de Sprites de Nomes
                  │                              │
          Dense 900×19              Composição: Imagem + Sprite
                  │                              │
             Argmax 19                           ▼
                  │                       Monitor VGA
           class_id[4:0] ──► FSM Orquestradora ──┘
```

### Fluxo de Operação

1. O script Python captura um frame da webcam e tenta detectar um rosto via Haarcascade.
2. Se um rosto é detectado, envia o byte de controle `0xFF` seguido de 1024 bytes (imagem 32×32 em escala de cinza, quantizada em Q1.7). Se não detecta rosto, envia `0x00` seguido de 16384 bytes (imagem 128×128 para exibição apenas).
3. A FPGA armazena a imagem no framebuffer correspondente. O VGA lê continuamente e exibe a imagem ampliada (12× para rosto, 3× para vídeo) centrada no monitor (384×384 pixels na região central de 640×480).
4. Para frames de rosto, o pipeline CNN processa automaticamente: convolução com 4 filtros 3×3, max pooling 2×2, camada densa 900→19 e argmax.
5. O `class_id` resultante determina qual sprite de nome aparece na parte inferior da tela (verde para pessoa reconhecida, vermelho para desconhecido) e qual LED acende.

---

## Classes

A rede foi treinada com 19 neurônios de saída. O neurônio 0 representa "Desconhecido". Na saída do módulo `argmax_19`, soma-se 1 ao índice, de modo que `class_id = 0` fica reservado para indicar que nenhuma inferência foi realizada ("Vazio").

| class_id | Nome | class_id | Nome | class_id | Nome |
|----------|------|----------|------|----------|------|
| 0 | Vazio | 7 | Naira | 14 | Eduardo |
| 1 | **Desconhecido** | 8 | Rafael | 15 | Fabio |
| 2 | Igor | 9 | Samuel | 16 | Felipe |
| 3 | Joao | 10 | Yuri | 17 | Gabriel |
| 4 | Jose Henrique | 11 | Anna Carol | 18 | Horacio |
| 5 | Julia | 12 | Bruno | 19 | Hugo |
| 6 | Lucio | 13 | Diego | | |

---

## Estrutura de Arquivos

```
Lab---IA/
├── README.md                               ← Este arquivo
├── haarcascade_frontalface_default.xml      ← Classificador Haar para detecção de rostos
│
├── modulos_verilog/                         ← Módulos Verilog do pipeline CNN + orquestrador
│   ├── fpga_top_unified.v                   ← Top-level (une CNN e VGA)
│   ├── cnn_top.v                            ← Orquestrador e FSM do pipeline CNN
│   ├── uart_rx.v                            ← Receptor UART
│   ├── framebuffer_32x32.v                  ← BRAM dual-port para rosto (1024 bytes)
│   ├── framebuffer_128x128.v                ← BRAM para vídeo (16384 bytes)
│   ├── line_buffer_32x32.v                  ← Buffer de linhas para janelamento 3×3
│   ├── convolucao_mac.v                     ← 4 filtros convolucionais + ReLU
│   ├── max_pooling_design.v                 ← Max pooling 2×2 com FIFO
│   ├── flatten.v                            ← Serialização dos mapas de características
│   ├── dense_900x19.v                       ← Camada fully-connected (900→19)
│   ├── argmax_19.v                          ← Seleção da classe com maior score
│   ├── weights_shared_rom.v                 ← ROM de pesos com bootloader de hardware
│   └── weights_all.mif                      ← Pesos quantizados (INT8, 17159 bytes)
│
├── vga_artefato/                            ← Módulos VGA (IPs Altera + gerador de timing)
│   ├── vga_pll.v                            ← PLL: 50 MHz → 25 MHz (IP MegaWizard)
│   ├── vga_sync.v                           ← Gerador de temporização 640×480 @ 60 Hz
│   ├── rom_sprites.v                        ← ROM de sprites de nomes (IP altsyncram)
│   ├── rom_sprites.mif                      ← Bitmaps dos 20 nomes (163840 pixels)
│   ├── vga_pll.qip, vga_pll.ppf             ← Arquivos auxiliares do IP PLL
│   └── rom_sprites.qip                      ← Arquivo auxiliar do IP ROM
│
├── quartus_cnn/                             ← Projeto Quartus Prime
│   ├── cnn_inference.qpf                    ← Manifesto do projeto
│   ├── cnn_inference.qsf                    ← Pin assignments e configurações
│   ├── cnn_inference.sdc                    ← Constraints de timing
│   └── (db/, incremental_db/, output_files/) ← Pastas geradas pelo compilador
│
├── scripts/                                 ← Software do computador host
│   └── send_image_32x32.py                  ← Captura de vídeo + envio UART
│
└── explicacoes/                             ← Documentação técnica detalhada
    ├── modulos_verilog.md                   ← Descrição de cada módulo Verilog (principal)
    ├── vga_artefato.md                      ← Descrição dos módulos VGA
    ├── quartus_cnn.md                       ← Configuração de síntese e constraints
    ├── scripts.md                           ← Documentação do script Python
    └── maquinas_de_estado.md                ← Diagramas de todas as FSMs
```

---

## Como Executar

### 1. Síntese e Gravação na Placa

1. Abra o Quartus Prime (versão 18.1 ou superior).
2. Vá em **File → Open Project** e selecione `quartus_cnn/cnn_inference.qpf`.
3. Compile o projeto: **Processing → Start Compilation** (`Ctrl+L`).
4. Conecte a placa DE2-115 ao computador via USB Blaster.
5. Programe via JTAG: **Tools → Programmer → Start**, selecionando o arquivo `.sof` em `output_files/`.
6. Conecte um cabo serial (RS-232) à porta da DE2-115.
7. Conecte o cabo VGA da DE2-115 ao monitor.

### 2. Envio de Vídeo (Python)

Instale as dependências:
```bash
pip install opencv-python pyserial numpy
```

Inicie a captura da webcam com preview:
```bash
python scripts/send_image_32x32.py --port /dev/ttyUSB0 --preview
```

**Parâmetros:**
| Parâmetro | Descrição |
|-----------|-----------|
| `--port` | Porta serial (`/dev/ttyUSB0` no Linux, `COM3` no Windows) |
| `--baud` | Baud rate (padrão: 2000000) |
| `--preview` | Exibe janela de preview no PC |
| `--camera 0` | Índice da câmera |
| `--file imagem.jpg` | Envia imagem estática em vez da webcam |
| `--mock` | Testa sem a placa conectada |

No modo preview, pressione **ESPAÇO** para ativar/pausar a detecção de rostos e **Q** para encerrar.

---

## Controles na Placa

| Componente | Função |
|-----------|--------|
| **KEY[0]** | Reset do sistema. Apaga os framebuffers, reinicia a FSM e a tela VGA fica preta. |
| **LEDG[0]** | Acende quando a última classe predita é uma pessoa reconhecida (class_id 2–19). |
| **LEDR[0]** | Acende quando a última classe predita é "Desconhecido" (class_id 1). |

Os LEDs apagam automaticamente quando um novo frame começa a ser recebido.

---

## Documentação Técnica

A pasta `explicacoes/` contém a documentação detalhada do projeto, organizada por área:

| Documento | Conteúdo |
|-----------|----------|
| [modulos_verilog.md](explicacoes/modulos_verilog.md) | **Documento principal.** Hierarquia de instanciação e descrição de cada módulo Verilog. |
| [vga_artefato.md](explicacoes/vga_artefato.md) | PLL, gerador de temporização VGA e ROM de sprites. |
| [quartus_cnn.md](explicacoes/quartus_cnn.md) | Pin assignments, padrões elétricos e constraints de timing. |
| [scripts.md](explicacoes/scripts.md) | Script Python: pipeline de imagem, protocolo UART e detecção de rostos. |
| [maquinas_de_estado.md](explicacoes/maquinas_de_estado.md) | Diagramas e tabelas de transição de todas as FSMs do projeto. |