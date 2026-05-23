# Artefato 3 — Pipeline UART → VGA (DE2-115)

Sistema sintetizável para a placa DE2-115 (Cyclone IV E) que recebe uma imagem de rosto detectado via Haarcascade em resolução 64×64 (escala de cinza) via UART serial a 230400 baud e a exibe em tempo real no monitor VGA com supersampling 6x centrado.

## Arquitetura

```
PC (Webcam + Haarcascade + Python)  ──UART 230400 baud──▶  FPGA DE2-115  ──VGA 640×480──▶  Monitor
                                      4.096 bytes/frame     uart_rx → BRAM → Supersampling → VGA
```

| Bloco | Módulo | Clock | Função |
|-------|--------|-------|--------|
| UART RX | `uart_rx.v` | 50 MHz | Recebe bytes seriais a 230400 baud |
| Framebuffer | Inline em `system_top.v` | 50/25 MHz | BRAM dual-port 4.096×8 (escrita@50, leitura@25) |
| PLL | `vga_pll.v` | — | Converte 50 MHz → 25.175 MHz (pixel clock) |
| VGA Sync | `vga_sync.v` | 25 MHz | Gera timing 640×480 @ 60Hz (hsync, vsync) |
| Supersampling | Inline em `system_top.v` | 25 MHz | Nearest-neighbor 64×64 → ampliado 6x centrado |

## Pipeline de Processamento

1. **Webcam** captura frame em tempo real
2. **Haarcascade** detecta rostos no frame
   - Se rosto encontrado: recorta a região do rosto, redimensiona para 64×64
   - Se nenhum rosto: mantém o último rosto válido detectado
3. **UART** envia os 4.096 bytes (64×64 pixels grayscale) a 230400 baud (~0,178s = ~5 FPS)
4. **FPGA** armazena em BRAM e exibe via VGA com supersampling 6x para 384x384 centrado em 640x480

## Estrutura de Arquivos

```
Lab---IA/
├── haarcascade_frontalface_default.xml  ← Classificador de rostos
├── modulos_verilog/
│   └── uart_rx.v              ← Receptor UART (FSM 4 estados)
├── vga_artefato/
│   ├── system_top.v           ← Top-level (integra tudo + supersampling)
│   ├── vga_sync.v             ← Gerador de timing VGA
│   ├── vga_pll.v              ← PLL Altera MegaFunction
│   ├── vga_pll_bb.v           ← Black-box do PLL (simulação)
│   ├── vga_pll_sim.v          ← Stub do PLL (Icarus Verilog)
│   ├── vga_pll.qip            ← IP do PLL
│   ├── top_vga.qpf            ← Projeto Quartus
│   ├── top_vga.qsf            ← Pin assignments DE2-115
│   └── tb_system_top.v        ← Testbench (UART simulado)
├── scripts/
│   └── send_image.py          ← Webcam + Haarcascade → UART (Python)
└── README.md
```

## Síntese no Quartus

1. Abra `vga_artefato/top_vga.qpf` no Quartus Prime 18.1+
2. Compile: **Processing → Start Compilation** (Ctrl+L)
3. Programe: **Tools → Programmer** → selecione o USB-Blaster → **Start**

O top-level entity é `system_top` e o dispositivo alvo é `EP4CE115F29C7`.

## Envio de Imagem (Python)

```bash
pip install opencv-python pyserial numpy

# Webcam contínua com detecção de rosto (padrão)
python scripts/send_image.py --port /dev/ttyUSB0

# Sem detecção de rosto (envia frame completo)
python scripts/send_image.py --port /dev/ttyUSB0 --no-haar

# Imagem estática
python scripts/send_image.py --file foto.png --port /dev/ttyUSB0

# Teste sem placa (modo simulação)
python scripts/send_image.py --mock
```

### Parâmetros

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--port` | `/dev/ttyUSB0` | Porta serial |
| `--baud` | `230400` | Baud rate (máximo seguro para MAX3232) |
| `--camera` | `0` | Índice da câmera |
| `--file` | — | Enviar imagem estática |
| `--once` | — | Enviar apenas um frame |
| `--mock` | — | Simulação sem placa |
| `--no-haar` | — | Desabilitar detecção de rosto |

## Simulação (Icarus Verilog)

```bash
iverilog -g2005 -o sim_out \
  vga_artefato/tb_system_top.v \
  vga_artefato/system_top.v \
  vga_artefato/vga_sync.v \
  vga_artefato/vga_pll_sim.v \
  modulos_verilog/uart_rx.v

vvp sim_out
```

Resultado esperado:
```
  frame_mem[0]      = 0 (esperado: 0)
  frame_mem[63]     = 63 (esperado: 63)
  frame_mem[255]    = 255 (esperado: 255)
  frame_mem[4095]   = 255 (esperado: 255)
  LEDG (frame_received) = 1
--- Simulacao concluida com sucesso ---
```

## Controles na Placa

| Controle | Função |
|----------|--------|
| **KEY[0]** | Reset (pressionar para resetar) |
| **LEDG[0]** | Acende quando o 1° frame completo é recebido |
| **LEDR[9:0]** | Progresso do endereço de escrita UART |