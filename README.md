# Artefato 3 — Pipeline UART → VGA (DE2-115)

Sistema sintetizável para a placa DE2-115 (Cyclone IV E) que recebe uma imagem 32×32 em escala de cinza via UART serial e a exibe em tempo real no monitor VGA com scaling 8× (256×256 pixels centrados em 640×480).

## Arquitetura

```
PC (Webcam + Python)  ──UART 115200 baud──▶  FPGA DE2-115  ──VGA 640×480──▶  Monitor
                          1024 bytes/frame     uart_rx → BRAM → VGA Controller
```

| Bloco | Módulo | Clock | Função |
|-------|--------|-------|--------|
| UART RX | `uart_rx.v` | 50 MHz | Recebe bytes seriais a 115200 baud |
| Framebuffer | Inline em `system_top.v` | 50/25 MHz | BRAM dual-port 1024×8 (escrita@50, leitura@25) |
| PLL | `vga_pll.v` | — | Converte 50 MHz → 25.175 MHz (pixel clock) |
| VGA Sync | `vga_sync.v` | 25 MHz | Gera timing 640×480 @ 60Hz (hsync, vsync) |
| Rendering | Inline em `system_top.v` | 25 MHz | Scaling 8×, compensação de latência |

## Estrutura de Arquivos

```
Lab---IA/
├── modulos_verilog/
│   └── uart_rx.v              ← Receptor UART (FSM 4 estados)
├── vga_artefato/
│   ├── system_top.v           ← Top-level (integra tudo)
│   ├── vga_sync.v             ← Gerador de timing VGA
│   ├── vga_pll.v              ← PLL Altera MegaFunction
│   ├── vga_pll_bb.v           ← Black-box do PLL (simulação)
│   ├── vga_pll_sim.v          ← Stub do PLL (Icarus Verilog)
│   ├── vga_pll.qip            ← IP do PLL
│   ├── top_vga.qpf            ← Projeto Quartus
│   ├── top_vga.qsf            ← Pin assignments DE2-115
│   └── tb_system_top.v        ← Testbench (UART simulado)
├── scripts/
│   └── send_image.py          ← Webcam → UART (Python)
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

# Webcam contínua
python scripts/send_image.py --port /dev/ttyUSB0

# Imagem estática
python scripts/send_image.py --file foto.png --port /dev/ttyUSB0
```

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
frame_mem[0]   = 0   (esperado: 0)
frame_mem[127] = 127 (esperado: 127)
frame_mem[255] = 255 (esperado: 255)
frame_mem[1023]= 255 (esperado: 255)
LEDG[0] (frame_received) = 1
--- Simulacao concluida com sucesso ---
```

## Controles na Placa

| Controle | Função |
|----------|--------|
| **KEY[0]** | Reset (pressionar para resetar) |
| **LEDG[0]** | Acende quando o 1° frame completo é recebido |
| **LEDR[9:0]** | Progresso do endereço de escrita UART |