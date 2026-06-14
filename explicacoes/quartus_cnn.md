# Configuração de Síntese (Quartus Prime) — Documentação Detalhada

Este documento descreve os arquivos da pasta `quartus_cnn/`, responsáveis pela configuração de síntese, roteamento físico e constraints de timing do projeto na FPGA DE2-115.

**Documento principal:** [modulos_verilog.md](modulos_verilog.md) — Descrição de todos os módulos de hardware.

---

## Visão Geral

A pasta `quartus_cnn/` contém o projeto Quartus Prime que compila todos os módulos Verilog em um bitstream gravável no chip EP4CE115F29C7. Os arquivos editáveis são apenas 3:

| Arquivo | Função |
|---------|--------|
| `cnn_inference.qpf` | Manifesto do projeto |
| `cnn_inference.qsf` | Configurações de compilação e pin assignments |
| `cnn_inference.sdc` | Constraints de timing |

As subpastas `db/`, `incremental_db/`, `output_files/` e `simulation/` são geradas automaticamente pelo compilador e não devem ser versionadas.

---

## 1. `cnn_inference.qpf` — Projeto Quartus

**Finalidade:** Arquivo manifesto mínimo do Quartus Prime. Define o nome do projeto e a revisão ativa.

**Conteúdo:** Apenas uma linha declarando a revisão padrão. Este arquivo é gerenciado pela GUI do Quartus e não requer edição manual.

---

## 2. `cnn_inference.qsf` — Configurações de Compilação

**Finalidade:** Arquivo central de configuração do projeto. Define o chip alvo, o mapeamento de pinos, os padrões elétricos, e a lista de arquivos fonte.

### 2.1 Configurações Globais

| Configuração | Valor | Descrição |
|-------------|-------|-----------|
| `FAMILY` | Cyclone IV E | Família do FPGA |
| `DEVICE` | EP4CE115F29C7 | Chip específico da placa DE2-115 |
| `TOP_LEVEL_ENTITY` | fpga_top_unified | Módulo de nível mais alto |
| `NUM_PARALLEL_PROCESSORS` | ALL | Usar todos os núcleos do PC na compilação |
| `PROJECT_OUTPUT_DIRECTORY` | output_files | Pasta para arquivos gerados (.sof, .pof, etc.) |

### 2.2 Pin Assignments (Mapeamento de Pinos)

Cada sinal do módulo top-level é atribuído a um pino físico específico do chip, conforme o esquemático da placa DE2-115:

**Clock e controle:**

| Sinal | Pino | Descrição |
|-------|------|-----------|
| `CLOCK_50` | PIN_Y2 | Oscilador de 50 MHz |
| `KEY` | PIN_M23 | Botão KEY[0] (reset) |
| `UART_RXD` | PIN_AB21 | RS-232 RXD via MAX3232 |

**VGA (DAC ADV7123):**

| Sinal | Pinos | Banco |
|-------|-------|-------|
| `VGA_HS` | PIN_G13 | Bank 4 (3.3V) |
| `VGA_VS` | PIN_C13 | Bank 4 (3.3V) |
| `VGA_CLK` | PIN_A12 | Bank 4 (3.3V) |
| `VGA_BLANK_N` | PIN_F11 | Bank 4 (3.3V) |
| `VGA_SYNC_N` | PIN_C10 | Bank 4 (3.3V) |
| `VGA_R[7:0]` | PIN_E12..H10 | Bank 4 (3.3V) |
| `VGA_G[7:0]` | PIN_G8..C9 | Bank 4 (3.3V) |
| `VGA_B[7:0]` | PIN_B10..D12 | Bank 4 (3.3V) |

**LEDs verdes (LEDG[8:0]):**

| Sinal | Pino |
|-------|------|
| `LEDG[0]` | PIN_E21 |
| `LEDG[1]` | PIN_E22 |
| `LEDG[2]` | PIN_E25 |
| `LEDG[3]` | PIN_E24 |
| `LEDG[4]` | PIN_H21 |
| `LEDG[5]` | PIN_G20 |
| `LEDG[6]` | PIN_G22 |
| `LEDG[7]` | PIN_G21 |
| `LEDG[8]` | PIN_F17 |

**LEDs vermelhos (LEDR[17:0]):**

| Sinal | Pino |
|-------|------|
| `LEDR[0]` | PIN_G19 |
| `LEDR[1]` | PIN_F19 |
| `LEDR[2..17]` | PIN_E19..H15 |

### 2.3 Padrões Elétricos (I/O Standards)

| Componente | Padrão | Banco |
|-----------|--------|-------|
| Clock | 3.3-V LVTTL | Bank 2 (3.3V) |
| KEY | 2.5 V | Bank 6 (2.5V) |
| UART | 3.3-V LVTTL | Bank 4 (3.3V) |
| VGA | 3.3-V LVTTL | Bank 4 (3.3V) |
| LEDs | 2.5 V | Bank 7 (2.5V) |

### 2.4 Drive Strength e Slew Rate

Todos os pinos de saída possuem:
- **Drive Strength:** 8 mA — valor suficiente para LEDs e o DAC VGA, evitando warnings 15714 do compilador.
- **Slew Rate:** 2 (rápido) — garante transições nítidas nos sinais VGA.

### 2.5 Lista de Arquivos Fonte

O QSF referencia todos os arquivos Verilog e IPs do projeto usando caminhos relativos:

**Pipeline CNN (módulos Verilog):**
- `../modulos_verilog/fpga_top_unified.v`
- `../modulos_verilog/cnn_top.v`
- `../modulos_verilog/uart_rx.v`
- `../modulos_verilog/framebuffer_32x32.v`, `framebuffer_128x128.v`
- `../modulos_verilog/line_buffer_32x32.v`
- `../modulos_verilog/convolucao_mac.v`
- `../modulos_verilog/max_pooling_design.v`
- `../modulos_verilog/flatten.v`
- `../modulos_verilog/dense_900x19.v`
- `../modulos_verilog/argmax_19.v`
- `../modulos_verilog/weights_shared_rom.v`
- `../modulos_verilog/weights_all.mif`

**Pipeline VGA (IPs Altera):**
- `../vga_artefato/vga_sync.v`
- `../vga_artefato/vga_pll.qip`
- `../vga_artefato/rom_sprites.qip`

**Search path:** `../vga_artefato` — permite que o Quartus localize o `rom_sprites.mif` automaticamente.

---

## 3. `cnn_inference.sdc` — Constraints de Timing

**Finalidade:** Define as restrições de timing que o compilador Quartus deve respeitar ao rotear os sinais no chip.

**Conteúdo:**

| Constraint | Comando | Descrição |
|-----------|---------|-----------|
| Clock principal | `create_clock -name CLOCK_50 -period 20.000` | Define 50 MHz (período de 20 ns) |
| Clocks derivados | `derive_pll_clocks` | Instrui o Quartus a detectar automaticamente o clock de 25 MHz gerado pelo PLL |
| Incerteza de clock | `derive_clock_uncertainty` | Calcula automaticamente o jitter |
| UART (assíncrono) | `set_false_path -from [get_ports UART_RXD]` | O pino UART não possui relação de timing com o clock interno |
| KEY (assíncrono) | `set_false_path -from [get_ports KEY]` | Botão é entrada assíncrona |
| LEDs (relaxado) | `set_false_path -to [get_ports LEDG*]` e `LEDR*` | LEDs não possuem requisitos de timing |
| VGA_SYNC_N | `set_false_path -to [get_ports VGA_SYNC_N]` | Sinal estático (sempre 0) |

**False paths:** Os `set_false_path` informam ao Fitter do Quartus que certas trilhas não precisam atender requisitos de timing, liberando o otimizador para focar nos caminhos críticos (como o pipeline CNN a 50 MHz e o VGA a 25 MHz).

---

## 4. Subpastas Geradas

| Pasta | Conteúdo |
|-------|----------|
| `db/` | Banco de dados interno do Quartus (cache de compilação) |
| `incremental_db/` | Dados para compilação incremental (acelera recompilações parciais) |
| `output_files/` | Arquivos de saída: `.sof` (bitstream JTAG), `.pof` (flash), relatórios de compilação |
| `simulation/` | Arquivos de simulação gerados para Questa/ModelSim |

Todas essas pastas são geradas automaticamente e estão listadas no `.gitignore` do projeto.
