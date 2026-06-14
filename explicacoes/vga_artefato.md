# Artefatos VGA — Documentação Detalhada

Este documento descreve os arquivos da pasta `vga_artefato/`, responsáveis pela interface gráfica do sistema. Esses módulos são instanciados pelo [fpga_top_unified](../modulos_verilog/fpga_top_unified.v) e operam no domínio de clock de 25 MHz.

**Documento principal:** [modulos_verilog.md](modulos_verilog.md) — Descrição de todos os módulos de hardware e hierarquia de instanciação.

---

## Visão Geral

O subsistema VGA é composto por três blocos:

```
CLOCK_50 (50 MHz)
    │
    ▼
vga_pll ──► clk_25mhz (25 MHz)
                │
                ├──► vga_sync ──► hsync, vsync, pixel_x, pixel_y, video_on
                │
                └──► rom_sprites ──► pixel_do_sprite (1 bit por pixel)
```

O `fpga_top_unified` usa as saídas desses módulos para compor a imagem final enviada ao DAC ADV7123 da placa DE2-115.

---

## 1. `vga_pll.v` — PLL (Phase-Locked Loop)

**Finalidade:** Gerar o clock de 25 MHz necessário pelo padrão VGA 640×480 @ 60 Hz a partir do clock de 50 MHz da placa.

**Tipo:** IP gerado automaticamente pelo MegaWizard da Altera (Quartus). **Não editar manualmente** — o arquivo é regenerado pela ferramenta.

**Configuração:**

| Parâmetro | Valor |
|-----------|-------|
| Clock de entrada | 50 MHz (período de 20 ns) |
| Clock de saída (`c0`) | 25 MHz (divisão por 2) |
| Duty cycle | 50% |
| Família alvo | Cyclone IV E |

**Portas:**

| Porta | Direção | Descrição |
|-------|---------|-----------|
| `inclk0` | Entrada | Clock de referência (50 MHz) |
| `areset` | Entrada | Reset assíncrono |
| `c0` | Saída | Clock de 25 MHz gerado |
| `locked` | Saída | Ativo quando o PLL está estabilizado |

**Arquivos auxiliares:**
- `vga_pll.qip` — Arquivo de integração (Quartus IP) que declara o IP ao projeto.
- `vga_pll.ppf` — Arquivo de parâmetros do PLL (PPF = PLL Parameter File), usado pelo MegaWizard para regenerar o IP.

---

## 2. `vga_sync.v` — Gerador de Temporização VGA

**Finalidade:** Gerar os sinais de sincronização horizontal (hsync), vertical (vsync) e as coordenadas de pixel para o padrão VESA 640×480 @ 60 Hz.

**Temporização 640×480 @ 60 Hz:**

| Região | Horizontal | Vertical |
|--------|-----------|----------|
| Área ativa | 640 pixels | 480 linhas |
| Front porch | 16 pixels | 10 linhas |
| Pulso de sync | 96 pixels | 2 linhas |
| Back porch | 48 pixels | 33 linhas |
| **Total** | **800 pixels** | **525 linhas** |

**Funcionamento:**

O módulo possui dois contadores:
- `h_count` — Incrementado a cada borda de subida do clock de 25 MHz. Ao atingir 799 (H_TOTAL), volta a 0.
- `v_count` — Incrementado quando `h_count` atinge H_TOTAL (fim de uma linha). Ao atingir 524 (V_TOTAL), volta a 0.

**Geração dos sinais de sync:**

Os sinais de sync são gerados por lógica combinacional (assigns):
- `hsync` é nível baixo (ativo) durante a região de sync horizontal (pixels 656 a 751).
- `vsync` é nível baixo (ativo) durante a região de sync vertical (linhas 490 a 491).
- `video_on` é ativo apenas durante a área visível (h_count < 640 e v_count < 480).

**Saídas:**

| Porta | Descrição |
|-------|-----------|
| `hsync` | Sincronização horizontal (ativo em nível baixo) |
| `vsync` | Sincronização vertical (ativo em nível baixo) |
| `pixel_x[9:0]` | Coordenada X do pixel atual (= h_count) |
| `pixel_y[9:0]` | Coordenada Y do pixel atual (= v_count) |
| `video_on` | Ativo durante a área visível da tela |

---

## 3. `rom_sprites.v` — ROM de Sprites de Nomes

**Finalidade:** Armazenar as imagens bitmap (1 bit por pixel) dos nomes das 20 classes (Vazio + Desconhecido + 18 pessoas), que são renderizados como texto na parte inferior da tela VGA.

**Tipo:** IP gerado automaticamente pelo MegaWizard (ROM: 1-PORT usando `altsyncram`). **Não editar manualmente.**

**Configuração da ROM:**

| Parâmetro | Valor |
|-----------|-------|
| Largura de dados | 1 bit (preto ou branco por pixel) |
| Largura de endereço | 18 bits |
| Profundidade | 163840 palavras (20 classes × 256 × 32 pixels) |
| Arquivo de inicialização | `rom_sprites.mif` |
| Registro de saída | Desabilitado (UNREGISTERED) — dado disponível no mesmo ciclo |
| Família alvo | Cyclone IV E |

**Organização do espaço de endereçamento:**

Cada classe ocupa um bloco de 256×32 = 8192 bits (pixels). O endereço é calculado pelo `fpga_top_unified`:

```
endereço = {display_class_id, 13'd0} + (sprite_y × 256 + sprite_x)
```

Onde:
- `display_class_id[4:0]` — Classe a exibir (0–19)
- `sprite_y[4:0]` — Linha dentro do sprite (0–31)
- `sprite_x[7:0]` — Coluna dentro do sprite (0–255)

**Posicionamento na tela:**

O sprite é renderizado em uma região fixa de 256×32 pixels posicionada na parte inferior da tela:
- Início horizontal: pixel 192
- Início vertical: pixel 440
- Fim: pixel 447 (192+256) × pixel 471 (440+32)

**Portas:**

| Porta | Direção | Descrição |
|-------|---------|-----------|
| `address[17:0]` | Entrada | Endereço de leitura |
| `clock` | Entrada | Clock de leitura (25 MHz) |
| `q[0:0]` | Saída | Pixel do sprite (1=texto, 0=fundo) |

---

## 4. `rom_sprites.mif` — Dados dos Sprites

**Finalidade:** Arquivo de inicialização de memória contendo os bitmaps renderizados de todos os nomes das 20 classes.

**Tamanho:** Aproximadamente 1.8 MB (163840 linhas, uma por pixel).

**Formato:** Memory Initialization File (MIF) do Quartus. Cada entrada contém 1 bit (0 ou 1) indicando se aquele pixel do sprite é fundo (0) ou texto (1).

**Geração:** Este arquivo é gerado por um script externo de renderização de texto e não deve ser editado manualmente.

---

## 5. Arquivos `.qip` — Integração com o Projeto Quartus

- `vga_pll.qip` — Declara o IP `vga_pll` ao sistema de compilação do Quartus, incluindo os caminhos para os arquivos fonte e de simulação.
- `rom_sprites.qip` — Declara o IP `rom_sprites` ao sistema de compilação, incluindo o caminho para o `.mif`.

Esses arquivos são referenciados no [cnn_inference.qsf](../quartus_cnn/cnn_inference.qsf) e são necessários para que o Quartus encontre e compile os IPs corretamente.
