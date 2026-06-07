# Arquitetura e Pipeline — Tiny-CNN FPGA (19 Classes)

Documentação técnica completa da implementação em hardware da CNN de classificação biométrica facial sintetizada na FPGA DE2-115 (Cyclone IV EP4CE115F29C7).

---

## 1. Hierarquia de Módulos

```
fpga_top_de2115.v          ← Top-level sintetizável (pinos físicos)
└── cnn_top.v              ← Orquestrador FSM + barramentos
    ├── uart_rx.v           ← Receptor serial
    ├── framebuffer_32x32.v ← Buffer de imagem
    ├── line_buffer_32x32.v ← Janelas 3×3
    ├── convolucao_mac.v    ← Convolução (4 filtros, ReLU)
    ├── max_pooling_design.v← Max Pooling 2×2
    ├── flatten.v           ← Serialização
    ├── dense_900x19.v      ← Camada densa (19 classes)
    ├── argmax_19.v         ← Decisão final (argmax puro)
    └── weights_shared_rom.v← ROM de pesos (arquivo único)
```

O testbench `tb_cnn_layer_capture.v` instancia diretamente o `cnn_top` para acesso a todos os sinais internos.

---

## 2. Fluxo de Dados Completo

```
PC
 │ 1024 bytes via RS-232 (115200 baud, 8N1)
 ▼
UART_RXD (pino AB21 da DE2-115)
 │
 ▼ fpga_top_de2115.v (wrapper — repassa rx_pin)
 │
 ▼ uart_rx.v
 │ Decodifica: start bit → 8 bits LSB-first → stop bit
 │ Emite: uart_data[7:0] + uart_valid (pulso de 1 ciclo/byte)
 │
 ▼ Lógica UART em cnn_top.v
 │ uart_wr_en_comb = uart_valid & FSM_IDLE & !frame_ready
 │ Escreve uart_data → framebuffer[uart_wr_addr]
 │ uart_wr_addr: 0 → 1023; ao atingir 1023 → uart_frame_pending
 │
 ▼ framebuffer_32x32.v
 │ 1024 × 8 bits (mem_a + mem_b espelhados para leitura dupla)
 │ Quando endereço 1023 escrito → frame_ready = 1
 │
 ▼ FSM cnn_top (ST_IDLE → ST_READ)
 │ Detecta uart_start_pulse | start_system & frame_ready
 │ Emite frame_clear (consome frame_ready)
 │ Varre endereços 0..1023 → 1 pixel/ciclo → line_buffer
 │
 ▼ line_buffer_32x32.v
 │ Mantém 2 linhas de atraso (row1[32], row2[32])
 │ Janela 3×3 válida a partir do pixel (2,2)
 │ win[0..8] = [row2_prev | row2_cur | row1_prev | row1_cur | px_prev | px_cur]
 │
 ▼ convolucao_mac.v   (conv_4_filters_relu_window)
 │ 4 filtros 3×3 paralelos: MAC(win[9], pesos[9]) + bias
 │ Pesos INT8 (Q1.7) × pixels UINT8 → acumulador 20 bits
 │ ReLU + saturação em Q2.14 [0, 32767]
 │ Saída: 4 canais simultâneos, 1 resultado/ciclo de janela válida
 │
 ▼ max_pooling_design.v
 │ Redução 2×2 em cada canal (max entre 4 vizinhos)
 │ Usa buffer de 30 posições por canal para linha anterior
 │ Pool ocorre em posições (x ímpar, y ímpar)
 │ FIFO 32×64 bits (4 canais empacotados) → serialização por chan_idx
 │ Saída: 15×15×4 = 900 valores
 │
 ▼ flatten.v
 │ Identidade — repassa dados e conta até 900
 │ Ao elemento 899: pulso done (1 ciclo)
 │
 ▼ dense_900x19.v   (dense_900x19_scores)
 │ 19 acumuladores paralelos em Q3.21 (48 bits cada)
 │ A cada flat_valid: acc[i] += x_in × weight_rom[dense_addr][i]
 │ Após 900 entradas: acc[i] += bias[i] → shift_right 11 → Q6.10
 │ Saturação INT16 [-32768, +32767]
 │ Emite: scores[0..18][15:0] + valid_out + done
 │
 ▼ argmax_19.v
 │ Combinacional: compara scores[0..18], extrai max_idx e max_val
 │ Sem limiar estático — classe 0 é a classe de rejeição nativa (Desconhecido)
 │ class_id = max_idx
 │ unknown = 1 se max_idx == 0, senão 0
 │
 ▼ FSM cnn_top (ST_DONE)
 │ access_done = 1 (1 ciclo único)
 │
 ▼ fpga_top_de2115.v (registrador latch dos LEDs)
 │ LEDG[4:0] ← class_id   (0 = Desconhecido, 1–18 = membro)
 │ LEDG[5]   ← debug_frame_nonzero
 │ LEDG[6]   ← 1 (inferência concluída)
 │ LEDG[7]   ← unknown    (1 = classe 0 predita)
 ▼
LEDs da DE2-115 (resultado mantido até reset ou nova imagem)
```

---

## 3. Detalhamento dos Módulos

### `fpga_top_de2115.v` — Wrapper Top-Level

Único módulo visível para o sintetizador. Mapeia os pinos físicos:
- **CLOCK_50** → clock principal do sistema
- **KEY[0]** → reset global (active-low)
- **KEY[1]** → disparo manual (active-low, sincronizado, borda de descida)
- **UART_RXD** → recepção serial (MAX3232 na DE2-115)
- **LEDG[7:0]** → saída do resultado

Implementa um **registrador de captura (latch)** que preserva os valores de `class_id` e `unknown` nos LEDs durante o pulso de `access_done` e os mantém até o próximo evento relevante.

O botão KEY[1] passa por 3 flip-flops de sincronização para eliminar metaestabilidade e gerar um pulso limpo de 1 ciclo com detecção de borda de descida (`key1_prev & ~key1_sync_2`).

---

### `uart_rx.v` — Receptor UART

FSM de 4 estados (IDLE → START → DATA → STOP) que decodifica a transmissão serial 8N1.

Parâmetros:
- `CLK_FREQ = 50_000_000` Hz
- `BAUD_RATE = 115_200` bps
- `CYCLES_PER_BIT = 434` ciclos

No estado START, amostra no meio do período (ciclo 217) para rejeitar ruídos. Captura 8 bits LSB-first via shift register. Pulsa `data_valid` por 1 ciclo ao completar o stop bit.

> **No testbench**, o BAUD_RATE é sobrescrito via `defparam` para 1 Mbaud, reduzindo o tempo de simulação sem alterar a lógica.

---

### `framebuffer_32x32.v` — Buffer de Imagem

Duas RAMs espelhadas `mem_a[1024]` e `mem_b[1024]` de 8 bits, para leituras simultâneas independentes (pipeline CNN + futuro controlador VGA).

Comportamento especial:
- Escrita quando `wr_en = 1` (bytes da UART)
- `frame_ready = 1` quando `wr_addr == 1023` (imagem completa)
- `frame_clear` consome a flag (evita re-disparo da FSM)

Em síntese, o Quartus infere blocos M9K automáticos.

---

### `line_buffer_32x32.v` — Gerador de Janelas 3×3

Transforma o fluxo linear de pixels em janelas 3×3 contíguas.

Internamente mantém:
- `row1[32]` — penúltima linha processada
- `row2[32]` — antepenúltima linha

A cada pixel (`shift_en = 1`):
1. `row2[x] ← row1[x]` (desloca para trás)
2. `row1[x] ← pixel_in` (registra linha atual)
3. Coluna direita da janela atualizada: `win[2]←row2[x]`, `win[5]←row1[x]`, `win[8]←pixel_in`
4. Colunas esquerda/centro deslocam-se: `win[0]←win[1]`, `win[1]←win[2]`, etc.

`window_valid = 1` somente para `x ≥ 2 AND y ≥ 2` (janela completa disponível).

---

### `convolucao_mac.v` — Convolução 3×3 (4 Filtros)

Instância: `conv_4_filters_relu_window`

4 MACs paralelos, cada um calcula:

```
acc[f] = Σ(i=0..8) win[i] × weight[f][i]  +  bias[f]
```

Os 9 pesos de cada filtro são extraídos da ROM por assigns combinacionais estáticos (endereços fixos na ROM). Pesos em INT8 (Q1.7), pixels em UINT8 → produto em 16 bits → acumulador em 20 bits.

Após ReLU: `out_f = (acc < 0) ? 0 : saturate(acc, Q2.14)`

Saída: 4 valores de 16 bits simultâneos por janela válida.

---

### `max_pooling_design.v` — Max Pooling 2×2

Opera sobre os 4 canais convolucionais (saída 30×30×4 → 15×15×4).

Lógica:
- Buffer de 30 posições por canal armazena a linha anterior
- Pool ocorre quando `x[0] AND y[0]` (posições ímpares)
- Compara `max(atual, buffer[x-1], linha_ant[x], linha_ant[x-1])`
- FIFO de 32 entradas (64 bits = 4 canais empacotados) suaviza o fluxo
- Multiplexador `chan_idx` (0→3) serializa: 1 canal/ciclo

Saída: 900 valores de 16 bits em fluxo serial.

---

### `flatten.v` — Flatten

Identidade com contador de 900 elementos. Repassa `data_in → data_out` e pulsa `done` no elemento 899. Sinaliza a camada densa sobre o fim do vetor de entrada.

---

### `dense_900x19.v` — Camada Densa (19 Classes)

Instância: `dense_900x19_scores`

19 acumuladores paralelos de 48 bits (Q3.21) — evitam overflow ao somar 900 produtos.

A cada `flat_valid`:
```
acc[i] += signed(x_in) × signed(weight_rom[dense_addr * 19 + i])
```

Ao `flat_done`:
```
score[i] = saturate((acc[i] + bias[i]) >>> 7, INT16)
```

> Implementado **sem nenhum `for` loop** — todos os 19 acumuladores e 19 assigns de score são instâncias explícitas. Isso dá controle preciso ao sintetizador sobre o mapeamento lógico.

Emite `valid_out` e `done` simultâneos quando os 19 scores estão prontos.

---

### `argmax_19.v` — Decisão Final

Puramente combinacional. Compara os 19 scores e extrai o máximo por árvore de comparadores explícita (sem `for`).

**Lógica da Classe de Rejeição (Desconhecido)**:
Em vez de um limiar estático, a rede foi treinada com a classe 0 como classe explícita para "Desconhecido".

Saídas:
- `class_id[4:0]`: `max_idx` (0 para desconhecido, 1–18 para membro)
- `unknown`: 1 quando `class_id == 0`, senão 0
- `final_result[15:0]`: valor numérico do score máximo

Isso elimina a necessidade de fine-tuning manual de um limiar, transferindo a responsabilidade de incerteza para a rede neural.

---

### `weights_shared_rom.v` — ROM de Pesos

Array `mem[0:TOTAL_WORDS-1]` de 8 bits inicializado via `$readmemh("weights_all.hex")`.

Layout para 19 classes (17.159 bytes total):

| Faixa | Conteúdo | Tamanho |
|-------|----------|---------|
| `[0..35]` | Pesos conv (4 filtros × 9 coefs) | 36 bytes |
| `[36..39]` | Biases conv | 4 bytes |
| `[40..17139]` | Pesos densos (900 × 19 classes) | 17.100 bytes |
| `[17140..17158]` | Biases densos (19) | 19 bytes |

Os pesos convolucionais são extraídos por `assign` estático (endereços fixos). Os pesos densos são acessados por `dense_addr` (0..899) em runtime, com 19 assigns simultâneos por endereço.

> Para síntese no Quartus, o bloco `initial $readmemh` dentro de `ifdef SIMULATION` é ignorado e o arquivo `.mif` inicializa os M9K automaticamente via atributo Altera.

---

### `cnn_top.v` — Orquestrador

Módulo de integração que instancia todos os submódulos e implementa a FSM de 4 estados. Ver [`FSM.md`](FSM.md) para detalhes completos dos estados e transições.

Sinais-chave do barramento interno:

| Sinal | Direção | Descrição |
|-------|---------|-----------|
| `uart_wr_en_comb` | combinacional | Habilitação de escrita no FB via UART |
| `uart_wr_addr[9:0]` | registrado | Endereço atual de escrita UART |
| `uart_frame_pending` | registrado | Flag: 1024 bytes recebidos |
| `uart_start_pulse` | combinacional | Pulso de auto-start pós-recepção |
| `fb_rd_en` | registrado (FSM) | Habilita leitura do framebuffer |
| `fb_rd_addr[9:0]` | registrado (FSM) | Endereço de leitura |
| `dense_addr[9:0]` | registrado | Endereço dos pesos densos (0..899) |
| `dense_scores[18:0][15:0]` | wire | 19 scores da camada densa |
| `access_done` | registrado (FSM) | Pulso 1 ciclo = resultado pronto |

---

## 4. Formatos Numéricos de Ponto Fixo

| Formato | Bits | Sinal | Inteiro | Fracionário | Intervalo | Resolução | Uso |
|---------|------|-------|---------|-------------|-----------|-----------|-----|
| INT8 (Q1.7) | 8 | 1 | 0 | 7 | [-1.0, +0.992] | 1/128 | Pesos da ROM |
| Q2.14 | 16 | 1 | 1 | 14 | [-2.0, +1.999] | 1/16384 ≈ 6×10⁻⁵ | Ativações (Conv/Pool/Flat) |
| Q3.21 | 48 | 1 | 2 | 21 | — | — | Acumuladores internos da dense |
| Q6.10 | 16 | 1 | 5 | 10 | [-32.0, +31.999] | 1/1024 ≈ 1×10⁻³ | Scores (Camada Densa) |

A conversão Q3.21 → Q6.10 na camada Densa ocorre por deslocamento aritmético de 11 bits à direita (`>>> 11`), seguida de saturação INT16.

---

## 5. Testbench de Captura (`tb_cnn_layer_capture.v`)

O testbench instancia diretamente o `cnn_top` (não o `fpga_top_de2115`) para acessar os sinais internos das camadas.

**Fluxo:**
1. Reset (100 ns)
2. Envia 1024 bytes via UART simulada (task `send_uart_byte`)
3. Aguarda `access_done` (auto-start após recepção completa)
4. Salva saídas em arquivos `.txt`

**Captura por evento:**

| Evento | Arquivo | Formato |
|--------|---------|---------|
| `conv_valid` | `conv_out.txt` | `f0 f1 f2 f3` (Q2.14 com sinal) |
| `pool_valid` | `pool_out.txt` | `val` (Q2.14) |
| `flat_valid` | `flat_out.txt` | `val` (Q2.14) |
| `dense_valid` | `dense_out.txt` | `s0 s1 … s18` (19 scores Q6.10) |

**Uso:**
```tcl
do scripts/compile_project.do
do scripts/run_project.do inputs/teste2.txt comparacao/hw/
```

Ou com argumentos diretos ao simulador:
```
vsim -c -lib sim_work/work tb_cnn_layer_capture \
    +IMG=inputs/teste2.txt +OUTDIR=comparacao/hw/
```

---

## 6. Pipeline de Verificação SW vs. HW

```
inputs/foto.jpg
       │
       ├─── extract_sw_activations.py ──► comparacao/sw/{conv,pool,flat,dense}_out_sw.txt
       │
       └─── simulate.do (testbench) ────► comparacao/hw/{conv,pool,flat,dense}_out.txt
                                                    │
                                          compare_sw_hw.py
                                                    │
                                         comparacao/plots/
                                           01_hist_errors.png
                                           02_scatter_layers.png
                                           03_barplot_scores.png
                                           04_heatmap_conv.png
                                           report.txt
```

`extract_sw_activations.py` aplica automaticamente o Haar Cascade para detecção e recorte do rosto antes de processar com o modelo Keras, replicando o pré-processamento esperado pela FPGA.

---

## 7. Projeto Quartus

| Arquivo | Descrição |
|---------|-----------|
| `cnn_inference.qpf` | Arquivo de projeto |
| `cnn_inference.qsf` | Device (EP4CE115F29C7), pin assignments, lista de fontes |
| `cnn_inference.sdc` | Clock 50 MHz + false paths nos pinos assíncronos |

A síntese usa o arquivo `weights_all.mif` diretamente para inicializar os blocos M9K (não requer o `.hex`). O `.hex` é gerado apenas para simulação.

---

## 8. Notas de Implementação

- **Sem `for` loops nos módulos críticos** (`dense_900x18`, `argmax_threshold_18`, `weights_shared_rom`): todos os 18 caminhos de dados são instâncias explícitas, dando ao sintetizador controle preciso sobre o mapeamento lógico.
- **Arquivo único de pesos**: toda a ROM é um array linear. Não há divisão por camada ou por classe — o endereçamento é calculado por offsets parametrizados.
- **Threshold facilmente alterável**: basta modificar `THRESH_Q2_14` em `argmax_threshold_18.v`. O valor 15564 = `int(0.95 × 16384)`.
- **Inferência automática**: ao receber 1024 bytes pela UART, a FSM dispara automaticamente sem necessidade de botão.
