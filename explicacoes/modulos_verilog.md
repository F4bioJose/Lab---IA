# Módulos Verilog — Documentação Detalhada

Este é o **documento principal** da documentação técnica do projeto. Ele descreve todos os arquivos da pasta `modulos_verilog/`, que contém a lógica de hardware sintetizável do sistema.

Os módulos desta pasta são instanciados hierarquicamente: o **`fpga_top_unified`** é a entidade de nível mais alto (top-level) e instancia o **`cnn_top`**, que por sua vez instancia todos os módulos do pipeline da rede neural. O `fpga_top_unified` também instancia diretamente os módulos de vídeo VGA (documentados em [vga_artefato.md](vga_artefato.md)).

**Documentos complementares:**
- [vga_artefato.md](vga_artefato.md) — Módulos VGA (PLL, gerador de timing, ROM de sprites)
- [quartus_cnn.md](quartus_cnn.md) — Configuração de síntese (pin assignments, constraints de timing)
- [scripts.md](scripts.md) — Script Python de captura e envio de vídeo
- [maquinas_de_estado.md](maquinas_de_estado.md) — Diagramas e tabelas de transição de todas as FSMs

---

## Hierarquia de Instanciação

O diagrama abaixo mostra como os módulos se conectam. As setas indicam instanciação direta (quem cria quem).

```
fpga_top_unified (top-level)
├── vga_pll                   ← PLL: 50 MHz → 25 MHz (documentado em vga_artefato.md)
├── altddio_out               ← DDR output para VGA_CLK (IP Altera nativo)
├── vga_sync                  ← Gerador de temporização VGA (documentado em vga_artefato.md)
├── rom_sprites               ← ROM de sprites de nomes (documentado em vga_artefato.md)
├── framebuffer_128x128       ← BRAM para frames de vídeo (128×128)
└── cnn_top                   ← Orquestrador do pipeline CNN
    ├── uart_rx               ← Receptor serial UART
    ├── framebuffer_32x32     ← BRAM dual-port para frame de rosto (32×32)
    ├── line_buffer_32x32     ← Buffer de linhas para janelamento 3×3
    ├── conv_4_filters_relu   ← Convolução: 4 filtros 3×3 + ReLU
    ├── max_pooling_design    ← Max pooling 2×2
    ├── flatten               ← Serialização dos mapas de características
    ├── weights_shared_rom    ← ROM de pesos com bootloader
    ├── dense_900x19_scores   ← Camada fully-connected (900→19)
    └── argmax_19             ← Seleção da classe com maior score
```

---

## 1. `fpga_top_unified.v` — Top-Level do Sistema

**Finalidade:** Entidade de nível mais alto do projeto. Une o pipeline da CNN com o controlador de vídeo VGA em um único módulo sintetizável. Toda a lógica do FPGA parte daqui.

**Entradas e saídas físicas:**

| Sinal | Direção | Descrição |
|-------|---------|-----------|
| `CLOCK_50` | Entrada | Clock principal de 50 MHz da placa DE2-115 |
| `KEY` | Entrada | Botão de reset (active-low, com Schmitt trigger) |
| `UART_RXD` | Entrada | Pino serial RX (RS-232 via MAX3232) |
| `VGA_CLK`, `VGA_HS`, `VGA_VS`, `VGA_BLANK_N`, `VGA_SYNC_N` | Saída | Sinais de controle do DAC VGA (ADV7123) |
| `VGA_R[7:0]`, `VGA_G[7:0]`, `VGA_B[7:0]` | Saída | Canais de cor VGA (8 bits cada) |
| `LEDG[8:0]` | Saída | LEDs verdes da placa |
| `LEDR[17:0]` | Saída | LEDs vermelhos da placa |

**Responsabilidades:**

1. **Geração de clocks** — Instancia o PLL (`vga_pll`) para gerar o clock de 25 MHz necessário pelo padrão VGA 640×480 @ 60 Hz.

2. **Mapeamento de coordenadas VGA** — Converte as coordenadas de pixel (0–639, 0–479) em endereços de framebuffer. Existem dois modos de exibição:
   - **Modo rosto (32×32):** Imagem ampliada 12× (384×384 pixels na tela). A divisão por 12 é feita por multiplicação recíproca: `coordenada × 5462 >> 16`.
   - **Modo vídeo (128×128):** Imagem ampliada 3× (384×384 pixels na tela). Divisão por 3 via `coordenada × 21845 >> 16`.

3. **Renderização VGA** — Compõe a imagem final que aparece no monitor, combinando três camadas com a seguinte prioridade:
   - Imagem do framebuffer (escala de cinza)
   - Sprite de texto com o nome da classe predita (verde se reconhecido, vermelho se desconhecido)
   - Fundo preto

4. **Orquestração CNN ↔ VGA** — Mantém o registrador `display_class_id` que controla qual nome é exibido no sprite:
   - No reset ou ao receber novo frame: `display_class_id = 0` (exibe "Vazio")
   - Quando a CNN conclui a inferência: `display_class_id` recebe o `class_id` produzido

5. **Controle dos LEDs** — Acende `LEDG[0]` quando a classe é reconhecida (aprovada) ou `LEDR[0]` quando é desconhecida. Ambos apagam ao iniciar um novo frame.

6. **Compensação de latência** — Os sinais de controle (hsync, vsync, video_on, in_image, in_sprite_window) são atrasados em 1 ciclo de clock para alinhar com a latência de leitura das memórias BRAM e da ROM de sprites.

**FSM detalhada:** Ver [maquinas_de_estado.md](maquinas_de_estado.md), seção "FSM do fpga_top_unified".

---

## 2. `cnn_top.v` — Orquestrador do Pipeline CNN

**Finalidade:** Módulo que coordena todas as camadas da rede neural. Opera a 50 MHz e controla o fluxo de dados desde a recepção UART até a geração do `class_id`.

**Protocolo UART com byte de controle:**

O primeiro byte recebido via UART define o tipo de frame:
- `0xFF` → Frame de rosto (32×32 = 1024 bytes). Armazenado no `framebuffer_32x32` e processado pela CNN.
- Qualquer outro valor → Frame de vídeo (128×128 = 16384 bytes). Armazenado no `framebuffer_128x128` apenas para exibição VGA, sem inferência.

**Sincronização UART:**

O sinal `rx_pin` passa por dois flip-flops de sincronização (`rx_sync_1`, `rx_sync_2`) para evitar metaestabilidade ao cruzar do domínio assíncrono da UART para o clock de 50 MHz.

**FSM de 6 estados:**

| Estado | Descrição |
|--------|-----------|
| `ST_IDLE` | Aguarda byte de controle ou frame pendente |
| `ST_RX_FACE` | Recebe 1024 bytes do frame de rosto |
| `ST_RX_VIDEO` | Recebe 16384 bytes do frame de vídeo |
| `ST_READ` | Lê o framebuffer 32×32 e alimenta o pipeline da CNN |
| `ST_WAIT` | Aguarda a conclusão da camada densa (`dense_done`) |
| `ST_DONE` | Pulsa `access_done` por 1 ciclo e retorna ao IDLE |

**Sinais de saída para o top-level:**

| Sinal | Descrição |
|-------|-----------|
| `class_id[4:0]` | Classe predita (0=Vazio, 1=Desconhecido, 2–19=pessoa) |
| `unknown` | Ativo quando `class_id == 1` |
| `access_done` | Pulso de 1 ciclo indicando fim da inferência |
| `frame_ready` | Ativo quando o framebuffer 32×32 está completo |
| `frame_mode` | 0=vídeo (128×128), 1=rosto (32×32) |

**Endereçamento da ROM de pesos:**

O registrador `dense_addr` é incrementado a cada dado válido do flatten e serve como endereço de leitura síncrona dos 19 blocos M9K de pesos dentro do `weights_shared_rom`.

**FSM detalhada:** Ver [maquinas_de_estado.md](maquinas_de_estado.md), seção "FSM do cnn_top".

---

## 3. `uart_rx.v` — Receptor UART

**Finalidade:** Decodifica a comunicação serial assíncrona (UART) em bytes de 8 bits. Configurado para operar a 2 Mbaud com clock de 50 MHz.

**Parâmetros:**

| Parâmetro | Valor padrão | Descrição |
|-----------|-------------|-----------|
| `CLK_FREQ` | 50.000.000 | Frequência do clock de entrada (Hz) |
| `BAUD_RATE` | 2.000.000 | Velocidade da comunicação serial (baud) |

O valor `CYCLES_PER_BIT = CLK_FREQ / BAUD_RATE` define quantos ciclos de clock correspondem a 1 bit serial. Para 2 Mbaud: 25 ciclos por bit.

**Funcionamento:**

1. **Detecção do Start Bit:** A linha serial fica em nível alto quando ociosa. Quando detecta nível baixo, transiciona para o estado START.
2. **Confirmação no centro do bit:** Aguarda metade do tempo de um bit (`CYCLES_PER_BIT / 2`) e verifica se a linha ainda está em nível baixo. Se estiver, o start bit é confirmado; caso contrário, era ruído e retorna ao IDLE.
3. **Leitura dos 8 bits de dados:** Para cada bit, aguarda `CYCLES_PER_BIT` ciclos (para amostrar no centro do bit seguinte) e armazena o valor em um shift register, do bit 0 (LSB) ao bit 7 (MSB).
4. **Stop Bit:** Aguarda o término do stop bit e transfere o byte completo para `data_out`, pulsando `data_valid` por 1 ciclo.

**Saídas:**

| Sinal | Descrição |
|-------|-----------|
| `data_out[7:0]` | Byte decodificado |
| `data_valid` | Pulso de 1 ciclo indicando que `data_out` contém um byte válido |

**FSM detalhada:** Ver [maquinas_de_estado.md](maquinas_de_estado.md), seção "FSM do uart_rx".

---

## 4. `framebuffer_32x32.v` — Memória do Frame de Rosto

**Finalidade:** Armazena a imagem de rosto de 32×32 pixels (1024 bytes em escala de cinza). Fornece duas portas de leitura independentes: uma para a CNN e outra para o VGA.

**Arquitetura de memória:**

Como o Cyclone IV não possui blocos RAM com 3 portas nativas (1 escrita + 2 leituras simultâneas), o módulo utiliza **duas memórias espelhadas**:
- `mem_a` — Leitura dedicada à CNN (50 MHz)
- `mem_b` — Leitura dedicada ao VGA (25 MHz, via clock crossing no top-level)

Ambas recebem os mesmos dados de escrita simultaneamente, mantendo conteúdo idêntico.

**Lógica de clear por hardware:**

Ao receber reset, um contador (`clear_addr`) percorre todos os 1024 endereços escrevendo `0x00` em ambas as memórias. Isso garante que a tela fique preta imediatamente após o reset, sem depender de software.

**Sinalização de frame pronto:**

O sinal `frame_ready` é ativado quando o endereço de escrita atinge 1023 (último pixel). É desativado pelo sinal `frame_clear` emitido pelo `cnn_top` ao iniciar uma nova inferência.

**Inferência de BRAM:**

O bloco de escrita é implementado sem reset (`always @(posedge clk)` sem condição de `rst`) para que o compilador Quartus infira corretamente blocos M9K. Se houvesse reset no bloco de memória, o Quartus usaria registradores lógicos em vez de BRAM, esgotando os recursos do chip.

---

## 5. `framebuffer_128x128.v` — Memória do Frame de Vídeo

**Finalidade:** Armazena a imagem de vídeo de 128×128 pixels (16384 bytes em escala de cinza). Possui apenas uma porta de leitura (VGA), pois frames de vídeo não passam pela inferência da CNN.

**Diferenças em relação ao `framebuffer_32x32`:**
- Apenas uma memória (sem espelhamento), pois não há leitura simultânea pela CNN.
- Endereçamento de 14 bits (0–16383) em vez de 10 bits.
- O `frame_ready` é controlado apenas pelo endereço de escrita, sem sinal `frame_clear` externo.

A lógica de clear por hardware funciona da mesma forma: um contador percorre os 16384 endereços no reset.

---

## 6. `line_buffer_32x32.v` — Buffer de Linhas para Janelamento 3×3

**Finalidade:** Converte o fluxo sequencial de pixels (1 pixel por ciclo) em janelas 3×3 simultâneas, necessárias para a operação de convolução.

**Funcionamento:**

O módulo armazena as duas últimas linhas completas da imagem em memórias M9K (`row1` e `row2`, cada uma com 32 posições). Quando um novo pixel chega:

1. O pixel atual é armazenado em `row1[x]`.
2. O valor anterior de `row1[x]` é empurrado para `row2[x]`.
3. A coluna mais à direita da janela 3×3 é preenchida com: `row2[x]` (topo), `row1[x]` (meio), `pixel_in` (base).
4. As colunas anteriores da janela são deslocadas como um shift register.

**Validade da janela:**

Uma janela 3×3 completa só existe a partir da posição (2, 2) da imagem, pois as duas primeiras linhas e colunas ainda não possuem vizinhos suficientes. O sinal `window_valid` só é ativado quando `in_x >= 2` e `in_y >= 2`, resultando em uma grade de saída de 30×30 pixels válidos.

**Saída:**

| Sinal | Descrição |
|-------|-----------|
| `win[0:8]` | 9 pixels da janela 3×3 (ordem: linha a linha, da esquerda para a direita, de cima para baixo) |
| `window_valid` | Ativo quando a janela contém 9 pixels válidos |

---

## 7. `convolucao_mac.v` — Convolução com 4 Filtros Paralelos

**Nome do módulo no código:** `conv_4_filters_relu_window`

**Finalidade:** Executa a operação de convolução 2D sobre a janela 3×3 de entrada usando 4 filtros independentes, aplica a função de ativação ReLU e satura o resultado para evitar overflow.

**Aritmética:**

Para cada filtro, o cálculo MAC (Multiply-Accumulate) é:

```
mac = Σ(pixel[i] × peso[i]) + bias    para i = 0..8
```

- **Pixels de entrada:** 8 bits sem sinal (0–127 em Q1.7), promovidos para signed na operação
- **Pesos:** 8 bits com sinal (INT8, formato Q1.7)
- **Bias:** 8 bits com sinal, estendido para o formato do acumulador com deslocamento de 7 bits (`{sign_extend, bias, 7'b0}`)
- **Resultado MAC:** 20 bits com sinal

**ReLU e saturação:**

A função `relu_sat` aplicada a cada resultado MAC:
- Se negativo → saída = 0 (ReLU)
- Se maior que 32767 → saída = 32767 (saturação para evitar overflow em 16 bits)
- Caso contrário → saída = bits [15:0] do MAC

**Saídas:** 4 valores de 16 bits com sinal (`out_f0` a `out_f3`), cada um correspondente a um filtro.

---

## 8. `max_pooling_design.v` — Max Pooling 2×2

**Finalidade:** Reduz a dimensionalidade espacial pela metade, selecionando o valor máximo de cada vizinhança 2×2. Recebe a saída dos 4 filtros da convolução (grid de 30×30) e produz mapas de 15×15.

**Funcionamento:**

O módulo armazena uma linha inteira de valores anteriores para cada filtro (`row1_f0` a `row1_f3`, 30 posições cada). Quando as coordenadas (x, y) são ambas ímpares e maiores que 0, os 4 valores do quadrante 2×2 estão disponíveis e o máximo é calculado pela função combinacional `pooling(a, b, c, d)`.

**Serialização dos 4 canais via FIFO:**

Como o max pooling produz 4 resultados simultâneos (um por filtro) mas o módulo flatten subsequente consome 1 valor por ciclo, uma FIFO interna de 32 posições armazena os resultados empacotados (64 bits = 4×16 bits) e os serializa em 4 ciclos consecutivos (canal 0, 1, 2, 3).

**Dimensões:**

| Parâmetro | Valor |
|-----------|-------|
| Entrada | 30×30 por filtro (4 filtros) |
| Saída | 15×15 por filtro (4 filtros) |
| Total de saídas | 15 × 15 × 4 = 900 valores |

---

## 9. `flatten.v` — Serialização dos Mapas de Características

**Finalidade:** Conta os elementos do fluxo de dados proveniente do max pooling e sinaliza quando todos os 900 valores (15×15×4) foram recebidos.

**Funcionamento:**

O módulo é essencialmente um passthrough com contador: ele repassa `data_in` diretamente para `data_out`, mantendo o sinal `valid_out` sincronizado com `valid_in`. A contribuição principal é o sinal `done`, que pulsa quando o contador atinge `TOTAL_SIZE - 1` (899), indicando que o vetor completo de 900 elementos está pronto para a camada densa.

**Parâmetros:**

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `DATA_WIDTH` | 16 | Largura de cada elemento (Q2.14) |
| `CHANNELS` | 4 | Número de filtros/canais |
| `HEIGHT` | 15 | Altura do mapa após pooling |
| `WIDTH` | 15 | Largura do mapa após pooling |
| `TOTAL_SIZE` | 900 | 4 × 15 × 15 |

---

## 10. `dense_900x19.v` — Camada Fully-Connected

**Nome do módulo no código:** `dense_900x19_scores`

**Finalidade:** Implementa a camada densa (fully-connected) da rede neural. Recebe os 900 valores do flatten e calcula os scores brutos (logits) para cada uma das 19 classes.

**Aritmética detalhada:**

A operação para cada classe `c` é:

```
score[c] = saturar( (Σ(x[i] × w[c][i]) + bias[c]) >> 11 )
```

As etapas de precisão são:

| Etapa | Formato | Bits |
|-------|---------|------|
| Entrada `x_in` | Q2.14 (16 bits signed) | 16 |
| Peso `w_in` | INT8 (8 bits signed) | 8 |
| Produto parcial | Q3.21 (24 bits signed) | 24 |
| Acumulador | Q3.21 estendido (48 bits) | 48 |
| Bias | INT8 escalado para Q3.21 (shift de 14 bits) | 48 |
| Saída `scores` | Q6.10 (16 bits signed, shift aritmético de 11 bits) | 16 |

**Funcionamento:**

1. **Acumulação:** Para cada um dos 900 valores de entrada, o produto `x_in × w_in[c]` é calculado para todas as 19 classes simultaneamente e somado ao acumulador de 48 bits da classe correspondente.
2. **Finalização:** Quando o contador `sample_count` atinge 899 (último elemento), o bias é adicionado, o resultado é deslocado 11 bits para a direita (conversão Q3.21 → Q6.10) e saturado no intervalo [-32768, +32767].
3. **Limpeza:** Os acumuladores são zerados para o próximo frame.

**Paralelismo:** As 19 multiplicações `x_in × w_in[c]` ocorrem em paralelo (19 multiplicadores combinacionais), mas a acumulação é sequencial (900 ciclos).

---

## 11. `argmax_19.v` — Seleção da Classe Predita

**Finalidade:** Recebe os 19 scores produzidos pela camada densa e identifica qual classe possui a maior pontuação (argmax). Também sinaliza se a classe predita é "Desconhecido".

**Funcionamento:**

Quando `valid_in` é pulsado (indicando que os 19 scores estão prontos), o módulo:

1. Inicializa `max_val = scores[0]` e `max_idx = 0`.
2. Compara sequencialmente com cada score de 1 a 18. Se `scores[i] > max_val`, atualiza `max_val` e `max_idx`.
3. A saída `class_id` recebe `max_idx + 1` (offset de +1 para reservar o valor 0 como "Vazio").

**Mapeamento de classes (offset +1):**

A rede neural possui 19 neurônios de saída (índices 0 a 18). O índice 0 foi treinado como "Desconhecido". Na saída do argmax, soma-se 1 para que:
- `class_id = 0` → Vazio (nenhuma inferência realizada)
- `class_id = 1` → Desconhecido (a rede predisse a classe 0)
- `class_id = 2..19` → Pessoa identificada

**Decisão sem threshold:**

Não há limiar de confiança. A decisão é puramente pelo argmax: a classe com maior score vence, independentemente da magnitude absoluta do score.

**Saídas:**

| Sinal | Descrição |
|-------|-----------|
| `class_id[4:0]` | Classe predita (0=Vazio, 1=Desconhecido, 2–19=pessoa) |
| `unknown` | 1 quando a rede prediz a classe 0 (Desconhecido) |
| `max_score[15:0]` | Valor do maior score encontrado (Q6.10) |

---

## 12. `weights_shared_rom.v` — ROM de Pesos com Bootloader

**Finalidade:** Armazena todos os pesos e biases pré-treinados da rede neural e os distribui para os módulos de convolução e camada densa.

**Estrutura da memória:**

Os pesos estão organizados sequencialmente no arquivo `weights_all.mif`:

| Região | Endereços | Conteúdo | Quantidade |
|--------|-----------|----------|------------|
| Pesos convolucionais | 0–35 | 4 filtros × 9 pesos (3×3) | 36 bytes |
| Biases convolucionais | 36–39 | 4 biases (1 por filtro) | 4 bytes |
| Pesos densos | 40–17139 | 19 classes × 900 pesos | 17100 bytes |
| Biases densos | 17140–17158 | 19 biases (1 por classe) | 19 bytes |
| **Total** | | | **17159 bytes** |

**Bootloader de hardware:**

Na inicialização, o módulo executa uma sequência de boot:

1. Uma ROM mestre (`master_rom`) é inicializada pelo Quartus diretamente a partir do arquivo `.mif`.
2. Um contador (`boot_addr`) percorre todos os 17159 endereços da ROM mestre.
3. Para cada endereço, o dado lido é roteado para o registrador ou bloco M9K correspondente:
   - Endereços 0–35 → Registradores dos pesos convolucionais
   - Endereços 36–39 → Registradores dos biases convolucionais
   - Endereços 40–17139 → 19 blocos M9K independentes (um por classe)
   - Endereços 17140–17158 → Registradores dos biases densos
4. Quando o último endereço é processado, o sinal `boot_done` é ativado e a inferência é liberada.

**Leitura durante inferência:**

Após o boot, os 19 blocos M9K são lidos em paralelo usando o endereço `dense_addr` fornecido pelo `cnn_top`. Cada bloco retorna o peso da posição correspondente para sua classe, permitindo que a camada densa processe todas as 19 classes simultaneamente.

**Pipeline de leitura:**

A ROM mestre possui latência de 1 ciclo (leitura registrada). Por isso, o endereço lido é armazenado em `boot_addr_d1` e o dado processado no ciclo seguinte ao da requisição.

---

## 13. `weights_all.mif` — Arquivo de Pesos Quantizados

**Finalidade:** Arquivo de inicialização de memória (Memory Initialization File) que contém os 17159 bytes dos pesos e biases da rede neural, no formato hexadecimal aceito pelo Quartus.

**Formato:** Cada linha contém um valor de 8 bits em hexadecimal. Os valores são inteiros com sinal (complemento de dois) que representam os pesos quantizados em INT8 extraídos do modelo treinado em Python.

**Geração:** Este arquivo é gerado pelo processo de treinamento/quantização em Python (externo ao projeto FPGA) e não deve ser editado manualmente.
