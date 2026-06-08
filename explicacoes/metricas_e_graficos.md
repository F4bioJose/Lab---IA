# Métricas e Gráficos — Validação da Inferência HW vs. SW

Este documento explica como interpretar o relatório numérico (`report.txt`) e os quatro gráficos gerados pelo script [`compare_sw_hw.py`](../scripts/compare_sw_hw.py). A finalidade do conjunto é validar, camada por camada, que a implementação em hardware (FPGA, ponto fixo) preserva a capacidade de classificação do modelo de software (Keras, ponto flutuante).

---

## 1. Contexto — O Erro de Quantização

### Por que existe erro?

O modelo Keras opera em ponto flutuante IEEE 754 (Float32), que dispõe de ~7 dígitos significativos. A FPGA opera em **ponto fixo de 16 bits** com formatos distintos por camada:

| Camada | Formato | Bits fracionários | Resolução mínima |
|--------|---------|-------------------|-------------------|
| Convolução / Pool / Flat | Q2.14 | 14 | $1/16384 \approx 6.1 \times 10^{-5}$ |
| Dense (scores) | Q6.10 | 10 | $1/1024 \approx 9.8 \times 10^{-4}$ |

A cada operação aritmética no hardware (multiplicação, acumulação, saturação), o resultado é **arredondado para o inteiro representável mais próximo**. Essas microdiferenças se acumulam ao longo das 900 multiplicações da camada densa.

### Exemplo numérico de uma operação elementar

Considere uma multiplicação na camada densa:

$$x = 0.50781 \quad (\text{Q2.14} = 8320), \qquad w = 0.25 \quad (\text{Q1.7} = 32)$$

$$\text{Produto exato (float):} \quad 0.50781 \times 0.25 = 0.12695$$

$$\text{Produto HW (Q3.21):} \quad 8320 \times 32 = 266240 \quad \Rightarrow \quad 266240 / 2^{21} = 0.12695$$

Neste caso o resultado é exato. Porém, quando o produto float tem mais dígitos do que os 21 bits fracionários permitem, ocorre truncamento. Após 900 acumulações, essas frações perdidas geram uma diferença mensurável — mas **previsível e limitada**.

### O objetivo da validação

Não se busca provar que $\text{HW} = \text{SW}$ em cada ponto (isso é matematicamente impossível). O objetivo é demonstrar que:

1. Os erros seguem uma **distribuição simétrica centrada em zero** (sem viés sistemático).
2. A **ordem relativa** dos scores é preservada (o argmax não muda).
3. A **correlação linear** entre as saídas é próxima de 1.

---

## 2. Métricas do Relatório (`report.txt`)

O relatório calcula as seguintes métricas para cada camada (`conv`, `pool`, `flat`, `dense`). Todas as métricas operam sobre os **valores inteiros quantizados** — ou seja, em unidades da representação de ponto fixo da respectiva camada.

### 2.1 MAE — Mean Absolute Error (Erro Absoluto Médio)

**Definição:**

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| \text{HW}_i - \text{SW}_i \right|$$

onde $N$ é o número total de amostras da camada.

**Significado:** Mede a distância média absoluta entre cada par de valores HW/SW, em unidades quantizadas. É robusto a valores extremos por não elevar ao quadrado.

**Faixa aceitável — derivação para a camada Dense (Q6.10):**

Na camada Dense, cada unidade quantizada vale $1/2^{10} = 1/1024$. Portanto:

$$\text{MAE}_{\text{float}} = \text{MAE}_{\text{quant}} \times \frac{1}{1024}$$

Um MAE de 130 unidades quantizadas equivale a:

$$130 \times \frac{1}{1024} \approx 0.127 \text{ em float}$$

Dado que os scores tipicamente variam de $-32$ a $+32$ (faixa Q6.10), um erro de ~0.13 é insignificante.

| Camada | Formato | Conversão float | MAE típico aceitável |
|--------|---------|-----------------|----------------------|
| conv / pool / flat | Q2.14 | MAE / 16384 | < 5 (≈ 0.0003 float) |
| dense | Q6.10 | MAE / 1024 | < 200 (≈ 0.20 float) |

### 2.2 RMSE — Root Mean Square Error (Raiz do Erro Quadrático Médio)

**Definição:**

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \text{HW}_i - \text{SW}_i \right)^2}$$

**Significado:** Semelhante ao MAE, mas penaliza erros grandes de forma quadrática. Se um único ponto tiver um erro desproporcionalmente alto, o RMSE será significativamente maior que o MAE.

**Critério de validação:** O RMSE deve ser **proporcional** ao MAE. Se $\text{RMSE} \gg \text{MAE}$, isso indica a presença de outliers severos (possível overflow ou bug no hardware). A razão $\text{RMSE}/\text{MAE}$ tipicamente fica entre $1.0$ e $2.0$ para ruído de quantização gaussiano.

### 2.3 MaxErr — Erro Máximo Absoluto

**Definição:**

$$\text{MaxErr} = \max_{i=1}^{N} \left| \text{HW}_i - \text{SW}_i \right|$$

**Significado:** Identifica o pior caso isolado. Funciona como um alarme: um valor excessivamente alto indica overflow, saturação incorreta ou erro lógico.

**Faixa aceitável:** Na camada Dense (Q6.10), um MaxErr de até ~600 equivale a $600/1024 \approx 0.59$ em float — aceitável. Valores acima de ~2000 ($\approx 2.0$ em float) indicam anomalia.

### 2.4 Exato% — Percentual de Valores Exatos

**Definição:**

$$\text{Exato\%} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\text{HW}_i = \text{SW}_i\right] \times 100\%$$

onde $\mathbb{1}[\cdot]$ é a função indicadora (1 se verdadeiro, 0 caso contrário).

**Significado:** Percentual de pontos onde o hardware produziu exatamente o mesmo valor inteiro que o software (erro = 0).

**Interpretação:** Nas camadas iniciais (conv, pool), este valor tende a ser alto (> 80%), porque as operações mais simples (poucas multiplicações) preservam a precisão. Na camada Dense, que acumula 900 multiplicações por neurônio, o Exato% naturalmente cai para 30–60%, o que é **esperado e normal** — não é um indicador de falha.

### 2.5 Correlação de Pearson ($r$)

**Definição:**

$$r = \frac{\sum_{i=1}^{N}(\text{SW}_i - \bar{\text{SW}})(\text{HW}_i - \bar{\text{HW}})}{\sqrt{\sum_{i=1}^{N}(\text{SW}_i - \bar{\text{SW}})^2 \cdot \sum_{i=1}^{N}(\text{HW}_i - \bar{\text{HW}})^2}}$$

onde $\bar{\text{SW}}$ e $\bar{\text{HW}}$ são as médias das respectivas amostras.

**Significado:** Mede o grau de **proporcionalidade linear** entre as saídas SW e HW. É invariante a escala e deslocamento — se o hardware reduzir todos os valores por um fator constante (por conta da quantização), o $r$ permanece inalterado.

**Por que é o crivo mais importante?**

A CNN utiliza **argmax** como regra de decisão: a classe predita é aquela com o maior score. Essa decisão depende apenas da **ordem relativa** dos scores, não dos valores absolutos. A correlação de Pearson captura exatamente isso: se $r \approx 1$, a relação de "quem é maior que quem" está preservada, garantindo que o argmax produzirá o mesmo resultado.

**Faixas de referência:**

| Valor de $r$ | Interpretação |
|---------------|---------------|
| $r \geq 0.999$ | Implementação de referência (estado da arte) |
| $0.99 \leq r < 0.999$ | Excelente — hardware totalmente funcional |
| $0.90 \leq r < 0.99$ | Aceitável — classificação preservada, mas com ruído perceptível |
| $r < 0.90$ | Problemático — investigar possíveis bugs de saturação ou endereçamento |

---

## 3. Gráficos — Detalhamento Individual

O script gera quatro gráficos, cada um com uma função específica na cadeia de validação. Os gráficos são numerados de `01` a `04` e salvos em `comparacao/plots/<nome>/`.

### 3.1 `01_hist_errors.png` — Histograma de Distribuição dos Erros

**Estrutura:** Um sub-gráfico por camada (`conv`, `pool`, `flat`), dispostos lado a lado. A camada `dense` não é incluída neste gráfico (tem seu próprio barplot).

**Eixos:**
- **Eixo X:** Valor do erro pontual $\text{HW}_i - \text{SW}_i$, em unidades quantizadas inteiras. Valores negativos indicam que o hardware produziu um resultado menor que o software; valores positivos, o contrário.
- **Eixo Y:** Contagem (frequência) — quantas amostras apresentam cada valor de erro.

**Elementos visuais:**
- **Barras verdes:** Histograma com 60 bins, mostrando a distribuição de frequência dos erros.
- **Linha tracejada preta vertical:** Marca o ponto $\text{erro} = 0$ (concordância perfeita).

**Como interpretar:**

| Observação | Diagnóstico |
|------------|-------------|
| Distribuição simétrica, estreita, centrada em 0 | ✓ Erro de quantização puro (ruído branco). Hardware correto. |
| Distribuição deslocada do zero (viés) | ✗ Possível erro de arredondamento sistemático ou bias mal aplicado. |
| Caudas longas ou picos distantes do centro | ✗ Outliers — possível overflow ou saturação incorreta em posições específicas. |

**Relevância:** Demonstra que o erro do hardware não possui viés direcional — ou seja, não há tendência sistemática de "subestimar" ou "superestimar" as ativações.

---

### 3.2 `02_scatter_layers.png` — Dispersão Cruzada SW × HW

**Estrutura:** Um sub-gráfico por camada (`conv`, `pool`, `flat`), dispostos lado a lado. Limita automaticamente a 5.000 pontos por camada para legibilidade.

**Eixos:**
- **Eixo X:** Valor de saída do software (quantizado), para cada amostra $i$.
- **Eixo Y:** Valor de saída do hardware (quantizado), para a mesma amostra $i$.

**Elementos visuais:**
- **Pontos azuis** (s=3, semi-transparentes): Cada ponto representa um par $(SW_i, HW_i)$ — a ativação de um único elemento do mapa de características.
- **Linha tracejada cinza:** Reta $y = x$ — a referência de concordância perfeita.

**Como interpretar:**

| Observação | Diagnóstico |
|------------|-------------|
| Pontos concentrados sobre a reta $y = x$ | ✓ Hardware reproduz fielmente o software. |
| Nuvem fina e alongada em torno da reta | ✓ Pequeno ruído de quantização — normal e esperado. |
| Nuvem espalhada ou desvio da diagonal | ✗ Divergência significativa — investigar camada específica. |
| Pontos "cortados" em valores extremos | ⚠ Saturação atingida — pode ser normal dependendo da faixa do formato. |

**Relevância:** É a representação visual direta da correlação de Pearson. Uma nuvem apertada sobre a diagonal confirma $r \approx 1$.

---

### 3.3 `03_barplot_scores.png` — Scores Finais das 19 Classes

**Estrutura:** Gráfico de barras agrupadas com todas as 19 classes no eixo X.

**Eixos:**
- **Eixo X:** Índice da classe (0 a 18). O rótulo `Desc` corresponde à classe 0 (Desconhecido); os demais são numerados de 1 a 18.
- **Eixo Y:** Score bruto (logit) em unidades Q6.10 — é o valor inteiro de 16 bits com sinal que a camada densa produz para cada classe.

**Elementos visuais:**
- **Barras azuis (esquerda):** Score calculado pelo software (Keras).
- **Barras laranja (direita):** Score calculado pelo hardware (Verilog).
- **Barras vermelhas:** Tanto SW quanto HW usam vermelho para a classe 0 (Desconhecido), destacando visualmente a classe de rejeição.
- **Linha preta horizontal:** Nível zero, para referência.
- **Anotações no topo:** Indicam a classe predita pelo SW (`SW→Nome`) e pelo HW (`HW→Nome`), posicionadas sobre a respectiva barra vencedora.

**Como interpretar:**

O gráfico mais importante da validação. Ele responde diretamente à pergunta: **o hardware chegou à mesma conclusão que o software?**

| Observação | Diagnóstico |
|------------|-------------|
| Barras azuis e laranja com proporções similares em todas as classes | ✓ Ordem relativa preservada. |
| A barra mais alta de SW e HW é a mesma classe | ✓ Argmax coincide — classificação correta. |
| Uma classe "cresce" ou "encolhe" visivelmente no HW | ⚠ Verificar se a diferença é suficiente para inverter o argmax. |
| A classe vencedora muda entre SW e HW | ✗ Falha de classificação — investigar camadas anteriores. |

**Relevância:** Testa o critério final de aceite: se ambos os sistemas predizem a mesma classe, a implementação está funcionalmente correta, independentemente das diferenças numéricas nas camadas intermediárias.

---

### 3.4 `04_heatmap_conv.png` — Mapa de Correlação dos Canais Convolucionais

**Estrutura:** Matriz de correlação 8×8 representada como mapa de calor. As 8 variáveis são: 4 canais do software (`SW_f0`, `SW_f1`, `SW_f2`, `SW_f3`) e 4 canais do hardware (`HW_f0`, `HW_f1`, `HW_f2`, `HW_f3`).

**Eixos:**
- **Eixo X e Y:** Rótulos dos 8 canais (`SW_f0`..`SW_f3`, `HW_f0`..`HW_f3`).
- **Cor da célula:** Coeficiente de correlação de Pearson entre as duas variáveis indicadas, calculado por `np.corrcoef`. A escala vai de -1 (vermelho escuro) a +1 (verde escuro), passando por amarelo em 0.
- **Valores numéricos:** O coeficiente $r$ é anotado em cada célula (2 casas decimais). Texto em branco quando $|r| \geq 0.70$; preto caso contrário.

**Barra de cores:** Lateral, indicando o mapeamento numérico da paleta `RdYlGn` (Red-Yellow-Green).

**Como interpretar:**

A matriz 8×8 se divide em 4 quadrantes lógicos:

| Quadrante | Posição | Conteúdo | Valor esperado |
|-----------|---------|----------|----------------|
| Superior-esquerdo (4×4) | SW vs. SW | Auto-correlação entre canais SW | Diagonal = 1.00; fora pode variar |
| Inferior-direito (4×4) | HW vs. HW | Auto-correlação entre canais HW | Diagonal = 1.00; padrão similar ao SW |
| Superior-direito (4×4) | SW vs. HW | **Correlação cruzada** (o alvo principal) | Diagonal ≈ 1.00 |
| Inferior-esquerdo (4×4) | HW vs. SW | Espelho do superior-direito | Diagonal ≈ 1.00 |

O **resultado-chave** está nos quadrantes fora da diagonal principal (SW vs. HW):

| Observação | Diagnóstico |
|------------|-------------|
| Diagonal dos quadrantes cruzados ≈ 1.00 (verde escuro) | ✓ Cada filtro HW corresponde corretamente ao filtro SW equivalente. |
| Fora da diagonal cruzada ≈ valores baixos | ✓ Os filtros são independentes entre si (sem cross-talk). |
| Diagonal cruzada com valores baixos (< 0.9) | ✗ Filtros trocados ou endereçamento incorreto da ROM de pesos. |
| Valores altos fora da diagonal cruzada | ✗ Possível cross-talk entre canais do barramento. |

**Relevância:** Valida que o mapeamento dos 4 filtros convolucionais está correto — filtro 0 do hardware opera com os mesmos pesos do filtro 0 do software, etc. É a única métrica que detecta erros de **endereçamento da ROM de pesos**.

---

## 4. Checklist de Aprovação

Para considerar a implementação em hardware **validada**, todos os seguintes critérios devem ser satisfeitos simultaneamente:

| # | Critério | Métrica/Gráfico | Limiar |
|---|----------|-----------------|--------|
| 1 | Erro sem viés | `01_hist_errors.png` | Distribuições centradas em 0 |
| 2 | Concordância linear | `02_scatter_layers.png` | Pontos sobre a reta $y = x$ |
| 3 | Correlação alta | Pearson $r$ no `report.txt` | $r \geq 0.99$ em todas as camadas |
| 4 | Classificação correta | `03_barplot_scores.png` | Mesma classe predita por SW e HW |
| 5 | Filtros corretos | `04_heatmap_conv.png` | Diagonal cruzada $\geq 0.99$ |
| 6 | Sem outliers extremos | MaxErr no `report.txt` | < 2000 unidades na camada Dense |
| 7 | RMSE proporcional ao MAE | `report.txt` | $\text{RMSE}/\text{MAE} < 2.0$ |
