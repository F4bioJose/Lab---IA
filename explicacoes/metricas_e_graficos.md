# Guia de Métricas e Gráficos — Comparação SW vs. HW

Este documento explica como interpretar o relatório gerado pelo comparador do pipeline (`compare_sw_hw.py`). Ele tem como objetivo traduzir os números e os gráficos gerados pela rede de classificação facial de forma acessível, mas mantendo o rigor matemático necessário para validações por comissões técnicas.

---

## 1. O Problema da Quantização (A Origem do Erro)

Ao transferir uma Rede Neural do Keras (Software) para uma FPGA (Hardware), os números deixam de ser processados em formato de "Ponto Flutuante" (Float32, que pode representar casas decimais com altíssima precisão) e passam a ser inteiros de **Ponto Fixo** limitados ao número de fios e registradores do hardware (neste caso, 16 bits).

Esse arredondamento forçado que acontece camada por camada durante as multiplicações é o que chamamos de **Erro de Quantização**. Quando a camada Densa da FPGA calcula 900 somas e multiplicações, pequenas frações são perdidas microscopicamente a cada soma. 

O objetivo das nossas métricas **não é** provar que não há diferença numérica — porque na física do hardware essa diferença sempre existirá. O objetivo das métricas é comprovar que **o formato de onda da perda estatística é esperado e inofensivo**, sem comprometer a lógica matemática de classificação da rede neural.

---

## 2. Entendendo as Métricas (`report.txt`)

A tabela do relatório calcula as discrepâncias numéricas da saída da FPGA (`HW`) em relação à predição-mãe em Python (`SW`), listadas camada a camada (`conv`, `pool`, `flat`, `dense`).

### MAE (Mean Absolute Error) e RMSE (Root Mean Square Error)
* **O que são:** O MAE mede a média simples da diferença absoluta em cada ponto do hardware em comparação com o software. O RMSE faz uma avaliação muito semelhante, mas como ele eleva todos os desvios individuais ao quadrado, ele "pune" severamente erros pontuais excessivamente grandes. 
* **Cálculo Matemático:**
  * MAE: $\frac{1}{N} \sum |HW - SW|$
  * RMSE: $\sqrt{\frac{1}{N} \sum (HW - SW)^2}$
* **Qual a faixa aceitável?** Nessas métricas, as distâncias são contadas usando os números brutos *quantizados*. Na camada Densa, por exemplo, o formato é `Q6.10`. Uma distância de `1` unidade quantizada reflete um valor decimal ínfimo de $1/1024 \approx 0.001$. Portanto, apresentar um MAE entre **`50` e `200`** na camada final é **completamente aceitável e esperado**, pois representa um erro físico médio na casa dos décimos (ex: $130/1024 \approx 0.12$), atestando apenas a diluição aceitável do ponto flutuante, não um bug lógico. O fato do RMSE se manter próximo e proporcional ao MAE comprova que não há aberrações monstruosas escondidas dentro do array que o MAE ignorou.

### MaxErr (Erro Máximo Absoluto)
* **O que é:** Busca exaustivamente pela matriz da respectiva camada e reporta o maior pico isolado de distorção de valor pontual encontrado (o pior cenário).
* **Qual a faixa aceitável?** Picos como `300 a 600` parecem estatisticamente altos em uma tabela inteira, mas da mesma forma que o MAE, se aplicarmos a divisão fracionária (`Q6.10` ou `Q2.14`), esse pico representará uma diferença inofensiva num logit isolado. Ele é útil apenas como sirene de alarme: caso excedesse a casa dos milhares (`> 2000`), indicaria claramente um overflow ou saturação estourada no FPGA.

### Exato%
* **O que é:** O percentual de pontos nos mapas de características que foram calculados perfeitamente pelo hardware — onde $HW - SW = 0$ limpo e sem margem de erro.
* **Qual a faixa aceitável?** Costuma orbitar com normalidade entre **`30% a 60%`** nas camadas finais da rede. É um erro clássico em painéis técnicos acreditar que esse valor deveria ser `100%`. É matematicamente irrealizável que as cadeias de arredondamento de hardware cruzem exatamente no zero decimal de um processador em Python ao longo de milhares de iterações. Esse valor serve mais como curiosidade de similaridade do que como crivo técnico de qualidade.

### Correlação de Pearson (`r`) — O Mestre de Validação!
* **O que é:** Mede a covariância das duas curvas ignorando pequenas reduções de grandeza. Ou seja, ela não se importa se um sinal caiu e se deformou de `15.0` para `14.8` devido à quantização; ela mede se, na hora em que o sinal precisou subir e cair, os picos se respeitaram harmonicamente e mantiveram as feições.
* **Por que é o melhor crivo técnico?** Nas arquiteturas CNNs de classificação baseadas em *Argmax*, pouco importa a pontuação absoluta individual do neurônio; o que impera é o peso relativo — ou seja, se a pessoa "Naira" for identificada, ela só precisa ser superior às demais, seja marcando 30 e os outros 10, seja ela marcando 20 e os outros 5. O Hardware apenas busca o cume. A Correlação de Pearson é perfeita porque ela atesta que a distância relativa se preservou blindada mesmo após perdas numéricas pontuais nos valores do Keras.
* **Qual a faixa aceitável?** 
   - **`≥ 0.90`**: Hardware viável e robusto.
   - **`≥ 0.99`**: Implementação impecável (estado de arte, o que é apresentado pelo atual hardware do repositório).

---

## 3. Decodificando os Gráficos Visuais

### 1. `01_hist_errors.png` (Histograma de Distribuição)
* **O que indica:** Plota no eixo X todos os erros de subtração (`HW - SW`) cometidos em cada camada, e constrói barras empilhadas pela ocorrência (frequência) desses erros.
* **Propósito técnico:** Mostrar a comissão julgadora que a nossa perda de precisão segue uma clássica **"Curva de Sino" (Gaussiana)**, estreita e com centro perfeito no valor `Zero`. Se a curva pendesse inteiramente para um dos lados, diagnosticaríamos que a implementação FPGA possui um "viés matemático ou de deslocamento" (bias) estragando as convoluções. Sendo centralizada, provamos que as falhas causadas pela quantização agem apenas como um tênue ruído branco inócuo.

### 2. `02_scatter_layers.png` (Gráfico de Dispersão Cruzada)
* **O que indica:** Cruza cada saída isolada da FPGA (Eixo Y) com a sua respectiva saída nativa no Keras (Eixo X). 
* **Propósito técnico:** Em um mundo hipotético perfeitamente alinhado, todos os pontos gerados formariam uma única reta diagonal sem fios de cabelo ($Y = X$). Uma nuvem de dados gorda e dispersa condenaria o circuito à aleatoriedade. O fato do nosso projeto manter a dispersão agrupada rentemente em volta da reta tracejada atesta e reflete perfeitamente o alto coeficiente de correlação de `0.999` exibido nos relatórios matemáticos.

### 3. `03_barplot_scores.png` (Comparação das Pontuações Finais)
* **O que indica:** Exibe o resultado nu e cru da última camada de inteligência da rede — os famigerados "Scores" (conhecidos no treinamento de IAs como *Logits*).
* **O que é um Logit?** Uma rede neural não fornece "certezas mágicas"; a sua última etapa cospe um pacote numérico solto que reflete o nível da confiança de identificação do circuito. Ex: `+30.2` para a pessoa "A" e `-15.3` para a pessoa "B". Como no FPGA implementamos a decisão baseada em `Argmax`, a placa elege como a face ganhadora, instintivamente, o maior Logit final presente no barramento.
* **Propósito técnico:** Atestar com total transparência visual que as barras vermelhas (Desconhecido) e as demais barras de candidatos no FPGA preservaram exatamente as mesmas proporções arquitetadas pelo Software. Trata-se do desfecho onde visualizamos que o hardware não inventou vencedores falsos.

### 4. `04_heatmap_conv.png` (Mapa de Correlação de Canais)
* **O que indica:** A primeira camada da rede aplica os 4 filtros (3x3) para tentar garimpar as texturas primárias de reconhecimento no rosto da imagem. O mapa calcula o coeficiente de cruzamento entre os processamentos em Software vs Hardware em todos os seus 4 canais.
* **Propósito técnico:** Comprovar a ordem de montagem física dos filtros. Para o hardware estar funcionalmente coerente, as retas de correspondência entre canais (`Filtro_0 vs Filtro_0`) precisam destacar um quadrado vermelho escuro na diagonal principal e cores fracas nas diagonais de fora. Isso informa à banca que os canais multiplicadores paralelos da placa não sofreram cruzamentos indevidos nos cabos virtuais (cross-talk de barramento).
