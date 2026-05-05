# Tiny-CNN FPGA: Fechadura Biométrica Inteligente

## Aviso: Dados Simulados (Status Atual)

A finalidade desta etapa é atestar o arranjo arquitetural dos blocos, o tempo de ciclo dos processadores MAC e as garantias do formato de ponto fixo. Por esta razão:

* **Pesos Fabricados:** O arquivo atual de inicialização da memória ROM de parâmetros (`weights_all.mif`), localizado em `modulos_verilog/`, carrega dados arbitrários gerados sinteticamente para validação dos fios. Ele **não provém** de uma rede neural formalmente treinada (TensorFlow/Keras).
* **Entradas Aleatórias:** O vetor de processamento injetado através do simulador via testbench (ex: imagens dentro de `/inputs/`) também compõe matrizes hexadecimais aleatórias/sequenciais confeccionadas estritamente para submeter as instâncias a testes de sanidade geométrica (filtros 3x3, pooling 2x2, argmax), e não uma fotografia real proveniente de webcam.

## Estrutura de Diretórios

* `modulos_verilog/`: Acomoda a totalidade dos blocos construtivos da rede. Inclui o orquestrador macro (`cnn_top.v`), o buffer de janelas, os núcleos convolucionais espacializados, módulo de max-pooling, a interface flatten, memória ROM compartilhada, camada densa preditiva e o cômputo da função de ativação Argmax com Threshold.
* `scripts/`: Dispõe de rotinas em script TCL para ambiente ModelSim voltadas à automação de integração; agrupando tarefas de compilação dos módulos (`compile_project.do`) e as rotinas instanciadoras da simulação RTL de verificação (`run_project.do`).
* `inputs/`: (A ser povoado) Reservado para estocar os tensores fotográficos transformados para valores hexadecimais puros, visando a injeção estática no simulador.
* `explicacoes/`: Pasta dedicada à documentação estendida do projeto. Contém a explicação completa da arquitetura do pipeline (`pipeline.txt`) e o diagrama visual de transição da FSM principal (`FSM.md`).

## Como Simular o Projeto (Validação RTL)

O ambiente de simulação no ModelSim está completamente automatizado através dos scripts TCL. Para executar o teste do pipeline:

1. Assegure-se de que a imagem de teste (vetor hexadecimal, ex: `teste2.txt`) encontra-se no diretório `inputs/`.
2. Abra o terminal interativo do **ModelSim** garantindo que o diretório atual seja a raiz do repositório.
3. Para **compilar** a hierarquia completa de módulos Verilog, digite:
   ```tcl
   do scripts/compile_project.do
   ```
4. Para **iniciar a simulação** (que injetará o frame na arquitetura e revelará os scores numéricos no console), digite:
   ```tcl
   do scripts/run_project.do
   ```

## Documentação Estendida

Para uma compreensão detalhada acerca do percurso exato dos barramentos numéricos, diagramação da FSM principal de controle e do comportamento de cada módulo individual, verifique a documentação aprofundada incluída na raiz do repositório:

**[Consulte o arquivo pipeline.txt](explicacoes/pipeline.txt)**

A documentação aborda em linguagem minuciosa o comportamento do pipeline implementado, a adequação exigida do formato Q1.7 e Q2.14, bem como as restrições projetadas visando o futuro acoplamento de SRAM externa paralela para controle de Vídeo VGA e barramentos receptores UART.