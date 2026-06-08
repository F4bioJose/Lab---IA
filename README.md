# Nova Feature: Overlay de Sprites de Identificação (VGA)

Esta etapa do projeto é uma expansão do pipeline de vídeo pré-existente (system_top). O objetivo foi implementar a exibição dinâmica do nome do usuário identificado pela rede neural, renderizando um letreiro de texto em tempo real abaixo do frame de vídeo (SRAM) no monitor VGA.
## 1. Arquivos Acrescentados

    rom_sprites.mif: Arquivo de inicialização de memória (Memory Initialization File) contendo a matriz binária de todos os pixels dos letreiros.

    rom_sprites.v / .qip: Módulos HDL gerados através do IP Catalog (MegaWizard) da Intel/Altera para instanciar a memória física no FPGA.

## 2. Modelagem e Geração dos Sprites (HTML/CSS)

Para evitar artefatos de renderização e perda de nitidez nas fontes, optamos por não utilizar editores de imagem convencionais.
Os sprites foram gerados renderizando um arquivo HTML/CSS contendo as caixas de texto com fontes sem serifa. O CSS garantiu dimensões absolutas no nível do pixel, permitindo a extração de uma "fita" perfeita, onde o branco representa o bit lógico 1 (texto) e o preto representa o bit 0 (fundo). Um script Python/OpenCV foi utilizado para varrer essa imagem e convertê-la no arquivo .mif.
## 3. Dimensionamento e Arquitetura de Memória

A escolha das dimensões da caixa de texto foi guiada pela otimização extrema dos blocos de memória (M9K) e portas lógicas do Cyclone IV:

    Tamanho do Sprite: 256px de largura por 32px de altura. Essa proporção garante que o tamanho do bloco de cada aluno seja uma potência de 2 exata (8.192 pixels).

    Capacidade: A ROM foi dimensionada para comportar 19 classes (Sendo o ID = 0 reservado para estado "Desconhecido", e os IDs 1 a 18 para os alunos cadastrados).

    Endereçamento: A fita vertical completa possui 155.648 endereços. Graças à arquitetura em potência de 2, a transição entre os nomes não exige circuitos multiplicadores: o FPGA localiza a página de memória aplicando um simples Bit Shift de 13 casas no class_id.

## 4. Instanciação da ROM (MegaWizard)

Para acomodar a estrutura acima, utilizamos o MegaWizard para alocar uma ROM 1-PORT de 1 bit de largura por 155.648 de profundidade. O barramento de endereço (address) foi dimensionado com 18 bits para cobrir toda a memória. A ROM foi pré-carregada com o arquivo rom_sprites.mif nativamente na compilação.
## 5. Reutilização do Pipeline de Vídeo (system_top)

A infraestrutura de letreiros aproveitou integralmente a lógica de temporização de 25.175 MHz pré-existente.

    Foram criados parâmetros de offset (H_START_SPRITE, V_START_SPRITE) para posicionar a bounding box do nome (256x32) perfeitamente centralizada e imediatamente abaixo do frame de 384x384 vindo da UART/SRAM.

    A latência de 1 ciclo de clock da nova ROM foi compensada criando o registrador de atraso in_sprite_window_d.

    O multiplexador de renderização (always @*) foi expandido para atuar como um sistema de camadas (Layers), priorizando a escrita da SRAM na área da foto, e da ROM na área do texto.

## 6. Simulação de Teste via Hardware (Switches)

Como o módulo da Inteligência Artificial (CNN) ainda está em desenvolvimento, a validação visual do hardware foi construída injetando sinais diretos das chaves da placa DE2-115:

    SW[4:0]: Codificam um número binário de 5 bits para simular a mudança instantânea do class_id de 0 a 18, paginando os diferentes nomes na tela.

    SW[17]: Simula o estado access_granted, alterando a cor de pintura do texto ativamente entre vermelho (negado) e verde (autorizado).

## 7. Próximos Passos (Integração com a CNN)

Na versão final (produção), os barramentos de chaves físicas (SW) serão desativados. O parâmetro class_id passará a receber o fio out_class_id gerado como output do módulo da rede neural, tornando a identificação biométrica e o salto de memória na ROM operações automáticas e combinacionais.