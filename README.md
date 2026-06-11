# Sistema Integrado: Fechadura Biométrica (CNN + VGA) em FPGA

Implementação em hardware de uma Rede Neural Convolucional (CNN) unificada a um pipeline de vídeo VGA, sintetizada na FPGA **DE2-115 (Cyclone IV EP4CE115F29C7)** para classificação biométrica facial em tempo real. 

O sistema recebe um fluxo de vídeo contínuo via UART, exibe a imagem no monitor VGA em tempo real (ampliada com supersampling) e executa a inferência na CNN em hardware dedicado. O resultado da inferência determina qual sprite de nome será renderizado no monitor e também é refletido nos LEDs da placa.

---

## 🏗️ Arquitetura do Sistema Unificado

O projeto funde dois pipelines anteriormente separados: o fluxo da rede neural (19 classes) e a renderização gráfica via VGA. O orquestrador central garante que a imagem do framebuffer seja compartilhada entre a CNN e o VGA.

```
PC (Python OpenCV)  ──[UART 115200 baud]──►  FPGA DE2-115
                                                  │
                                          uart_rx (32×32)
                                                  │
                                         Framebuffer M9K ◄─── (KEY[0] limpa memória)
                                                  │
                ┌─────────────────────────────────┴─────────────────────────────────┐
                ▼                                                                   ▼
       Pipeline CNN (50 MHz)                                             Pipeline VGA (25 MHz)
   Line Buffer → Conv 3×3 (ReLU)                                   vga_sync gera pixels 640×480
                │                                                                   │
         Max Pooling 2×2                                         Lê do framebuffer com scaling 12×
                │                                                 (Exibe 384×384 no centro da tela)
          Flatten (900)                                                             │
                │                                                    Acessa ROM de Sprites de Nomes
        Dense 900×19 classes                                                        │
                │                                                 Mistura: Imagem + Sprite + Fundo
           Argmax Puro                                                              │
                │                                                                   ▼
         class_id[4:0] ────────────────FSM Orquestradora─────────────────────► Monitor VGA
```

### Comportamento da FSM Orquestradora
- **Reset / Sem Rosto:** O framebuffer é apagado e o sprite exibe "Desconhecido" na cor vermelha.
- **Recebendo Imagem:** A imagem é exibida ao vivo no monitor.
- **Inferência Concluída:** A CNN sinaliza a FSM, que captura o `class_id` gerado. O sprite de texto no monitor atualiza para exibir o nome da pessoa na cor verde (se membro autorizado) ou "Desconhecido" em vermelho (se não reconhecido).

---

## 👥 Classes (Membros Cadastrados)

A rede foi treinada com 19 classes. A Classe 0 representa uma pessoa "Desconhecida" ou "Acesso Negado". As classes de 1 a 18 representam membros autorizados.

| ID | Nome | ID | Nome | ID | Nome |
|----|------|----|------|----|------|
| 0 | **Desconhecido** | 7 | Rafael | 14 | Fabio |
| 1 | Igor | 8 | Samuel | 15 | Felipe |
| 2 | Joao | 9 | Yuri | 16 | Gabriel |
| 3 | Jose Henrique | 10 | Anna Carol | 17 | Horacio |
| 4 | Julia | 11 | Bruno | 18 | Hugo |
| 5 | Lucio | 12 | Diego | | |
| 6 | Naira | 13 | Eduardo | | |

---

## 📂 Estrutura de Arquivos

```
Lab---IA/
├── haarcascade_frontalface_default.xml  ← Algoritmo de detecção facial
├── modulos_verilog/
│   ├── fpga_top_unified.v       ← Top-level principal (Orquestrador CNN + VGA)
│   ├── framebuffer_32x32.v      ← BRAM dual-port com lógica de hardware clear
│   ├── cnn_top.v                ← Top-level e FSM do pipeline da CNN
│   ├── uart_rx.v                ← Receptor UART
│   └── (outros módulos da CNN: conv, pooling, dense, flatten, argmax)
├── vga_artefato/
│   ├── vga_sync.v               ← Temporizador de sync 640x480 do VGA
│   ├── rom_sprites.v            ← ROM contendo os 19 sprites de nomes
│   └── vga_pll.v                ← Gerador de clock 25MHz para o VGA
├── quartus_cnn/                 
│   ├── cnn_inference.qpf        ← Projeto Quartus unificado
│   └── cnn_inference.qsf        ← Pin assignments (VGA + UART + LEDs)
└── scripts/
    └── send_image_32x32.py      ← Script Python de transmissão de vídeo ao vivo
```

---

## 🚀 Como Executar

### 1. Síntese e Gravação na Placa

1. Abra o Quartus Prime (versão 18.1 ou superior).
2. Vá em **File → Open Project** e selecione `quartus_cnn/cnn_inference.qpf`.
3. Compile o projeto: **Processing → Start Compilation** (`Ctrl+L`).
4. Conecte a placa DE2-115 via USB Blaster.
5. Programe via JTAG: **Tools → Programmer → Start** selecionando o arquivo `.sof`.
6. Conecte um cabo serial (RS-232) na porta apropriada da DE2-115.
7. Conecte o cabo VGA da DE2-115 ao monitor.

### 2. Transmissão de Vídeo (Python)

Instale as dependências no computador host:
```bash
pip install opencv-python pyserial numpy
```

Rode o script de envio com a opção de preview ativada para iniciar a captura contínua da webcam:
```bash
python scripts/send_image_32x32.py --port /dev/ttyUSB0 --preview
```

**Parâmetros suportados:**
- `--port /dev/ttyUSB0` : Porta serial (mude para `COM3`, etc no Windows).
- `--preview` : Abre uma janela no PC mostrando a visão da câmera.
- `--camera 0` : Índice da câmera a ser usada (padrão é 0).
- `--file imagem.jpg` : Envia uma imagem estática ao invés do feed da webcam.

Pressione a tecla `q` na janela de preview ou `Ctrl+C` no terminal para encerrar a transmissão.

---

## 🎛️ Controles na Placa (Hardware)

| Componente | Função |
|------------|--------|
| **KEY[0]** | **Reset do Sistema / Apagar Monitor:** Pressionar este botão limpa ativamente toda a memória do Framebuffer e força a FSM a reiniciar. A tela do monitor VGA ficará preta imediatamente e o sprite voltará a exibir "Desconhecido". |
| **LEDG[4:0]** | `class_id`: Exibe a classe predita em binário (0 = Desconhecido; 1–18 = membro autorizado). Mantém o valor salvo (latched) até a próxima inferência. |
| **LEDG[5]** | Debug: Aceso se a imagem recebida conter pixels não-nulos. |
| **LEDG[6]** | `done`: Aceso enquanto a última inferência estiver concluída com sucesso. Apaga brevemente ao receber um novo frame. |
| **LEDG[7]** | `unknown`: Aceso caso a CNN não reconheça o rosto da pessoa (classe 0 predita). |

---

## 📚 Documentação Técnica Avançada

Para um aprofundamento extremo em todos os aspectos arquiteturais do projeto (Análise das camadas neurais, explicação do conversor OpenCV em Python, formatos matemáticos Q1.7, constrições do Quartus e Diagramas de Máquina de Estado), acesse:
👉 **[Documentação Detalhada do Sistema](file:///mnt/Data/Lab---IA/explicacoes/documentacao_detalhada.md)**