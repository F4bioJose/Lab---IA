# Scripts Python — Documentação Detalhada

Este documento descreve os arquivos da pasta `scripts/` e o classificador Haarcascade na raiz do projeto, responsáveis pela captura de vídeo, detecção de rostos e transmissão serial para a FPGA.

**Documento principal:** [modulos_verilog.md](modulos_verilog.md) — Descrição de todos os módulos de hardware que recebem os dados enviados por este script.

---

## Visão Geral

O script Python opera no computador host e é a única interface de entrada do sistema. Ele captura vídeo da webcam, detecta rostos, pré-processa a imagem e a transmite via porta serial para a FPGA.

```
Webcam → OpenCV → Haarcascade → CLAHE → Resize → Q1.7 → UART → FPGA
```

---

## 1. `send_image_32x32.py` — Script Principal

**Finalidade:** Capturar imagens da webcam (ou de um arquivo), pré-processá-las e enviá-las via UART serial para a FPGA DE2-115.

### 1.1 Modos de Operação

| Modo | Comando | Descrição |
|------|---------|-----------|
| **Arquivo estático** | `--file imagem.jpg` | Envia uma única imagem e encerra |
| **Arquivo hex** | `--file dados.txt --hex` | Envia dados brutos em hexadecimal |
| **Webcam contínua** | (sem `--file`) | Captura da câmera em loop contínuo |
| **Simulação** | `--mock` | Testa o pipeline sem abrir porta serial |

### 1.2 Parâmetros da Linha de Comando

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--port` | `/dev/ttyUSB0` | Porta serial (ex: `COM3` no Windows) |
| `--baud` | 2000000 | Baud rate (deve coincidir com o `uart_rx`) |
| `--camera` | 0 | Índice da câmera |
| `--file` | — | Enviar imagem estática em vez da webcam |
| `--hex` | — | Indica que `--file` contém valores hexadecimais |
| `--mock` | — | Rodar sem porta serial real |
| `--no-haar` | — | Desabilitar detecção de rosto |
| `--preview` | — | Exibir janela de preview no PC |
| `--raw` | — | Enviar sem quantizar para Q1.7 |

### 1.3 Pipeline de Processamento de Imagem

O processamento segue etapas idênticas às do treinamento do modelo, garantindo que a FPGA receba imagens no mesmo formato esperado pela rede neural:

1. **Conversão para escala de cinza** — `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`. Melhora o contraste local da imagem, compensando variações de iluminação. O parâmetro `clipLimit=2.0` limita a amplificação de ruído.

3. **Detecção de rosto (Haarcascade)** — Usa cascatas adaptativas de Haar (`detectMultiScale`) com três configurações de sensibilidade crescente:
   - scaleFactor=1.2, minNeighbors=7, minSize=200 (mais restritiva)
   - scaleFactor=1.1, minNeighbors=6, minSize=150
   - scaleFactor=1.1, minNeighbors=5, minSize=120 (mais permissiva)
   
   Se a configuração mais restritiva não encontrar rosto, tenta a próxima. Quando detecta, seleciona o maior rosto e aplica padding de 15% ao redor da bounding box.

4. **Redimensionamento** — `cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)` para frames de rosto, ou `(128, 128)` para frames de vídeo.

5. **Quantização Q1.7** — Converte de `[0, 255]` (uint8) para `[0, 127]` (Q1.7 positivo) via: `round(pixel × 127 / 255)`. Esse formato é o esperado pelos pesos da CNN no FPGA.

### 1.4 Protocolo de Comunicação UART

Cada frame enviado é precedido por um **byte de controle** que informa à FPGA o tipo de dados que virão a seguir:

| Byte de controle | Valor | Tipo de frame | Tamanho | Processamento na FPGA |
|-----------------|-------|---------------|---------|----------------------|
| `CONTROL_FACE` | `0xFF` | Rosto 32×32 | 1024 bytes | Armazenado no framebuffer_32x32, inferência CNN executada |
| `CONTROL_NO_FACE` | `0x00` | Vídeo 128×128 | 16384 bytes | Armazenado no framebuffer_128x128, apenas exibição VGA |

A sequência completa de um envio é:
1. Enviar 1 byte de controle
2. Enviar N bytes do frame (1024 ou 16384)
3. O hardware processa automaticamente ao receber o último byte

### 1.5 Máquina de Estados do Modo Webcam

No modo de captura contínua, o script segue uma máquina de estados que controla quando rostos são procurados e enviados:

| Estado | Comportamento |
|--------|--------------|
| `paused` | Envia apenas vídeo 128×128 sem procurar rostos. Ativado por padrão quando `--preview` é usado. O usuário pode alternar com a tecla ESPAÇO. |
| `detecting` | Procura rostos em cada frame capturado. Se encontrar, envia o rosto 32×32 e transiciona para `cooldown`. Se não encontrar, envia vídeo 128×128. |
| `cooldown` | Pausa total de 2.5 segundos após enviar um rosto. Nenhum frame é enviado durante este período, permitindo que a FPGA conclua a inferência. |
| `grace` | Após o cooldown, envia apenas vídeo 128×128 por mais 2.5 segundos antes de retomar a detecção de rostos. Evita enviar rostos em sequência rápida. |

**Diagrama de transições:**
```
paused ──[ESPAÇO]──► detecting ──[rosto encontrado]──► cooldown
  ▲                      ▲                                  │
  │                      │                            [2.5s]│
  └──[ESPAÇO]────────────┤                                  ▼
                         └──────────[2.5s]────────── grace
```

**FSM detalhada:** Ver [maquinas_de_estado.md](maquinas_de_estado.md), seção "FSM do script Python".

### 1.6 Recorte Central (Center Crop)

Para frames de vídeo (sem rosto detectado), a função `center_crop_1080` recorta um quadrado centralizado do frame capturado pela webcam. O tamanho do quadrado é `min(altura, largura, 1080)`. Isso garante que a imagem enviada seja quadrada e centrada, independente da proporção da câmera.

---

## 2. `haarcascade_frontalface_default.xml` — Classificador de Cascata

**Localização:** Raiz do projeto (`Lab---IA/haarcascade_frontalface_default.xml`)

**Finalidade:** Arquivo XML contendo o modelo pré-treinado de cascata de Haar para detecção frontal de rostos. Usado pelo OpenCV (`cv2.CascadeClassifier`) dentro do script Python.

**Origem:** Distribuído como parte do pacote OpenCV. O arquivo foi copiado para a raiz do projeto para garantir que funcione independentemente da instalação do OpenCV no sistema.

**Tamanho:** Aproximadamente 930 KB.

**Uso no código:** O script tenta resolver o caminho do classificador em várias localizações, priorizando o arquivo local do projeto:
1. `Lab---IA/haarcascade_frontalface_default.xml` (prioridade)
2. `cv2.data/haarcascade_frontalface_default.xml`
3. Caminhos do sistema Linux (`/usr/share/opencv4/...`)
