#!/bin/bash
# =============================================================================
# run_pipeline.sh
# Roda todo o pipeline (Conversão Hex -> Questa -> Keras -> Comparação)
#
# Uso:
#   ./scripts/run_pipeline.sh <nome_da_imagem.jpg>
# Exemplo:
#   ./scripts/run_pipeline.sh anna.jpg
#   ./scripts/run_pipeline.sh anna
# =============================================================================

if [ -z "$1" ]; then
    echo "Uso: $0 <imagem.jpg>"
    echo "Exemplo: $0 anna.jpg"
    exit 1
fi

INPUT="$1"

# Tenta encontrar a imagem no diretório atual ou em inputs/
if [ -f "$INPUT" ]; then
    IMG_PATH="$INPUT"
elif [ -f "inputs/$INPUT" ]; then
    IMG_PATH="inputs/$INPUT"
elif [ -f "inputs/$INPUT.jpg" ]; then
    IMG_PATH="inputs/$INPUT.jpg"
else
    echo "[ERRO] Imagem não encontrada: $INPUT"
    exit 1
fi

BASENAME=$(basename "$IMG_PATH")
BASENAME="${BASENAME%.*}"

echo "============================================================"
echo " 1. Convertendo imagem para HEX"
echo "============================================================"
python3 scripts/image_to_hex.py "$IMG_PATH"
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha ao converter imagem."
    exit 1
fi

HEX_FILE="inputs/imgs_hex/${BASENAME}_hex.txt"

echo "============================================================"
echo " 2. Rodando Testbench (Modelsim/Questa)"
echo "============================================================"
vsim -c -do "do scripts/run_project.do $HEX_FILE"
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha na simulação."
    exit 1
fi

echo "============================================================"
echo " 3. Extraindo Ativações de Software (Keras)"
echo "============================================================"
python3 scripts/extract_sw_activations.py --model tiny_cnn_multiclasse.h5 --image "$IMG_PATH"
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha na extração Keras."
    exit 1
fi

echo "============================================================"
echo " 4. Comparando SW vs HW"
echo "============================================================"
python3 scripts/compare_sw_hw.py --image "$IMG_PATH"
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha na comparação."
    exit 1
fi

echo "============================================================"
echo "[OK] PIPELINE CONCLUÍDO COM SUCESSO!"
echo " Resultados salvos em: comparacao/plots/$BASENAME/"
echo "============================================================"
