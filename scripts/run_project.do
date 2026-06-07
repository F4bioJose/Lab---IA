# =============================================================================
# run_project.do
# Compila e roda o testbench de captura de camadas da CNN (19 classes).
#
# Este script unifica compilação + simulação num fluxo único.
#
# Uso:
#   do scripts/run_project.do                          (imagem padrão)
#   do scripts/run_project.do inputs/frame0_hex.txt    (imagem customizada)
#   do scripts/run_project.do inputs/frame0_hex.txt comparacao/hw/frame0/
#
# Execute SEMPRE a partir da raiz do projeto
# =============================================================================

# -----------------------------------------------------------------
# 1. Parâmetros — alteráveis via argumento ou editando aqui
# -----------------------------------------------------------------
if {[info exists 1] && $1 ne ""} {
    set img_file $1
} else {
    set img_file "inputs/teste2.txt"
}

if {[info exists 2] && $2 ne ""} {
    set out_dir $2
} else {
    set out_dir "comparacao/hw/"
}

# -----------------------------------------------------------------
# 2. Validações básicas
# -----------------------------------------------------------------
set hex_file "modulos_verilog/weights_all.hex"

if {![file exists $img_file]} {
    error "[ERRO] Imagem não encontrada: $img_file\nUso: do scripts/run_project.do inputs/sua_imagem.txt"
}
if {![file exists $hex_file]} {
    error "[ERRO] Pesos não encontrados: $hex_file\nGere com: python -c \"import sys; sys.path.insert(0,'rede_pipeline'); from src.export_mif import MIFExporter; ...\""
}

file mkdir $out_dir

echo $img_file
echo $out_dir

# -----------------------------------------------------------------
# 3. Compilação dos módulos Verilog
# -----------------------------------------------------------------
echo "============================================================"
echo " FASE 1: Compilação dos módulos Verilog"
echo "============================================================"

quit -sim

if {[file exists work]} { vdel -lib work -all }
vlib work
vmap work work

set src "modulos_verilog"

vlog -sv -work work \
    $src/uart_rx.v             \
    $src/framebuffer_32x32.v   \
    $src/line_buffer_32x32.v   \
    $src/convolucao_mac.v      \
    $src/max_pooling_design.v  \
    $src/flatten.v             \
    $src/weights_shared_rom.v  \
    $src/dense_900x19.v        \
    $src/argmax_19.v           \
    $src/cnn_top.v             \
    $src/tb_cnn_layer_capture.v

echo "[OK] Compilação concluída com sucesso."

# -----------------------------------------------------------------
# 4. Simulação
# -----------------------------------------------------------------
echo "============================================================"
echo " FASE 2: Simulação — CNN 19 Classes"
echo " Classe 0=Desconhecido (nativo) | Sem threshold"
echo "============================================================"
echo " Imagem  : $img_file"
echo " Saída   : $out_dir"
echo " Pesos   : $hex_file"
echo "============================================================"

vsim -c work.tb_cnn_layer_capture \
    +IMG=$img_file \
    +OUTDIR=$out_dir

run -all
quit -sim

# -----------------------------------------------------------------
# 5. Resumo dos arquivos gerados
# -----------------------------------------------------------------
echo ""
echo "============================================================"
echo " Arquivos gerados em '$out_dir':"
echo "============================================================"
foreach f [glob -nocomplain ${out_dir}*.txt] {
    set lines [expr {[exec wc -l < $f] - 1}]
    echo "   [file tail $f]  ($lines amostras)"
}
echo "============================================================"
echo " Próximo passo (comparação SW/HW):"
echo "   python scripts/extract_sw_activations.py --model tiny_cnn_multiclasse.h5 --image foto.jpg"
echo "   python scripts/compare_sw_hw.py --hw-dir $out_dir"
echo "============================================================"