# =============================================================================
# run_project.do
# Roda o testbench de captura de camadas da CNN (18 classes).
#
# Uso:
#   do scripts/run_project.do                          (imagem padrão)
#   do scripts/run_project.do inputs/frame0.txt        (imagem customizada)
#   do scripts/run_project.do inputs/frame0.txt comparacao/hw/frame0/
#
# Pré-requisito: executar compile_project.do antes (ou quando .v mudar)
# Execute SEMPRE a partir da raiz do projeto
# =============================================================================

# -----------------------------------------------------------------
# Parâmetros — alteráveis via argumento ou editando aqui
# -----------------------------------------------------------------
# Imagem de entrada (.txt com pixels em hex, um por linha, 1024 total)
if {[info exists 1] && $1 ne ""} {
    set img_file $1
} else {
    set img_file "inputs/teste2.txt"
}

# Diretório de saída para os .txt de cada camada
if {[info exists 2] && $2 ne ""} {
    set out_dir $2
} else {
    set out_dir "comparacao/hw/"
}

# -----------------------------------------------------------------
# Validações básicas
# -----------------------------------------------------------------
if {![file exists $img_file]} {
    error "[ERRO] Imagem não encontrada: $img_file\nUso: do scripts/run_project.do inputs/sua_imagem.txt"
}
if {![file exists "modulos_verilog/weights_all.hex"]} {
    error "[ERRO] Pesos não encontrados: modulos_verilog/weights_all.hex\nExecute primeiro: do scripts/compile_project.do"
}

# Cria diretório de saída se necessário
file mkdir $out_dir

echo "============================================================"
echo " CNN 18 Classes — Simulação tb_cnn_layer_capture"
echo "============================================================"
echo " Imagem  : $img_file"
echo " Saída   : $out_dir"
echo " Pesos   : modulos_verilog/weights_all.hex"
echo "============================================================"

# -----------------------------------------------------------------
# Inicia simulação em modo batch
# -----------------------------------------------------------------
vsim -c work.tb_cnn_layer_capture \
    +IMG=$img_file \
    +OUTDIR=$out_dir

run -all
quit -sim

# -----------------------------------------------------------------
# Resumo dos arquivos gerados
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
echo "   python scripts/extract_sw_activations.py --model modelo.h5 --image foto.jpg"
echo "   python scripts/compare_sw_hw.py --hw-dir $out_dir"
echo "============================================================"