# =============================================================================
# compile_project.do
# Compila todos os módulos Verilog da CNN (18 classes) para simulação.
#
# Uso (dentro do ModelSim/Questa GUI ou modo batch):
#   do scripts/compile_project.do
#
# Execute SEMPRE a partir da raiz do projeto (onde está a pasta modulos_verilog/)
# =============================================================================

# Fecha qualquer simulação em andamento
quit -sim

# -----------------------------------------------------------------
# 1. Converte weights_all.mif → weights_all.hex (formato $readmemh)
#    Roda inline via Tcl — não requer script externo
# -----------------------------------------------------------------
set mif_file  "modulos_verilog/weights_all.mif"
set hex_file  "modulos_verilog/weights_all.hex"

if {![file exists $mif_file]} {
    error "[ERRO] Arquivo de pesos não encontrado: $mif_file"
}

echo "--- Convertendo $mif_file -> $hex_file ---"
set fin  [open $mif_file r]
set fout [open $hex_file  w]
set n 0
while {[gets $fin line] >= 0} {
    # Linha de dado: "address : HH ;" (ADDRESS_RADIX=DEC, DATA_RADIX=HEX)
    if {[regexp {^\s*\d+\s*:\s*([0-9A-Fa-f]+)\s*;} $line -> hexval]} {
        puts $fout $hexval
        incr n
    }
}
close $fin
close $fout
echo "--- $n palavras gravadas em $hex_file ---"

# -----------------------------------------------------------------
# 2. Cria/limpa biblioteca de trabalho
# -----------------------------------------------------------------
if {[file exists work]} { vdel -lib work -all }
vlib work
vmap work work

# -----------------------------------------------------------------
# 3. Compila módulos em ordem de dependência
# -----------------------------------------------------------------
set src "modulos_verilog"

vlog -sv -work work \
    $src/uart_rx.v             \
    $src/framebuffer_32x32.v   \
    $src/line_buffer_32x32.v   \
    $src/convolucao_mac.v      \
    $src/max_pooling_design.v  \
    $src/flatten.v             \
    $src/weights_shared_rom.v  \
    $src/dense_900x18.v        \
    $src/argmax_threshold_18.v \
    $src/cnn_top.v             \
    $src/tb_cnn_layer_capture.v

echo "============================================================"
echo " Compilação concluída — 0 erros esperados"
echo " Execute: do scripts/run_project.do"
echo " Ou com imagem customizada:"
echo "   do scripts/run_project.do inputs/outra_imagem.txt"
echo "============================================================"