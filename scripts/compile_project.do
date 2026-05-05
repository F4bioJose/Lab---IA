quit -sim

vlib work
vmap work work

vlog -sv -work work modulos_verilog/*.v

echo "--- Compilacao Finalizada! ---"
echo "para rodar testbench: vsim work.testbench"
echo "run -all"