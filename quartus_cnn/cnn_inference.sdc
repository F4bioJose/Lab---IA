# ==============================================================================
# Timing Constraints — Projeto Unificado: CNN + VGA (DE2-115)
# ==============================================================================

# Clock principal: 50 MHz (período = 20 ns)
create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]

# Deriva automaticamente os clocks gerados pelo PLL (clk_25mhz = 25 MHz)
derive_pll_clocks

# Aplica incerteza de clock (jitter)
derive_clock_uncertainty

# UART RXD é um sinal assíncrono — não aplicar constraints de timing
set_false_path -from [get_ports UART_RXD]

# KEY é entrada assíncrona (botão com debounce por Schmitt Trigger)
set_false_path -from [get_ports KEY]

# LEDs são saídas lentas — relaxar constraints
set_false_path -to [get_ports LEDG*]

# VGA: saídas são amostradas pelo DAC na borda do VGA_CLK (gerado pelo FPGA).
# Como o VGA_CLK é gerado por DDR a partir do mesmo PLL, as saídas VGA_R/G/B
# precisam atender timing relativo ao clk_25mhz (tratado automaticamente pelo
# derive_pll_clocks). Adicionamos false paths apenas para os sinais de controle
# VGA que possuem requisitos relaxados.
set_false_path -to [get_ports VGA_SYNC_N]
