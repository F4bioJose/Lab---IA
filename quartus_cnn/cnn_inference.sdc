# ==============================================================================
# Timing Constraints — Inferência CNN (DE2-115)
# ==============================================================================

# Clock principal: 50 MHz (período = 20 ns)
create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]

# Aplica incerteza de clock (jitter)
derive_clock_uncertainty

# UART RXD é um sinal assíncrono — não aplicar constraints de timing
set_false_path -from [get_ports UART_RXD]

# KEYs são entradas assíncronas (botões com debounce)
set_false_path -from [get_ports KEY*]

# LEDs são saídas lentas — relaxar constraints
set_false_path -to [get_ports LEDG*]
