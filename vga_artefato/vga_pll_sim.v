// ==============================================================================
// Stub de simulação do PLL para Icarus Verilog
// Substitui o vga_pll Altera MegaFunction por um divisor simples (50→25 MHz)
// ==============================================================================
module vga_pll (
    input  wire areset,
    input  wire inclk0,
    output wire c0,
    output wire locked
);
    // Divisor por 2: 50 MHz → 25 MHz (simplificação para simulação)
    reg clk_div = 0;

    always @(posedge inclk0 or posedge areset) begin
        if (areset)
            clk_div <= 1'b0;
        else
            clk_div <= ~clk_div;
    end

    assign c0     = clk_div;
    assign locked = ~areset;
endmodule
