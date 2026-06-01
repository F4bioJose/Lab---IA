// ==============================================================================
// Módulo: weights_shared_rom
// Descrição: Memória ROM (Read-Only Memory) estática que carrega os pesos
//            e os vieses treinados do arquivo .mif (Memory Initialization File).
//            Fornece acesso de leitura para a pipeline da CNN.
// ==============================================================================
module weights_shared_rom #(
    parameter MEM_FILE_MIF = "modulos_verilog/weights_all.mif",
    parameter integer CONV_FILTERS = 4,
    parameter integer CONV_KERNEL = 9,
    parameter integer DENSE_SIZE = 900,
    parameter integer DENSE_CLASSES = 7,
    parameter integer TOTAL_WORDS = (CONV_FILTERS * CONV_KERNEL)
        + CONV_FILTERS
        + (DENSE_SIZE * DENSE_CLASSES)
        + DENSE_CLASSES,
    parameter integer ADDR_WIDTH = (TOTAL_WORDS > 1) ? $clog2(TOTAL_WORDS) : 1
)(
    input wire [ADDR_WIDTH-1:0] dense_addr,
    output wire signed [7:0] conv_w0 [0:CONV_KERNEL-1],
    output wire signed [7:0] conv_w1 [0:CONV_KERNEL-1],
    output wire signed [7:0] conv_w2 [0:CONV_KERNEL-1],
    output wire signed [7:0] conv_w3 [0:CONV_KERNEL-1],
    output wire signed [7:0] conv_b0,
    output wire signed [7:0] conv_b1,
    output wire signed [7:0] conv_b2,
    output wire signed [7:0] conv_b3,
    output wire signed [7:0] dense_w [0:DENSE_CLASSES-1],
    output wire signed [7:0] dense_b [0:DENSE_CLASSES-1]
);

    // Splitted ROM arrays to avoid giant multiplexer synthesis bottlenecks
    reg signed [7:0] mem_conv [0:39];
    reg signed [7:0] mem_dense_b [0:6];
    
    reg signed [7:0] mem_dense_0 [0:899];
    reg signed [7:0] mem_dense_1 [0:899];
    reg signed [7:0] mem_dense_2 [0:899];
    reg signed [7:0] mem_dense_3 [0:899];
    reg signed [7:0] mem_dense_4 [0:899];
    reg signed [7:0] mem_dense_5 [0:899];
    reg signed [7:0] mem_dense_6 [0:899];

    initial begin
`ifdef ALTERA_RESERVED_QIS
        $readmemh("../modulos_verilog/weights_conv.hex", mem_conv);
        $readmemh("../modulos_verilog/weights_dense_b.hex", mem_dense_b);
        $readmemh("../modulos_verilog/weights_dense_0.hex", mem_dense_0);
        $readmemh("../modulos_verilog/weights_dense_1.hex", mem_dense_1);
        $readmemh("../modulos_verilog/weights_dense_2.hex", mem_dense_2);
        $readmemh("../modulos_verilog/weights_dense_3.hex", mem_dense_3);
        $readmemh("../modulos_verilog/weights_dense_4.hex", mem_dense_4);
        $readmemh("../modulos_verilog/weights_dense_5.hex", mem_dense_5);
        $readmemh("../modulos_verilog/weights_dense_6.hex", mem_dense_6);
`else
        $readmemh("modulos_verilog/weights_conv.hex", mem_conv);
        $readmemh("modulos_verilog/weights_dense_b.hex", mem_dense_b);
        $readmemh("modulos_verilog/weights_dense_0.hex", mem_dense_0);
        $readmemh("modulos_verilog/weights_dense_1.hex", mem_dense_1);
        $readmemh("modulos_verilog/weights_dense_2.hex", mem_dense_2);
        $readmemh("modulos_verilog/weights_dense_3.hex", mem_dense_3);
        $readmemh("modulos_verilog/weights_dense_4.hex", mem_dense_4);
        $readmemh("modulos_verilog/weights_dense_5.hex", mem_dense_5);
        $readmemh("modulos_verilog/weights_dense_6.hex", mem_dense_6);
`endif
    end

    genvar k;
    generate
        // Mapeamento Estático Combinacional
        for (k = 0; k < CONV_KERNEL; k = k + 1) begin : conv_map
            assign conv_w0[k] = mem_conv[(0 * CONV_KERNEL) + k];
            assign conv_w1[k] = mem_conv[(1 * CONV_KERNEL) + k];
            assign conv_w2[k] = mem_conv[(2 * CONV_KERNEL) + k];
            assign conv_w3[k] = mem_conv[(3 * CONV_KERNEL) + k];
        end
    endgenerate

    assign conv_b0 = mem_conv[36];
    assign conv_b1 = mem_conv[37];
    assign conv_b2 = mem_conv[38];
    assign conv_b3 = mem_conv[39];

    assign dense_w[0] = mem_dense_0[dense_addr];
    assign dense_w[1] = mem_dense_1[dense_addr];
    assign dense_w[2] = mem_dense_2[dense_addr];
    assign dense_w[3] = mem_dense_3[dense_addr];
    assign dense_w[4] = mem_dense_4[dense_addr];
    assign dense_w[5] = mem_dense_5[dense_addr];
    assign dense_w[6] = mem_dense_6[dense_addr];

    assign dense_b[0] = mem_dense_b[0];
    assign dense_b[1] = mem_dense_b[1];
    assign dense_b[2] = mem_dense_b[2];
    assign dense_b[3] = mem_dense_b[3];
    assign dense_b[4] = mem_dense_b[4];
    assign dense_b[5] = mem_dense_b[5];
    assign dense_b[6] = mem_dense_b[6];

endmodule
