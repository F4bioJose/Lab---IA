// ==============================================================================
// Módulo: weights_shared_rom
// Descrição: Memória ROM (Read-Only Memory) estática que carrega os pesos
//            e os vieses treinados do arquivo .mif (Memory Initialization File).
//            Fornece acesso de leitura para a pipeline da CNN.
// ==============================================================================
module weights_shared_rom #(
    parameter MEM_FILE_MIF = "modulos_verilog/weights_all.mif",
    parameter CONV_FILTERS = 4,
    parameter CONV_KERNEL = 9,
    parameter DENSE_SIZE = 900,
    parameter DENSE_CLASSES = 7
)(
    input wire [9:0] dense_addr,
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

    localparam integer CONV_WEIGHT_TOTAL = CONV_FILTERS * CONV_KERNEL;
    localparam integer CONV_BIAS_BASE = CONV_WEIGHT_TOTAL;
    localparam integer DENSE_BASE = CONV_WEIGHT_TOTAL + CONV_FILTERS;
    localparam integer DENSE_WEIGHT_TOTAL = DENSE_SIZE * DENSE_CLASSES;
    localparam integer DENSE_BIAS_BASE = DENSE_BASE + DENSE_WEIGHT_TOTAL;
    localparam integer TOTAL_WORDS = DENSE_BIAS_BASE + DENSE_CLASSES;

    // Inicialização do array de memória com a diretiva de síntese
    (* ram_init_file = "modulos_verilog/weights_all.mif" *) reg signed [7:0] mem [0:TOTAL_WORDS-1];

    integer i;

    integer file, r;
    reg [8*100:1] line;
    integer addr;
    reg [7:0] data_val;

    initial begin
        for (i = 0; i < TOTAL_WORDS; i = i + 1) begin
            mem[i] = 8'sd0;
        end
        // As ferramentas de Síntese em hardware utilizarão a diretiva ram_init_file automaticamente
        // Em tempo de simulação, o bloco abaixo lê e faz o parse do arquivo .mif manualmente
        `ifndef ALTERA_RESERVED_QIS
        file = $fopen(MEM_FILE_MIF, "r");
        if (file) begin
            while (!$feof(file)) begin
                r = $fgets(line, file);
                if (r > 0) begin
                    if ($sscanf(line, "%d : %h;", addr, data_val) == 2) begin
                        if (addr < TOTAL_WORDS) begin
                            mem[addr] = data_val;
                        end
                    end
                end
            end
            $fclose(file);
        end else begin
            $display("Warning: Could not open %s", MEM_FILE_MIF);
        end
        `endif
    end

    genvar k;
    generate
        // Mapeamento Estático Combinacional: Extrai diretamente os pesos 
        // da convolução das respectivas posições base na ROM.
        for (k = 0; k < CONV_KERNEL; k = k + 1) begin : conv_map
            assign conv_w0[k] = mem[(0 * CONV_KERNEL) + k];
            assign conv_w1[k] = mem[(1 * CONV_KERNEL) + k];
            assign conv_w2[k] = mem[(2 * CONV_KERNEL) + k];
            assign conv_w3[k] = mem[(3 * CONV_KERNEL) + k];
        end
    endgenerate

    // Mapeamento Estático Combinacional: Extrai os vieses (bias) da convolução
    assign conv_b0 = mem[CONV_BIAS_BASE + 0];
    assign conv_b1 = mem[CONV_BIAS_BASE + 1];
    assign conv_b2 = mem[CONV_BIAS_BASE + 2];
    assign conv_b3 = mem[CONV_BIAS_BASE + 3];

    genvar c;
    generate
        // Mapeamento Dinâmico Combinacional: Extrai os pesos da camada densa
        // dependendo do endereço (dense_addr) solicitado iterativamente pela rede.
        for (c = 0; c < DENSE_CLASSES; c = c + 1) begin : dense_map
            assign dense_w[c] = mem[DENSE_BASE + (c * DENSE_SIZE) + dense_addr];
            assign dense_b[c] = mem[DENSE_BIAS_BASE + c];
        end
    endgenerate

endmodule
