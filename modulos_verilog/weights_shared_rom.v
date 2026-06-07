// ==============================================================================
// Módulo: weights_shared_rom
// Descrição: Memória ROM (Read-Only Memory) estática que carrega os pesos
//            e os vieses treinados a partir de um único arquivo .hex.
//            Fornece acesso de leitura para o pipeline da CNN.
//
// *** CONFIGURAÇÃO ATUAL: 19 CLASSES ***
// Mapeamento de classes (ordem Keras — string sort das pastas do dataset):
//   0  = Desconhecido (classe nativa da rede)
//   1  = Igor         | 2  = Joao        | 3  = Jose Henrique
//   4  = Julia        | 5  = Lucio       | 6  = Naira
//   7  = Rafael       | 8  = Samuel      | 9  = Yuri
//   10 = Anna Carol   | 11 = Bruno       | 12 = Diego
//   13 = Eduardo      | 14 = Fabio       | 15 = Felipe
//   16 = Gabriel      | 17 = Horacio     | 18 = Hugo
//
// Estrutura do arquivo de pesos (tiny_cnn_multiclasse.h5 → all_weights.mif):
//   [0   .. 35]                          → pesos conv (4 filtros × 9 coefs = 36)
//   [36  .. 39]                          → biases conv (4)
//   [40  .. 40 + DENSE_SIZE*19 - 1]     → pesos densos (900 × 19 = 17100)
//   [17140 .. 17158]                     → biases densos (19)
//   Total = 36 + 4 + 17100 + 19 = 17159  ✓ (bate com DEPTH do .mif)
// ==============================================================================
module weights_shared_rom #(
    parameter MEM_FILE      = "modulos_verilog/weights_all.hex",
    parameter integer CONV_FILTERS  = 4,
    parameter integer CONV_KERNEL   = 9,
    parameter integer DENSE_SIZE    = 900,
    // 19 classes: 0=Desconhecido, 1..18=pessoas identificadas
    parameter integer DENSE_CLASSES = 19,
    parameter integer OFF_CONV_W     = 0,                                 // 0
    parameter integer OFF_CONV_B     = CONV_FILTERS * CONV_KERNEL,        // 36
    parameter integer OFF_DENSE_W    = OFF_CONV_B + CONV_FILTERS,         // 40
    parameter integer OFF_DENSE_B    = OFF_DENSE_W + DENSE_SIZE * DENSE_CLASSES, // 17140
    parameter integer TOTAL_WORDS    = OFF_DENSE_B + DENSE_CLASSES        // 17159
)(
    input  wire [13:0]              dense_addr,     // Endereço de 0 a 899 (entrada do flatten)
    // Pesos e biases da camada convolucional
    output wire signed [7:0]        conv_w0 [0:CONV_KERNEL-1],
    output wire signed [7:0]        conv_w1 [0:CONV_KERNEL-1],
    output wire signed [7:0]        conv_w2 [0:CONV_KERNEL-1],
    output wire signed [7:0]        conv_w3 [0:CONV_KERNEL-1],
    output wire signed [7:0]        conv_b0,
    output wire signed [7:0]        conv_b1,
    output wire signed [7:0]        conv_b2,
    output wire signed [7:0]        conv_b3,
    // Pesos e biases da camada densa (19 classes)
    output wire signed [7:0]        dense_w [0:DENSE_CLASSES-1],
    output wire signed [7:0]        dense_b [0:DENSE_CLASSES-1]
);

    // -------------------------------------------------------------------------
    // Array único de memória — inicializado a partir de um único arquivo .hex
    // -------------------------------------------------------------------------
    reg signed [7:0] mem [0:TOTAL_WORDS-1];

    initial begin
`ifdef ALTERA_RESERVED_QIS
        $readmemh({"../", MEM_FILE}, mem);
`else
        $readmemh(MEM_FILE, mem);
`endif
    end

    // -------------------------------------------------------------------------
    // Mapeamento estático dos pesos convolucionais (36 pesos + 4 biases)
    // Filtro k: mem[k*9 .. k*9+8]
    // -------------------------------------------------------------------------
    assign conv_w0[0] = mem[OFF_CONV_W +  0];
    assign conv_w0[1] = mem[OFF_CONV_W +  1];
    assign conv_w0[2] = mem[OFF_CONV_W +  2];
    assign conv_w0[3] = mem[OFF_CONV_W +  3];
    assign conv_w0[4] = mem[OFF_CONV_W +  4];
    assign conv_w0[5] = mem[OFF_CONV_W +  5];
    assign conv_w0[6] = mem[OFF_CONV_W +  6];
    assign conv_w0[7] = mem[OFF_CONV_W +  7];
    assign conv_w0[8] = mem[OFF_CONV_W +  8];

    assign conv_w1[0] = mem[OFF_CONV_W +  9];
    assign conv_w1[1] = mem[OFF_CONV_W + 10];
    assign conv_w1[2] = mem[OFF_CONV_W + 11];
    assign conv_w1[3] = mem[OFF_CONV_W + 12];
    assign conv_w1[4] = mem[OFF_CONV_W + 13];
    assign conv_w1[5] = mem[OFF_CONV_W + 14];
    assign conv_w1[6] = mem[OFF_CONV_W + 15];
    assign conv_w1[7] = mem[OFF_CONV_W + 16];
    assign conv_w1[8] = mem[OFF_CONV_W + 17];

    assign conv_w2[0] = mem[OFF_CONV_W + 18];
    assign conv_w2[1] = mem[OFF_CONV_W + 19];
    assign conv_w2[2] = mem[OFF_CONV_W + 20];
    assign conv_w2[3] = mem[OFF_CONV_W + 21];
    assign conv_w2[4] = mem[OFF_CONV_W + 22];
    assign conv_w2[5] = mem[OFF_CONV_W + 23];
    assign conv_w2[6] = mem[OFF_CONV_W + 24];
    assign conv_w2[7] = mem[OFF_CONV_W + 25];
    assign conv_w2[8] = mem[OFF_CONV_W + 26];

    assign conv_w3[0] = mem[OFF_CONV_W + 27];
    assign conv_w3[1] = mem[OFF_CONV_W + 28];
    assign conv_w3[2] = mem[OFF_CONV_W + 29];
    assign conv_w3[3] = mem[OFF_CONV_W + 30];
    assign conv_w3[4] = mem[OFF_CONV_W + 31];
    assign conv_w3[5] = mem[OFF_CONV_W + 32];
    assign conv_w3[6] = mem[OFF_CONV_W + 33];
    assign conv_w3[7] = mem[OFF_CONV_W + 34];
    assign conv_w3[8] = mem[OFF_CONV_W + 35];

    assign conv_b0 = mem[OFF_CONV_B + 0];
    assign conv_b1 = mem[OFF_CONV_B + 1];
    assign conv_b2 = mem[OFF_CONV_B + 2];
    assign conv_b3 = mem[OFF_CONV_B + 3];

    // -------------------------------------------------------------------------
    // Mapeamento dinâmico dos pesos densos — indexado por dense_addr (0..899)
    // Cada classe c ocupa mem[OFF_DENSE_W + c*900 .. OFF_DENSE_W + c*900 + 899]
    // Classe 0 = Desconhecido; Classes 1..18 = pessoas identificadas
    // -------------------------------------------------------------------------
    assign dense_w[ 0] = mem[OFF_DENSE_W +  0 * DENSE_SIZE + dense_addr];
    assign dense_w[ 1] = mem[OFF_DENSE_W +  1 * DENSE_SIZE + dense_addr];
    assign dense_w[ 2] = mem[OFF_DENSE_W +  2 * DENSE_SIZE + dense_addr];
    assign dense_w[ 3] = mem[OFF_DENSE_W +  3 * DENSE_SIZE + dense_addr];
    assign dense_w[ 4] = mem[OFF_DENSE_W +  4 * DENSE_SIZE + dense_addr];
    assign dense_w[ 5] = mem[OFF_DENSE_W +  5 * DENSE_SIZE + dense_addr];
    assign dense_w[ 6] = mem[OFF_DENSE_W +  6 * DENSE_SIZE + dense_addr];
    assign dense_w[ 7] = mem[OFF_DENSE_W +  7 * DENSE_SIZE + dense_addr];
    assign dense_w[ 8] = mem[OFF_DENSE_W +  8 * DENSE_SIZE + dense_addr];
    assign dense_w[ 9] = mem[OFF_DENSE_W +  9 * DENSE_SIZE + dense_addr];
    assign dense_w[10] = mem[OFF_DENSE_W + 10 * DENSE_SIZE + dense_addr];
    assign dense_w[11] = mem[OFF_DENSE_W + 11 * DENSE_SIZE + dense_addr];
    assign dense_w[12] = mem[OFF_DENSE_W + 12 * DENSE_SIZE + dense_addr];
    assign dense_w[13] = mem[OFF_DENSE_W + 13 * DENSE_SIZE + dense_addr];
    assign dense_w[14] = mem[OFF_DENSE_W + 14 * DENSE_SIZE + dense_addr];
    assign dense_w[15] = mem[OFF_DENSE_W + 15 * DENSE_SIZE + dense_addr];
    assign dense_w[16] = mem[OFF_DENSE_W + 16 * DENSE_SIZE + dense_addr];
    assign dense_w[17] = mem[OFF_DENSE_W + 17 * DENSE_SIZE + dense_addr];
    assign dense_w[18] = mem[OFF_DENSE_W + 18 * DENSE_SIZE + dense_addr];

    // -------------------------------------------------------------------------
    // Mapeamento estático dos biases densos (19 bytes, endereços fixos)
    // -------------------------------------------------------------------------
    assign dense_b[ 0] = mem[OFF_DENSE_B +  0];
    assign dense_b[ 1] = mem[OFF_DENSE_B +  1];
    assign dense_b[ 2] = mem[OFF_DENSE_B +  2];
    assign dense_b[ 3] = mem[OFF_DENSE_B +  3];
    assign dense_b[ 4] = mem[OFF_DENSE_B +  4];
    assign dense_b[ 5] = mem[OFF_DENSE_B +  5];
    assign dense_b[ 6] = mem[OFF_DENSE_B +  6];
    assign dense_b[ 7] = mem[OFF_DENSE_B +  7];
    assign dense_b[ 8] = mem[OFF_DENSE_B +  8];
    assign dense_b[ 9] = mem[OFF_DENSE_B +  9];
    assign dense_b[10] = mem[OFF_DENSE_B + 10];
    assign dense_b[11] = mem[OFF_DENSE_B + 11];
    assign dense_b[12] = mem[OFF_DENSE_B + 12];
    assign dense_b[13] = mem[OFF_DENSE_B + 13];
    assign dense_b[14] = mem[OFF_DENSE_B + 14];
    assign dense_b[15] = mem[OFF_DENSE_B + 15];
    assign dense_b[16] = mem[OFF_DENSE_B + 16];
    assign dense_b[17] = mem[OFF_DENSE_B + 17];
    assign dense_b[18] = mem[OFF_DENSE_B + 18];

endmodule
