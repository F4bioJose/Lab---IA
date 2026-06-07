`timescale 1ns/1ps

// ==============================================================================
// Módulo: dense_900x19_scores
// Descrição: Camada densa (Fully Connected) que recebe o mapa final achatado
//            (900 elementos) e gera os scores brutos (logits em Q2.14) para
//            as 19 classes da rede neural de forma acumulativa.
//
// Mapeamento de Classes:
//   0 = Desconhecido (classe nativa), 1..18 = pessoas identificadas
//
// Aritmética:
//   - Entrada x_in: Q2.14 (16 bits com sinal)
//   - Pesos w_in:   INT8  (8 bits com sinal, quantizados Q1.7)
//   - Produto:      Q3.21 (24 bits, estendido para 48 para o acumulador)
//   - Bias:         INT8, escalado para Q2.14 (deslocado 14 bits)
//   - Saída scores: Q2.14 (16 bits com sinal, saturado em [-32768, 32767])
// ==============================================================================
module dense_900x19_scores #(
    parameter integer INPUT_SIZE    = 900,
    parameter integer OUTPUT_CLASSES = 19
)(
    input  wire        clk,
    input  wire        rst,
    input  wire signed [15:0] x_in,
    input  wire signed [7:0]  w_in    [0:OUTPUT_CLASSES-1],
    input  wire signed [7:0]  bias_in [0:OUTPUT_CLASSES-1],
    input  wire        valid_in,
    output reg  signed [15:0] scores  [0:OUTPUT_CLASSES-1],
    output reg         valid_out,
    output reg         done
);

    // Limites de saturação Q2.14
    localparam signed [47:0] SAT_MAX = 48'sd32767;
    localparam signed [47:0] SAT_MIN = -48'sd32768;

    // Contador de amostras processadas
    reg [9:0] sample_count;

    // Acumuladores Q3.21 por classe (48 bits para evitar overflow durante acúmulo)
    reg signed [47:0] acc [ 0:18];

    // Wires intermediários para produtos e soma final por classe
    wire signed [23:0] prod [ 0:18];
    wire signed [47:0] fsum [ 0:18];
    wire signed [47:0] lval [ 0:18];

    // -------------------------------------------------------------------------
    // Produtos combinacionais: x_in × w_in[c] → Q3.21 (24 bits)
    // -------------------------------------------------------------------------
    assign prod[ 0] = $signed(x_in) * $signed(w_in[ 0]);
    assign prod[ 1] = $signed(x_in) * $signed(w_in[ 1]);
    assign prod[ 2] = $signed(x_in) * $signed(w_in[ 2]);
    assign prod[ 3] = $signed(x_in) * $signed(w_in[ 3]);
    assign prod[ 4] = $signed(x_in) * $signed(w_in[ 4]);
    assign prod[ 5] = $signed(x_in) * $signed(w_in[ 5]);
    assign prod[ 6] = $signed(x_in) * $signed(w_in[ 6]);
    assign prod[ 7] = $signed(x_in) * $signed(w_in[ 7]);
    assign prod[ 8] = $signed(x_in) * $signed(w_in[ 8]);
    assign prod[ 9] = $signed(x_in) * $signed(w_in[ 9]);
    assign prod[10] = $signed(x_in) * $signed(w_in[10]);
    assign prod[11] = $signed(x_in) * $signed(w_in[11]);
    assign prod[12] = $signed(x_in) * $signed(w_in[12]);
    assign prod[13] = $signed(x_in) * $signed(w_in[13]);
    assign prod[14] = $signed(x_in) * $signed(w_in[14]);
    assign prod[15] = $signed(x_in) * $signed(w_in[15]);
    assign prod[16] = $signed(x_in) * $signed(w_in[16]);
    assign prod[17] = $signed(x_in) * $signed(w_in[17]);
    assign prod[18] = $signed(x_in) * $signed(w_in[18]);

    // -------------------------------------------------------------------------
    // Soma final (último sample): acc + produto + bias escalado
    // bias é INT8 escalado para Q2.14 → shift de 14 bits
    // -------------------------------------------------------------------------
    assign fsum[ 0] = acc[ 0] + $signed({{24{prod[ 0][23]}}, prod[ 0]}) + $signed({{26{bias_in[ 0][7]}}, bias_in[ 0], 14'b0});
    assign fsum[ 1] = acc[ 1] + $signed({{24{prod[ 1][23]}}, prod[ 1]}) + $signed({{26{bias_in[ 1][7]}}, bias_in[ 1], 14'b0});
    assign fsum[ 2] = acc[ 2] + $signed({{24{prod[ 2][23]}}, prod[ 2]}) + $signed({{26{bias_in[ 2][7]}}, bias_in[ 2], 14'b0});
    assign fsum[ 3] = acc[ 3] + $signed({{24{prod[ 3][23]}}, prod[ 3]}) + $signed({{26{bias_in[ 3][7]}}, bias_in[ 3], 14'b0});
    assign fsum[ 4] = acc[ 4] + $signed({{24{prod[ 4][23]}}, prod[ 4]}) + $signed({{26{bias_in[ 4][7]}}, bias_in[ 4], 14'b0});
    assign fsum[ 5] = acc[ 5] + $signed({{24{prod[ 5][23]}}, prod[ 5]}) + $signed({{26{bias_in[ 5][7]}}, bias_in[ 5], 14'b0});
    assign fsum[ 6] = acc[ 6] + $signed({{24{prod[ 6][23]}}, prod[ 6]}) + $signed({{26{bias_in[ 6][7]}}, bias_in[ 6], 14'b0});
    assign fsum[ 7] = acc[ 7] + $signed({{24{prod[ 7][23]}}, prod[ 7]}) + $signed({{26{bias_in[ 7][7]}}, bias_in[ 7], 14'b0});
    assign fsum[ 8] = acc[ 8] + $signed({{24{prod[ 8][23]}}, prod[ 8]}) + $signed({{26{bias_in[ 8][7]}}, bias_in[ 8], 14'b0});
    assign fsum[ 9] = acc[ 9] + $signed({{24{prod[ 9][23]}}, prod[ 9]}) + $signed({{26{bias_in[ 9][7]}}, bias_in[ 9], 14'b0});
    assign fsum[10] = acc[10] + $signed({{24{prod[10][23]}}, prod[10]}) + $signed({{26{bias_in[10][7]}}, bias_in[10], 14'b0});
    assign fsum[11] = acc[11] + $signed({{24{prod[11][23]}}, prod[11]}) + $signed({{26{bias_in[11][7]}}, bias_in[11], 14'b0});
    assign fsum[12] = acc[12] + $signed({{24{prod[12][23]}}, prod[12]}) + $signed({{26{bias_in[12][7]}}, bias_in[12], 14'b0});
    assign fsum[13] = acc[13] + $signed({{24{prod[13][23]}}, prod[13]}) + $signed({{26{bias_in[13][7]}}, bias_in[13], 14'b0});
    assign fsum[14] = acc[14] + $signed({{24{prod[14][23]}}, prod[14]}) + $signed({{26{bias_in[14][7]}}, bias_in[14], 14'b0});
    assign fsum[15] = acc[15] + $signed({{24{prod[15][23]}}, prod[15]}) + $signed({{26{bias_in[15][7]}}, bias_in[15], 14'b0});
    assign fsum[16] = acc[16] + $signed({{24{prod[16][23]}}, prod[16]}) + $signed({{26{bias_in[16][7]}}, bias_in[16], 14'b0});
    assign fsum[17] = acc[17] + $signed({{24{prod[17][23]}}, prod[17]}) + $signed({{26{bias_in[17][7]}}, bias_in[17], 14'b0});
    assign fsum[18] = acc[18] + $signed({{24{prod[18][23]}}, prod[18]}) + $signed({{26{bias_in[18][7]}}, bias_in[18], 14'b0});

    // Conversão Q3.21 → Q2.14: right shift aritmético de 7 bits
    assign lval[ 0] = fsum[ 0] >>> 7;
    assign lval[ 1] = fsum[ 1] >>> 7;
    assign lval[ 2] = fsum[ 2] >>> 7;
    assign lval[ 3] = fsum[ 3] >>> 7;
    assign lval[ 4] = fsum[ 4] >>> 7;
    assign lval[ 5] = fsum[ 5] >>> 7;
    assign lval[ 6] = fsum[ 6] >>> 7;
    assign lval[ 7] = fsum[ 7] >>> 7;
    assign lval[ 8] = fsum[ 8] >>> 7;
    assign lval[ 9] = fsum[ 9] >>> 7;
    assign lval[10] = fsum[10] >>> 7;
    assign lval[11] = fsum[11] >>> 7;
    assign lval[12] = fsum[12] >>> 7;
    assign lval[13] = fsum[13] >>> 7;
    assign lval[14] = fsum[14] >>> 7;
    assign lval[15] = fsum[15] >>> 7;
    assign lval[16] = fsum[16] >>> 7;
    assign lval[17] = fsum[17] >>> 7;
    assign lval[18] = fsum[18] >>> 7;

    // =========================================================================
    // Lógica sequencial de acumulação
    // =========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sample_count <= 10'd0;
            valid_out    <= 1'b0;
            done         <= 1'b0;
            acc[ 0] <= 48'sd0; acc[ 1] <= 48'sd0; acc[ 2] <= 48'sd0;
            acc[ 3] <= 48'sd0; acc[ 4] <= 48'sd0; acc[ 5] <= 48'sd0;
            acc[ 6] <= 48'sd0; acc[ 7] <= 48'sd0; acc[ 8] <= 48'sd0;
            acc[ 9] <= 48'sd0; acc[10] <= 48'sd0; acc[11] <= 48'sd0;
            acc[12] <= 48'sd0; acc[13] <= 48'sd0; acc[14] <= 48'sd0;
            acc[15] <= 48'sd0; acc[16] <= 48'sd0; acc[17] <= 48'sd0;
            acc[18] <= 48'sd0;
            scores[ 0] <= 16'sd0; scores[ 1] <= 16'sd0; scores[ 2] <= 16'sd0;
            scores[ 3] <= 16'sd0; scores[ 4] <= 16'sd0; scores[ 5] <= 16'sd0;
            scores[ 6] <= 16'sd0; scores[ 7] <= 16'sd0; scores[ 8] <= 16'sd0;
            scores[ 9] <= 16'sd0; scores[10] <= 16'sd0; scores[11] <= 16'sd0;
            scores[12] <= 16'sd0; scores[13] <= 16'sd0; scores[14] <= 16'sd0;
            scores[15] <= 16'sd0; scores[16] <= 16'sd0; scores[17] <= 16'sd0;
            scores[18] <= 16'sd0;
        end else begin
            valid_out <= 1'b0;
            done      <= 1'b0;

            if (valid_in) begin
                // -------------------------------------------------------
                // Último sample: finaliza acumulação, aplica bias e satura
                // -------------------------------------------------------
                if (sample_count == INPUT_SIZE - 1) begin
                    // Saturação e armazenamento dos 19 scores
                    scores[ 0] <= (lval[ 0] > SAT_MAX) ? 16'sd32767 : (lval[ 0] < SAT_MIN) ? 16'sh8000 : lval[ 0][15:0];
                    scores[ 1] <= (lval[ 1] > SAT_MAX) ? 16'sd32767 : (lval[ 1] < SAT_MIN) ? 16'sh8000 : lval[ 1][15:0];
                    scores[ 2] <= (lval[ 2] > SAT_MAX) ? 16'sd32767 : (lval[ 2] < SAT_MIN) ? 16'sh8000 : lval[ 2][15:0];
                    scores[ 3] <= (lval[ 3] > SAT_MAX) ? 16'sd32767 : (lval[ 3] < SAT_MIN) ? 16'sh8000 : lval[ 3][15:0];
                    scores[ 4] <= (lval[ 4] > SAT_MAX) ? 16'sd32767 : (lval[ 4] < SAT_MIN) ? 16'sh8000 : lval[ 4][15:0];
                    scores[ 5] <= (lval[ 5] > SAT_MAX) ? 16'sd32767 : (lval[ 5] < SAT_MIN) ? 16'sh8000 : lval[ 5][15:0];
                    scores[ 6] <= (lval[ 6] > SAT_MAX) ? 16'sd32767 : (lval[ 6] < SAT_MIN) ? 16'sh8000 : lval[ 6][15:0];
                    scores[ 7] <= (lval[ 7] > SAT_MAX) ? 16'sd32767 : (lval[ 7] < SAT_MIN) ? 16'sh8000 : lval[ 7][15:0];
                    scores[ 8] <= (lval[ 8] > SAT_MAX) ? 16'sd32767 : (lval[ 8] < SAT_MIN) ? 16'sh8000 : lval[ 8][15:0];
                    scores[ 9] <= (lval[ 9] > SAT_MAX) ? 16'sd32767 : (lval[ 9] < SAT_MIN) ? 16'sh8000 : lval[ 9][15:0];
                    scores[10] <= (lval[10] > SAT_MAX) ? 16'sd32767 : (lval[10] < SAT_MIN) ? 16'sh8000 : lval[10][15:0];
                    scores[11] <= (lval[11] > SAT_MAX) ? 16'sd32767 : (lval[11] < SAT_MIN) ? 16'sh8000 : lval[11][15:0];
                    scores[12] <= (lval[12] > SAT_MAX) ? 16'sd32767 : (lval[12] < SAT_MIN) ? 16'sh8000 : lval[12][15:0];
                    scores[13] <= (lval[13] > SAT_MAX) ? 16'sd32767 : (lval[13] < SAT_MIN) ? 16'sh8000 : lval[13][15:0];
                    scores[14] <= (lval[14] > SAT_MAX) ? 16'sd32767 : (lval[14] < SAT_MIN) ? 16'sh8000 : lval[14][15:0];
                    scores[15] <= (lval[15] > SAT_MAX) ? 16'sd32767 : (lval[15] < SAT_MIN) ? 16'sh8000 : lval[15][15:0];
                    scores[16] <= (lval[16] > SAT_MAX) ? 16'sd32767 : (lval[16] < SAT_MIN) ? 16'sh8000 : lval[16][15:0];
                    scores[17] <= (lval[17] > SAT_MAX) ? 16'sd32767 : (lval[17] < SAT_MIN) ? 16'sh8000 : lval[17][15:0];
                    scores[18] <= (lval[18] > SAT_MAX) ? 16'sd32767 : (lval[18] < SAT_MIN) ? 16'sh8000 : lval[18][15:0];
                    // Limpa acumuladores para o próximo frame
                    acc[ 0] <= 48'sd0; acc[ 1] <= 48'sd0; acc[ 2] <= 48'sd0;
                    acc[ 3] <= 48'sd0; acc[ 4] <= 48'sd0; acc[ 5] <= 48'sd0;
                    acc[ 6] <= 48'sd0; acc[ 7] <= 48'sd0; acc[ 8] <= 48'sd0;
                    acc[ 9] <= 48'sd0; acc[10] <= 48'sd0; acc[11] <= 48'sd0;
                    acc[12] <= 48'sd0; acc[13] <= 48'sd0; acc[14] <= 48'sd0;
                    acc[15] <= 48'sd0; acc[16] <= 48'sd0; acc[17] <= 48'sd0;
                    acc[18] <= 48'sd0;
                    sample_count <= 10'd0;
                    valid_out    <= 1'b1;
                    done         <= 1'b1;
                end else begin
                    // -------------------------------------------------------
                    // Samples intermediários: acumula produto (sem bias ainda)
                    // -------------------------------------------------------
                    acc[ 0] <= acc[ 0] + $signed({{24{prod[ 0][23]}}, prod[ 0]});
                    acc[ 1] <= acc[ 1] + $signed({{24{prod[ 1][23]}}, prod[ 1]});
                    acc[ 2] <= acc[ 2] + $signed({{24{prod[ 2][23]}}, prod[ 2]});
                    acc[ 3] <= acc[ 3] + $signed({{24{prod[ 3][23]}}, prod[ 3]});
                    acc[ 4] <= acc[ 4] + $signed({{24{prod[ 4][23]}}, prod[ 4]});
                    acc[ 5] <= acc[ 5] + $signed({{24{prod[ 5][23]}}, prod[ 5]});
                    acc[ 6] <= acc[ 6] + $signed({{24{prod[ 6][23]}}, prod[ 6]});
                    acc[ 7] <= acc[ 7] + $signed({{24{prod[ 7][23]}}, prod[ 7]});
                    acc[ 8] <= acc[ 8] + $signed({{24{prod[ 8][23]}}, prod[ 8]});
                    acc[ 9] <= acc[ 9] + $signed({{24{prod[ 9][23]}}, prod[ 9]});
                    acc[10] <= acc[10] + $signed({{24{prod[10][23]}}, prod[10]});
                    acc[11] <= acc[11] + $signed({{24{prod[11][23]}}, prod[11]});
                    acc[12] <= acc[12] + $signed({{24{prod[12][23]}}, prod[12]});
                    acc[13] <= acc[13] + $signed({{24{prod[13][23]}}, prod[13]});
                    acc[14] <= acc[14] + $signed({{24{prod[14][23]}}, prod[14]});
                    acc[15] <= acc[15] + $signed({{24{prod[15][23]}}, prod[15]});
                    acc[16] <= acc[16] + $signed({{24{prod[16][23]}}, prod[16]});
                    acc[17] <= acc[17] + $signed({{24{prod[17][23]}}, prod[17]});
                    acc[18] <= acc[18] + $signed({{24{prod[18][23]}}, prod[18]});
                    sample_count <= sample_count + 10'd1;
                end
            end
        end
    end

endmodule
