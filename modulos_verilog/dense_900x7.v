`timescale 1ns/1ps

// ==============================================================================
// Módulo: dense_900x7_scores
// Descrição: Camada densa (Fully Connected) que recebe o mapa final achatado 
//            (900 elementos) e gera as predições de probabilidade brutas 
//            (scores) para as 7 classes da rede neural de forma acumulativa.
// ==============================================================================
module dense_900x7_scores #(
    parameter integer INPUT_SIZE = 900,
    parameter integer OUTPUT_CLASSES = 7
)(
    input wire clk,
    input wire rst,
    input wire signed [15:0] x_in,
    input wire signed [7:0] w_in [0:OUTPUT_CLASSES-1],
    input wire signed [7:0] bias_in [0:OUTPUT_CLASSES-1],
    input wire valid_in,
    output reg signed [15:0] scores [0:OUTPUT_CLASSES-1],
    output reg valid_out,
    output reg done
);

    // Limites de saturação para o formato de ponto fixo de destino (Q2.14)
    localparam signed [47:0] SAT_MAX_Q2_14 = 48'sd32767;
    localparam signed [47:0] SAT_MIN_Q2_14 = -48'sd32768;

    reg [9:0] sample_count;
    reg signed [47:0] acc_q3_21 [0:OUTPUT_CLASSES-1];

    reg signed [23:0] product_q3_21 [0:OUTPUT_CLASSES-1];
    reg signed [47:0] full_sum_q3_21 [0:OUTPUT_CLASSES-1];
    reg signed [47:0] linear_q2_14 [0:OUTPUT_CLASSES-1];

    integer c;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sample_count <= 10'd0;
            valid_out <= 1'b0;
            done <= 1'b0;
            for (c = 0; c < OUTPUT_CLASSES; c = c + 1) begin
                acc_q3_21[c] <= 48'sd0;
                scores[c] <= 16'sd0;
            end
        end else begin
            valid_out <= 1'b0;
            done <= 1'b0;

            if (valid_in) begin
                // Fase 1: Multiplicação paralela do pixel atual pelos pesos de todas as 7 classes
                for (c = 0; c < OUTPUT_CLASSES; c = c + 1) begin
                    product_q3_21[c] = $signed(x_in) * $signed(w_in[c]);
                end

                // Fase 2: Verifica se é o último pixel da imagem/mapa achatado
                if (sample_count == INPUT_SIZE - 1) begin
                    for (c = 0; c < OUTPUT_CLASSES; c = c + 1) begin
                        // Adiciona o último produto e o bias, ambos com extensão de sinal apropriada
                        full_sum_q3_21[c] = acc_q3_21[c]
                            + $signed({{24{product_q3_21[c][23]}}, product_q3_21[c]})
                            + $signed({{26{bias_in[c][7]}}, bias_in[c], 14'b0});

                        // Conversão de formato (Right shift para compensar as casas decimais)
                        linear_q2_14[c] = full_sum_q3_21[c] >>> 7;

                        // Fase 3: Saturação final dos scores das classes para prevenir overflow
                        if (linear_q2_14[c] > SAT_MAX_Q2_14) begin
                            scores[c] <= 16'sd32767;
                        end else if (linear_q2_14[c] < SAT_MIN_Q2_14) begin
                            scores[c] <= 16'sh8000;
                        end else begin
                            scores[c] <= linear_q2_14[c][15:0];
                        end

                        acc_q3_21[c] <= 48'sd0;
                    end

                    sample_count <= 10'd0;
                    valid_out <= 1'b1;
                    done <= 1'b1;
                end else begin
                    sample_count <= sample_count + 10'd1;
                    for (c = 0; c < OUTPUT_CLASSES; c = c + 1) begin
                        acc_q3_21[c] <= acc_q3_21[c]
                            + $signed({{24{product_q3_21[c][23]}}, product_q3_21[c]});
                    end
                end
            end
        end
    end

endmodule
