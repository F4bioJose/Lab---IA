// ==============================================================================
// Módulo: argmax_threshold_18
// Descrição: Recebe os scores das 18 classes produzidos pela camada densa,
//            encontra o índice (classe) com a maior pontuação (argmax) e
//            aplica um limiar de confiança (threshold) configurável.
//
// Threshold:
//   - Expresso em formato ponto fixo Q2.14
//   - Padrão: 95% de confiança → 0.95 × 2^14 = 15564 (decimal) = 0x3CAC
//   - Altere o parâmetro THRESH_Q2_14 para mudar o limiar sem recompilar
//
// Saídas:
//   - class_id [4:0]: 0–17 = classe identificada; 18 = negado (abaixo do limiar)
//   - unknown: 1 quando a predição não atingiu o threshold
//   - max_score: valor Q2.14 do maior score encontrado
// ==============================================================================
module argmax_threshold_18 #(
    // Threshold de aceitação em Q2.14 (default = 95%)
    // Para alterar: 0.95 × 16384 = 15564
    // Exemplos: 90% → 14746 | 80% → 13107 | 70% → 11469
    parameter signed [15:0] THRESH_Q2_14 = 16'sd15564
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [15:0] scores [0:17],
    output reg         valid_out,
    output reg  [4:0]  class_id,   // 0-17 = classe válida; 18 = negado
    output reg         unknown,
    output reg  signed [15:0] max_score
);

    reg signed [15:0] max_val;
    reg [4:0]         max_idx;

    // -------------------------------------------------------------------------
    // Busca do argmax: comparação em cascata explícita (sem for loops)
    // A cada pulso valid_in os 18 scores são comparados em paralelo/sequência.
    // -------------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out <= 1'b0;
            class_id  <= 5'd0;
            unknown   <= 1'b0;
            max_score <= 16'sd0;
        end else begin
            valid_out <= 1'b0;

            if (valid_in) begin
                // Inicializa com a classe 0
                max_val = scores[0];
                max_idx = 5'd0;

                // Comparações explícitas, classe a classe
                if (scores[ 1] > max_val) begin max_val = scores[ 1]; max_idx = 5'd1;  end
                if (scores[ 2] > max_val) begin max_val = scores[ 2]; max_idx = 5'd2;  end
                if (scores[ 3] > max_val) begin max_val = scores[ 3]; max_idx = 5'd3;  end
                if (scores[ 4] > max_val) begin max_val = scores[ 4]; max_idx = 5'd4;  end
                if (scores[ 5] > max_val) begin max_val = scores[ 5]; max_idx = 5'd5;  end
                if (scores[ 6] > max_val) begin max_val = scores[ 6]; max_idx = 5'd6;  end
                if (scores[ 7] > max_val) begin max_val = scores[ 7]; max_idx = 5'd7;  end
                if (scores[ 8] > max_val) begin max_val = scores[ 8]; max_idx = 5'd8;  end
                if (scores[ 9] > max_val) begin max_val = scores[ 9]; max_idx = 5'd9;  end
                if (scores[10] > max_val) begin max_val = scores[10]; max_idx = 5'd10; end
                if (scores[11] > max_val) begin max_val = scores[11]; max_idx = 5'd11; end
                if (scores[12] > max_val) begin max_val = scores[12]; max_idx = 5'd12; end
                if (scores[13] > max_val) begin max_val = scores[13]; max_idx = 5'd13; end
                if (scores[14] > max_val) begin max_val = scores[14]; max_idx = 5'd14; end
                if (scores[15] > max_val) begin max_val = scores[15]; max_idx = 5'd15; end
                if (scores[16] > max_val) begin max_val = scores[16]; max_idx = 5'd16; end
                if (scores[17] > max_val) begin max_val = scores[17]; max_idx = 5'd17; end

                max_score <= max_val;

                // Aplica threshold: se o maior score não atingiu o limiar, rejeita
                if (max_val < THRESH_Q2_14) begin
                    class_id  <= 5'd18;  // Código reservado: classe "negada"
                    unknown   <= 1'b1;
                end else begin
                    class_id  <= max_idx;
                    unknown   <= 1'b0;
                end

                valid_out <= 1'b1;
            end
        end
    end

endmodule
