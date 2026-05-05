// ==============================================================================
// Módulo: argmax_threshold_7
// Descrição: Recebe os scores das 7 classes produzidos pela camada densa, 
//            encontra o índice (classe) com a maior probabilidade (argmax) 
//            e aplica um limiar de confiança (threshold).
// ==============================================================================
module argmax_threshold_7 #(
    // Limiar de aceitação em formato ponto fixo Q2.14 (ex: 0.7 de confiança)
    parameter signed [15:0] THRESH_Q2_14 = 16'sd11469
)(
    input wire clk,
    input wire rst,
    input wire valid_in,
    input wire signed [15:0] scores [0:6],
    output reg valid_out,
    output reg [2:0] class_id,
    output reg unknown,
    output reg signed [15:0] max_score
);

    integer i;
    reg signed [15:0] max_val;
    reg [2:0] max_idx;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out <= 1'b0;
            class_id <= 3'd0;
            unknown <= 1'b0;
            max_score <= 16'sd0;
        end else begin
            valid_out <= 1'b0;
            if (valid_in) begin
                // Inicializa o valor e índice máximos com a primeira classe (índice 0)
                max_val = scores[0];
                max_idx = 3'd0;
                
                // Itera sobre as demais 6 classes para encontrar o maior score absoluto
                for (i = 1; i < 7; i = i + 1) begin
                    if (scores[i] > max_val) begin
                        max_val = scores[i];
                        max_idx = i[2:0];
                    end
                end

                max_score <= max_val;
                
                // Avalia se o score máximo superou o limiar de aceitação predefinido
                if (max_val < THRESH_Q2_14) begin
                    class_id <= 3'd7; // ID 7 reservado para classificação incerta/desconhecida
                    unknown <= 1'b1;
                end else begin
                    class_id <= max_idx; // Confirma a classe de maior probabilidade
                    unknown <= 1'b0;
                end
                valid_out <= 1'b1;
            end
        end
    end

endmodule
