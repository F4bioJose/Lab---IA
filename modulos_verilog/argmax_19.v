// ==============================================================================
// Módulo: argmax_19
// Descrição: Recebe os scores das 19 classes produzidos pela camada densa,
//            encontra o índice (classe) com a maior pontuação (argmax) e
//            sinaliza "desconhecido" caso a classe vencedora seja a classe 0.
//
// Mapeamento de Classes (ordem Keras — string sort):
//   0  → Desconhecido (classe nativa da rede, treinada com Softmax)
//   1  → Igor
//   2  → Joao
//   3  → Jose Henrique
//   4  → Julia
//   5  → Lucio
//   6  → Naira
//   7  → Rafael
//   8  → Samuel
//   9  → Yuri
//   10 → Anna Carol
//   11 → Bruno
//   12 → Diego
//   13 → Eduardo
//   14 → Fabio
//   15 → Felipe
//   16 → Gabriel
//   17 → Horacio
//   18 → Hugo
//
// Lógica de Decisão:
//   - NÃO há threshold de confiança — a decisão é exclusivamente pelo argmax
//   - Se max_idx == 0  → unknown = 1 (pessoa não reconhecida)
//   - Se max_idx != 0  → unknown = 0, class_id = max_idx
//
// Saídas:
//   - class_id [4:0]: 0–18 = classe identificada (0 = desconhecido)
//   - unknown: 1 quando a rede prediz a classe "Desconhecido" (índice 0)
//   - max_score: valor Q6.10 do maior score encontrado
// ==============================================================================
module argmax_19 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [15:0] scores [0:18],
    output reg         valid_out,
    output reg  [4:0]  class_id,   // 0 = Desconhecido; 1..18 = pessoa identificada
    output reg         unknown,
    output reg  signed [15:0] max_score
);

    reg signed [15:0] max_val;
    reg [4:0]         max_idx;

    // -------------------------------------------------------------------------
    // Busca do argmax: comparação em cascata explícita das 19 classes.
    // A cada pulso valid_in os 19 scores são comparados em paralelo/sequência.
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

                // Comparações explícitas, classe a classe (1..18)
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
                if (scores[18] > max_val) begin max_val = scores[18]; max_idx = 5'd18; end

                max_score <= max_val;
                class_id  <= max_idx;

                // Desconhecido: a rede nativa treinou a classe 0 como "Desconhecido"
                // Não há threshold — a decisão é puramente pelo argmax
                unknown <= (max_idx == 5'd0) ? 1'b1 : 1'b0;

                valid_out <= 1'b1;
            end
        end
    end

endmodule
