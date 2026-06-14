// ==============================================================================
// Módulo: argmax_19
// Descrição: Recebe os scores das 19 classes produzidos pela camada densa,
//            encontra o índice (classe) com a maior pontuação (argmax) e
//            sinaliza "desconhecido" caso a classe vencedora seja a classe 0.
//
// Mapeamento de Classes na Saída (após soma de +1):
//   0  → Vazio (estado ocioso / sem inferência)
//   1  → Desconhecido (classe nativa da rede, treinada com Softmax, internamente 0)
//   2  → Igor
//   3  → Joao
//   4  → Jose Henrique
//   5  → Julia
//   6  → Lucio
//   7  → Naira
//   8  → Rafael
//   9  → Samuel
//   10 → Yuri
//   11 → Anna Carol
//   12 → Bruno
//   13 → Diego
//   14 → Eduardo
//   15 → Fabio
//   16 → Felipe
//   17 → Gabriel
//   18 → Horacio
//   19 → Hugo
//
// Lógica de Decisão:
//   - NÃO há threshold de confiança — a decisão é exclusivamente pelo argmax
//   - Se max_idx == 0  → unknown = 1 (pessoa não reconhecida)
//   - Se max_idx != 0  → unknown = 0, class_id = max_idx
//
// Saídas:
//   - class_id [4:0]: 0 = Vazio; 1 = Desconhecido; 2..19 = pessoa identificada
//   - unknown: 1 quando a rede prediz a classe "Desconhecido" (índice 0)
//   - max_score: valor Q6.10 do maior score encontrado
// ==============================================================================
module argmax_19 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [15:0] scores [0:18],
    output reg         valid_out,
    output reg  [4:0]  class_id,   // 0 = Vazio; 1 = Desconhecido; 2..19 = pessoa identificada
    output reg         unknown,
    output reg  signed [15:0] max_score
);

    reg signed [15:0] max_val;
    reg [4:0]         max_idx;
    reg [4:0]         current_idx;
    reg [1:0]         state;

    localparam IDLE      = 2'd0;
    localparam COMPARING = 2'd1;
    localparam DONE      = 2'd2;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out   <= 1'b0;
            class_id    <= 5'd0; // Inicializa como Vazio
            unknown     <= 1'b0;
            max_score   <= 16'sd0;
            max_val     <= 16'sd0;
            max_idx     <= 5'd0;
            current_idx <= 5'd0;
            state       <= IDLE;
        end else begin
            valid_out <= 1'b0; // Pulso único por default

            case (state)
                IDLE: begin
                    if (valid_in) begin
                        max_val     <= scores[0];
                        max_idx     <= 5'd0;
                        current_idx <= 5'd1;
                        state       <= COMPARING;
                    end
                end

                COMPARING: begin
                    if (scores[current_idx] > max_val) begin
                        max_val <= scores[current_idx];
                        max_idx <= current_idx;
                    end
                    
                    if (current_idx == 5'd18) begin
                        state <= DONE;
                    end else begin
                        current_idx <= current_idx + 5'd1;
                    end
                end

                DONE: begin
                    max_score <= max_val;
                    class_id  <= max_idx + 5'd1;
                    
                    // Desconhecido: a rede nativa treinou a classe 0 como "Desconhecido"
                    // Não há threshold — a decisão é puramente pelo argmax
                    unknown <= (max_idx == 5'd0) ? 1'b1 : 1'b0;
                    
                    valid_out <= 1'b1;
                    state     <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule
