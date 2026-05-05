// ==============================================================================
// Módulo: flatten
// Descrição: Responsável por serializar (achatar) mapas de características
//            multidimensionais em um array unidimensional. Este é um passo
//            essencial antes de passar os dados para a camada densa (fully connected).
// ==============================================================================
module flatten #(
    parameter DATA_WIDTH = 16,
    parameter CHANNELS = 4,
    parameter HEIGHT = 15,
    parameter WIDTH = 15
)(
    input wire clk,
    input wire rst,

    // Input data stream
    input wire [DATA_WIDTH-1:0] data_in,
    input wire valid_in,

    // Flatten output stream
    output reg [DATA_WIDTH-1:0] data_out,
    output reg valid_out,

    // Pulses high for one cycle on the final flattened element
    output reg done
);

    localparam integer TOTAL_SIZE = CHANNELS * HEIGHT * WIDTH;
    localparam integer COUNTER_WIDTH = (TOTAL_SIZE > 1) ? $clog2(TOTAL_SIZE) : 1;

    reg [COUNTER_WIDTH-1:0] counter;

    // Bloco principal de controle da serialização
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            counter   <= {COUNTER_WIDTH{1'b0}};
            data_out  <= {DATA_WIDTH{1'b0}};
            valid_out <= 1'b0;
            done      <= 1'b0;
        end else begin
            done <= 1'b0;

            if (valid_in) begin
                // Repassa o dado de entrada diretamente para a saída
                data_out  <= data_in;
                valid_out <= 1'b1;

                // Verifica se alcançou o último elemento do volume de dados
                if (counter == TOTAL_SIZE - 1) begin
                    counter <= {COUNTER_WIDTH{1'b0}};
                    done    <= 1'b1; // Sinaliza fim da serialização do frame atual
                end else begin
                    counter <= counter + 1'b1; // Incrementa a contagem de elementos
                end
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule
