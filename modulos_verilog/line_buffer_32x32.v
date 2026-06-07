// ==============================================================================
// Módulo: line_buffer_32x32
// Descrição: Buffer de linhas usado para armazenar as 2 últimas linhas completas 
//            da imagem de forma a permitir a extração contínua de janelas 3x3 
//            a cada ciclo de clock, otimizando o pipeline da convolução.
// ==============================================================================
module line_buffer_32x32 (
    input wire clk,
    input wire rst,
    input wire [7:0] pixel_in,
    input wire shift_en,

    output reg [7:0] win [0:8],
    output reg window_valid
);

    // Memórias para o histórico das duas últimas linhas. Essenciais para 
    // recriar as 3 linhas simultâneas (junto ao pixel de entrada)
    (* ramstyle = "no_rw_check, M9K" *) reg [7:0] row1 [0:31];
    (* ramstyle = "no_rw_check, M9K" *) reg [7:0] row2 [0:31];
    
    // Contadores de posição X e Y na imagem
    reg [5:0] in_x;
    reg [5:0] in_y;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            in_x <= 6'd0;
            in_y <= 6'd0;
            window_valid <= 1'b0;
        end else begin
            window_valid <= 1'b0;

            if (shift_en) begin
                // Atualiza as memórias de linha para a coluna 'in_x' atual
                row1[in_x] <= pixel_in;
                row2[in_x] <= row1[in_x];

                // Desloca o conteúdo das colunas anteriores da janela 3x3 (Shift Register)
                win[0] <= win[1];
                win[1] <= win[2];
                win[3] <= win[4];
                win[4] <= win[5];
                win[6] <= win[7];
                win[7] <= win[8];

                // Preenche a coluna mais à direita (índices 2, 5 e 8) da janela 3x3
                // com os dados novos da memória e o fluxo direto atual
                win[2] <= row2[in_x];
                win[5] <= row1[in_x];
                win[8] <= pixel_in;

                // Controle espacial bidimensional: Avança colunas e linhas
                if (in_x < 6'd31) begin
                    in_x <= in_x + 6'd1;
                end else begin
                    in_x <= 6'd0;
                    if (in_y < 6'd31) begin
                        in_y <= in_y + 6'd1;
                    end else begin
                        in_y <= 6'd0;
                    end
                end

                // Uma janela 3x3 completa e válida só existe a partir do pixel (2,2)
                if (in_x >= 6'd2 && in_y >= 6'd2) begin
                    window_valid <= 1'b1;
                end
            end
        end
    end

endmodule
