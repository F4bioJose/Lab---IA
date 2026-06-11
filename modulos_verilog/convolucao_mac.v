// ==============================================================================
// Módulo: conv_4_filters_relu_window
// Descrição: Núcleo de convolução que processa paralelamente 4 filtros de 
//            tamanho 3x3 sobre a janela de imagem atual. Aplica também a função 
//            de ativação ReLU e previne overflows (saturação).
// ==============================================================================
module conv_4_filters_relu_window (
    input wire clk,
    input wire rst,
    input wire window_valid,
    input wire [7:0] win_data [0:8],
    input wire signed [7:0] w_f0 [0:8],
    input wire signed [7:0] w_f1 [0:8],
    input wire signed [7:0] w_f2 [0:8],
    input wire signed [7:0] w_f3 [0:8],
    input wire signed [7:0] b_f0,
    input wire signed [7:0] b_f1,
    input wire signed [7:0] b_f2,
    input wire signed [7:0] b_f3,
    output reg valid_out,
    output reg signed [15:0] out_f0,
    output reg signed [15:0] out_f1,
    output reg signed [15:0] out_f2,
    output reg signed [15:0] out_f3
);

    // Fios combinacionais para armazenar os cálculos MAC (Multiply-Accumulate)
    wire signed [19:0] mac_0;
    wire signed [19:0] mac_1;
    wire signed [19:0] mac_2;
    wire signed [19:0] mac_3;

    // Cálculo MAC Filtro 0: Produto escalar da janela 3x3 com os pesos + viés (bias)
    assign mac_0 = ($signed(win_data[0]) * w_f0[0]) +
                   ($signed(win_data[1]) * w_f0[1]) +
                   ($signed(win_data[2]) * w_f0[2]) +
                   ($signed(win_data[3]) * w_f0[3]) +
                   ($signed(win_data[4]) * w_f0[4]) +
                   ($signed(win_data[5]) * w_f0[5]) +
                   ($signed(win_data[6]) * w_f0[6]) +
                   ($signed(win_data[7]) * w_f0[7]) +
                   ($signed(win_data[8]) * w_f0[8]) +
                   $signed({{5{b_f0[7]}}, b_f0, 7'b0});

    assign mac_1 = ($signed(win_data[0]) * w_f1[0]) +
                   ($signed(win_data[1]) * w_f1[1]) +
                   ($signed(win_data[2]) * w_f1[2]) +
                   ($signed(win_data[3]) * w_f1[3]) +
                   ($signed(win_data[4]) * w_f1[4]) +
                   ($signed(win_data[5]) * w_f1[5]) +
                   ($signed(win_data[6]) * w_f1[6]) +
                   ($signed(win_data[7]) * w_f1[7]) +
                   ($signed(win_data[8]) * w_f1[8]) +
                   $signed({{5{b_f1[7]}}, b_f1, 7'b0});

    assign mac_2 = ($signed(win_data[0]) * w_f2[0]) +
                   ($signed(win_data[1]) * w_f2[1]) +
                   ($signed(win_data[2]) * w_f2[2]) +
                   ($signed(win_data[3]) * w_f2[3]) +
                   ($signed(win_data[4]) * w_f2[4]) +
                   ($signed(win_data[5]) * w_f2[5]) +
                   ($signed(win_data[6]) * w_f2[6]) +
                   ($signed(win_data[7]) * w_f2[7]) +
                   ($signed(win_data[8]) * w_f2[8]) +
                   $signed({{5{b_f2[7]}}, b_f2, 7'b0});

    assign mac_3 = ($signed(win_data[0]) * w_f3[0]) +
                   ($signed(win_data[1]) * w_f3[1]) +
                   ($signed(win_data[2]) * w_f3[2]) +
                   ($signed(win_data[3]) * w_f3[3]) +
                   ($signed(win_data[4]) * w_f3[4]) +
                   ($signed(win_data[5]) * w_f3[5]) +
                   ($signed(win_data[6]) * w_f3[6]) +
                   ($signed(win_data[7]) * w_f3[7]) +
                   ($signed(win_data[8]) * w_f3[8]) +
                   $signed({{5{b_f3[7]}}, b_f3, 7'b0});

    // Função para aplicar Ativação ReLU e Saturação (Clamping) no formato ponto fixo
    function automatic signed [15:0] relu_sat;
        input signed [19:0] value;
        begin
            // Caso negativo (MSB == 1), zera o valor (Regra da ReLU)
            if (value[19]) begin
                relu_sat = 16'sd0;
            end else if (value > 20'sd32767) begin
                relu_sat = 16'sd32767;
            end else begin
                relu_sat = value[15:0];
            end
        end
    endfunction

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out <= 1'b0;
            out_f0 <= 16'sd0;
            out_f1 <= 16'sd0;
            out_f2 <= 16'sd0;
            out_f3 <= 16'sd0;
        end else begin
            // Sincroniza a validade do sinal de saída com a chegada de uma nova janela
            valid_out <= window_valid;
            if (window_valid) begin
                // Aplica a ReLU e satura os resultados MAC na saída registrada
                out_f0 <= relu_sat(mac_0);
                out_f1 <= relu_sat(mac_1);
                out_f2 <= relu_sat(mac_2);
                out_f3 <= relu_sat(mac_3);
            end else begin
                out_f0 <= 16'sd0;
                out_f1 <= 16'sd0;
                out_f2 <= 16'sd0;
                out_f3 <= 16'sd0;
            end
        end
    end

endmodule