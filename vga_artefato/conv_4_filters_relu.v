module conv_4_filters_relu #(
    parameter IMG_WIDTH = 32
)(
    input wire clk,
    input wire rst,
    
    // Pixel de entrada
    input wire signed [7:0] pixel_in,
    input wire valid_in,
    
    // --- Pesos para o FILTRO 0 ---
    input wire signed [7:0] f0_w00, f0_w01, f0_w02,
    input wire signed [7:0] f0_w10, f0_w11, f0_w12,
    input wire signed [7:0] f0_w20, f0_w21, f0_w22,
    input wire signed [7:0] f0_bias,

    // --- Pesos para o FILTRO 1 ---
    input wire signed [7:0] f1_w00, f1_w01, f1_w02,
    input wire signed [7:0] f1_w10, f1_w11, f1_w12,
    input wire signed [7:0] f1_w20, f1_w21, f1_w22,
    input wire signed [7:0] f1_bias,

    // --- Pesos para o FILTRO 2 ---
    input wire signed [7:0] f2_w00, f2_w01, f2_w02,
    input wire signed [7:0] f2_w10, f2_w11, f2_w12,
    input wire signed [7:0] f2_w20, f2_w21, f2_w22,
    input wire signed [7:0] f2_bias,

    // --- Pesos para o FILTRO 3 ---
    input wire signed [7:0] f3_w00, f3_w01, f3_w02,
    input wire signed [7:0] f3_w10, f3_w11, f3_w12,
    input wire signed [7:0] f3_w20, f3_w21, f3_w22,
    input wire signed [7:0] f3_bias,
    
    // Saídas processadas
    output reg signed [15:0] out_f0,
    output reg signed [15:0] out_f1,
    output reg signed [15:0] out_f2,
    output reg signed [15:0] out_f3,
    output reg valid_out
);

    // Line buffers para as duas linhas anteriores 
    reg signed [7:0] row1 [0:IMG_WIDTH-1];
    reg signed [7:0] row2 [0:IMG_WIDTH-1];
    
    // 3x3 registradores da janela 3x3
    reg signed [7:0] p11, p12, p13;
    reg signed [7:0] p21, p22, p23;
    reg signed [7:0] p31, p32, p33;
    
    // Registradores para sincronismo
    reg [5:0] in_x, in_y;         // Monitora o pixel que está chegando no input
    reg [5:0] window_x, window_y; // Monitora o pixel inferior direito da janela
    
    // Saídas dos MACs 20bits para prevenir overshoot devido a soma da multiplicação entre 18 valores 8bit

    wire signed [19:0] mac_0, mac_1, mac_2, mac_3;
    
    assign mac_0 = (p11*f0_w00) + (p12*f0_w01) + (p13*f0_w02) +
                   (p21*f0_w10) + (p22*f0_w11) + (p23*f0_w12) +
                   (p31*f0_w20) + (p32*f0_w21) + (p33*f0_w22) + f0_bias;

    assign mac_1 = (p11*f1_w00) + (p12*f1_w01) + (p13*f1_w02) +
                   (p21*f1_w10) + (p22*f1_w11) + (p23*f1_w12) +
                   (p31*f1_w20) + (p32*f1_w21) + (p33*f1_w22) + f1_bias;

    assign mac_2 = (p11*f2_w00) + (p12*f2_w01) + (p13*f2_w02) +
                   (p21*f2_w10) + (p22*f2_w11) + (p23*f2_w12) +
                   (p31*f2_w20) + (p32*f2_w21) + (p33*f2_w22) + f2_bias;

    assign mac_3 = (p11*f3_w00) + (p12*f3_w01) + (p13*f3_w02) +
                   (p21*f3_w10) + (p22*f3_w11) + (p23*f3_w12) +
                   (p31*f3_w20) + (p32*f3_w21) + (p33*f3_w22) + f3_bias;
    integer i;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            in_x <= 0; 
            in_y <= 0;
            window_x <= 0; 
            window_y <= 0;
            valid_out <= 0;
            out_f0 <= 0; 
            out_f1 <= 0; 
            out_f2 <= 0; 
            out_f3 <= 0;

        end else if (valid_in) begin
            // 1. Entrada de dados nos line buffers
            row1[0] <= pixel_in;
            for (i=1; i<IMG_WIDTH; i=i+1) row1[i] <= row1[i-1];
            
            row2[0] <= row1[IMG_WIDTH-1];
            for (i=1; i<IMG_WIDTH; i=i+1) row2[i] <= row2[i-1];
            
            // 2. Janela 3x3 correndo da esquerda para a direita
            p11 <= p12; p12 <= p13; p13 <= row2[IMG_WIDTH-1];
            p21 <= p22; p22 <= p23; p23 <= row1[IMG_WIDTH-1];
            p31 <= p32; p32 <= p33; p33 <= pixel_in;
            
            // 3. Pegando coordenadas do pixel que entrou na janela
            window_x <= in_x;
            window_y <= in_y;
            
            if (in_x < IMG_WIDTH - 1) begin
                in_x <= in_x + 1;
            end else begin
                in_x <= 0;
                in_y <= in_y + 1;
            end
            
            // 4. Saída válida + saturação + ReLU
            // Janela 3x3 é válida quando pixel inferior direito X & Y >= 2
            if (window_x >= 2 && window_y >= 2) begin
                valid_out <= 1'b1;
                // ReLU: Se o MSB (15º bit) for 1, o número é negativo -> Output 0.
                // --- ReLU & Lógica de saturação para F0 ---
                if (mac_0[19] == 1'b1) begin
                    out_f0 <= 16'd0;                // ReLU: Número negativo -> Output 0
                end else if (mac_0 > 20'sd32767) begin
                    out_f0 <= 16'sd32767;           // Saturação: Se o valor for maior que o máximo permitido para 16 bits, limitar a 32767
                end else begin
                    out_f0 <= mac_0[15:0];          // Se não for negativo e não exceder o limite, passar os 16 bits inferiores
                end

                // --- ReLU & Lógica de saturação para F1 ---
                if (mac_1[19] == 1'b1) begin
                    out_f1 <= 16'd0;
                end else if (mac_1 > 20'sd32767) begin
                    out_f1 <= 16'sd32767;
                end else begin
                    out_f1 <= mac_1[15:0];
                end

                // --- ReLU & Lógica de saturação para F2 ---
                if (mac_2[19] == 1'b1) begin
                    out_f2 <= 16'd0;
                end else if (mac_2 > 20'sd32767) begin
                    out_f2 <= 16'sd32767;
                end else begin
                    out_f2 <= mac_2[15:0];
                end

                // --- ReLU & Lógica de saturação para F3 ---
                if (mac_3[19] == 1'b1) begin
                    out_f3 <= 16'd0;
                end else if (mac_3 > 20'sd32767) begin
                    out_f3 <= 16'sd32767;
                end else begin
                    out_f3 <= mac_3[15:0];
                end
            end else begin
                valid_out <= 1'b0;
            end
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule