module top_vga (
    input  wire CLOCK_50,
    input  wire [0:0] KEY,
	 input  wire [7:0]  sram_data_out,  // Dado lido da SRAM (rosto)
	 input  wire access_granted, // 1 = Liberado, 0 = Negado
	 
    output wire VGA_HS,
    output wire VGA_VS,
    output wire VGA_CLK,
    output wire VGA_BLANK_N,
    output wire VGA_SYNC_N,
    output reg  [7:0] VGA_R,
    output reg  [7:0] VGA_G,
    output reg  [7:0] VGA_B,
	 output wire [19:0] sram_address   // Endereço para ler a SRAM
);

    wire clk_25mhz;
    wire video_on_orig;
	 wire hsync_orig;
	 wire vsync_orig;
    wire [9:0] pixel_x;
    wire [9:0] pixel_y;
    wire reset = ~KEY[0]; 

    assign VGA_CLK = clk_25mhz;
    assign VGA_SYNC_N = 1'b0; 

    vga_pll meu_pll (
        .inclk0(CLOCK_50),
        .c0(clk_25mhz)
    );

    vga_sync controlador_de_varredura (
        .clk(clk_25mhz),
        .rst(reset),       // <-- Atualizado: Conectando o reset virtual ao reset do controlador
        .hsync(hsync_orig), // mudança por conta do atraso de 1 clk 
        .vsync(vsync_orig), // mudança por conta do atraso de 1 clk 
        .pixel_x(pixel_x),
        .pixel_y(pixel_y),
        .video_on(video_on_orig)
    );
	 
	 
	 // LOGICA DE MAPEAMENTO
	 
	 localparam X_START = 10'd304; 
    localparam Y_START = 10'd224; 
     
    wire in_window = (pixel_x >= X_START) && (pixel_x < X_START + 32) && 
    (pixel_y >= Y_START) && (pixel_y < Y_START + 32); 
 
    assign sram_address = in_window ?  
    (((pixel_y - Y_START) << 5) + (pixel_x - X_START)) : 20'd0; 
 
    // Janela do Texto (128x32 Abaixo da Câmera) 
    localparam TXT_X_START = 10'd256; 
    localparam TXT_Y_START = 10'd266; 
     
    wire in_text_window = (pixel_x >= TXT_X_START) && (pixel_x < TXT_X_START + 128) &&
	 (pixel_y >= TXT_Y_START) && (pixel_y < TXT_Y_START + 32); 
 
    wire [11:0] text_address = in_text_window ?  
    (((pixel_y - TXT_Y_START) << 7) + (pixel_x - TXT_X_START)) : 12'd0; 
										 
	 // ========================================================================= 
    // INSTÂNCIAS DAS MEMÓRIAS ROM DO TEXTO 
    // ========================================================================= 
    wire pixel_liberado; 
    wire pixel_negado; 
 
    rom_liberado inst_liberado ( 
        .address (text_address), 
        .clock (clk_25mhz), 
        .q (pixel_liberado) 
    ); 
 
    rom_negado inst_negado ( 
        .address (text_address), 
        .clock (clk_25mhz), 
        .q (pixel_negado) 
    ); 
	 
	 
	 // ========================================================================= 
    // COMPENSAÇÃO DE LATÊNCIA (Atraso de 1 ciclo) 
    // ========================================================================= 
	 reg vga_hs_delayed; 
    reg vga_vs_delayed; 
    reg video_on_delayed; 
    reg in_window_delayed; 
    reg in_text_window_delayed; 
 
    always @(posedge clk_25mhz or posedge reset) begin 
        if (reset) begin 
            vga_hs_delayed <= 1'b1; 
            vga_vs_delayed <= 1'b1; 
            video_on_delayed <= 1'b0; 
            in_window_delayed <= 1'b0; 
            in_text_window_delayed <= 1'b0; 
        end else begin 
            vga_hs_delayed <= hsync_orig; 
            vga_vs_delayed <= vsync_orig; 
            video_on_delayed <= video_on_orig; 
            in_window_delayed <= in_window; 
            in_text_window_delayed <= in_text_window; 
        end 
    end 
 
    assign VGA_HS = vga_hs_delayed; 
    assign VGA_VS = vga_vs_delayed; 
    assign VGA_BLANK_N = video_on_delayed; 
		
		// Quadrado verde com fundo cinza
    always @(*) begin
        if (!video_on_delayed) begin
            VGA_R = 8'd0; VGA_G = 8'd0; VGA_B = 8'd0;
        end
		   else if (in_window_delayed) begin 
            // Mostra o rosto lido da SRAM 
            {VGA_R, VGA_G, VGA_B}= {sram_data_out, sram_data_out, sram_data_out}; 
        end 
        else if (in_text_window_delayed) begin
            if (access_granted) begin 
                {VGA_R, VGA_G, VGA_B} = pixel_liberado ? 24'h00FF00 : 24'h000000; // concatenacao
            end else begin
			end
                {VGA_R, VGA_G, VGA_B} = pixel_negado ? 24'hFF0000 : 24'h000000; 
            end 
            else begin
                VGA_R = 8'h10; VGA_G = 8'h10; VGA_B = 8'h10;
            end
        end
endmodule