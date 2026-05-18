module vga_controller ( 
    input  wire        clk_50,         // Clock de 50MHz da DE2-115 
    input  wire        reset,          // Reset ativo em alto 
    input  wire [7:0]  sram_data_out,  // Dado lido da SRAM (rosto) 
    input  wire        access_granted, // 1 = Liberado, 0 = Negado 
     
    output wire        vga_hs,         // Sincronismo Horizontal 
    output wire        vga_vs,         // Sincronismo Vertical 
    output wire        vga_blank_n,    // Habilita vídeo pro DAC 
    output wire        vga_clk,        // Clock de 25MHz pro DAC 
     
    output reg  [7:0]  vga_r,          // Canal Vermelho 
    output reg  [7:0]  vga_g,          // Canal Verde 
    output reg  [7:0]  vga_b,          // Canal Azul 
     
    output wire [19:0] sram_address    // Endereço para ler a SRAM 
); 
 
    // ========================================================================= 
    // 1. DIVISOR DE CLOCK (50MHz -> 25MHz) 
    // ========================================================================= 
    reg clk_25 = 0; 
    always @(posedge clk_50) clk_25 <= ~clk_25; 
    assign vga_clk = clk_25; 
 
 
    // ========================================================================= 
    // 2. CONTADORES VGA (640x480 @ 60Hz) 
    // ========================================================================= 
    reg [9:0] h_count = 0; 
    reg [9:0] v_count = 0; 
 
    always @(posedge clk_25 or posedge reset) begin 
        if (reset) begin 
            h_count <= 0; 
            v_count <= 0; 
        end else begin 
            if (h_count == 799) begin 
                h_count <= 0; 
                if (v_count == 524) 
                    v_count <= 0; 
                else 
                    v_count <= v_count + 1; 
            end else begin 
                h_count <= h_count + 1; 
            end 
        end 
    end 
 
 
    // ========================================================================= 
    // 3. GERAÇÃO DE SINCRONISMO ORIGINAL 
    // ========================================================================= 
    wire vga_hs_orig    = ~(h_count >= 656 && h_count < 752); 
    wire vga_vs_orig    = ~(v_count >= 490 && v_count < 492); 
    wire video_on_orig  = (h_count < 640) && (v_count < 480); 
     
    wire [9:0] pixel_x = video_on_orig ? h_count : 10'd0; 
    wire [9:0] pixel_y = video_on_orig ? v_count : 10'd0; 
 
 
    // ========================================================================= 
    // 4. MAPEAMENTO DE ÁREAS E ENDEREÇOS 
    // ========================================================================= 
    // Janela da Câmera (32x32 no Centro) 
    localparam X_START = 10'd304; 
    localparam Y_START = 10'd224; 
     
    wire in_window = (pixel_x >= X_START) && (pixel_x < X_START + 32) && 
                     (pixel_y >= Y_START) && (pixel_y < Y_START + 32); 
 
    assign sram_address = in_window ?  
                          (((pixel_y - Y_START) << 5) + (pixel_x - X_START)) :  
                          20'd0; 
 
    // Janela do Texto (128x32 Abaixo da Câmera) 
    localparam TXT_X_START = 10'd256; 
    localparam TXT_Y_START = 10'd266; 
     
    wire in_text_window = (pixel_x >= TXT_X_START) && (pixel_x < TXT_X_START + 128) && 
                          (pixel_y >= TXT_Y_START) && (pixel_y < TXT_Y_START + 32); 
 
    wire [11:0] text_address = in_text_window ?  
                               (((pixel_y - TXT_Y_START) << 7) + (pixel_x - TXT_X_START)) :  
                               12'd0; 
 
 
    // ========================================================================= 
    // 5. INSTÂNCIAS DAS MEMÓRIAS ROM DO TEXTO 
    // ========================================================================= 
    wire pixel_liberado; 
    wire pixel_negado; 
 
    rom_liberado inst_liberado ( 
        .address (text_address), 
        .clock   (clk_25), 
        .q       (pixel_liberado) 
    ); 
 
    rom_negado inst_negado ( 
        .address (text_address), 
        .clock   (clk_25), 
        .q       (pixel_negado) 
    ); 
 
 
    // ========================================================================= 
    // 6. COMPENSAÇÃO DE LATÊNCIA (Atraso de 1 ciclo) 
    // ========================================================================= 
    reg vga_hs_delayed; 
    reg vga_vs_delayed; 
    reg video_on_delayed; 
    reg in_window_delayed; 
    reg in_text_window_delayed; 
 
    always @(posedge clk_25 or posedge reset) begin 
        if (reset) begin 
            vga_hs_delayed         <= 1'b1; 
            vga_vs_delayed         <= 1'b1; 
            video_on_delayed       <= 1'b0; 
            in_window_delayed      <= 1'b0; 
            in_text_window_delayed <= 1'b0; 
        end else begin 
            vga_hs_delayed         <= vga_hs_orig; 
            vga_vs_delayed         <= vga_vs_orig; 
            video_on_delayed       <= video_on_orig; 
            in_window_delayed      <= in_window; 
            in_text_window_delayed <= in_text_window; 
        end 
    end 
 
    assign vga_hs      = vga_hs_delayed; 
    assign vga_vs      = vga_vs_delayed; 
    assign vga_blank_n = video_on_delayed; 
 
 
    // ========================================================================= 
    // 7. GERAÇÃO DE CORES (MULTIPLEXADOR) 
    // ========================================================================= 
    always @(*) begin 
        if (!video_on_delayed) begin 
            {vga_r, vga_g, vga_b} = 24'h000000; // Fora da tela 
        end  
        else if (in_window_delayed) begin 
            // Mostra o rosto lido da SRAM 
            {vga_r, vga_g, vga_b} = {sram_data_out, sram_data_out, sram_data_out}; 
        end  
        else if (in_text_window_delayed) begin 
            // Mostra o texto 
            if (access_granted) begin 
                {vga_r, vga_g, vga_b} = pixel_liberado ? 24'h00FF00 : 24'h000000; 
            end else begin 
                {vga_r, vga_g, vga_b} = pixel_negado ? 24'hFF0000 : 24'h000000; 
            end 
        end  
        else begin 
            {vga_r, vga_g, vga_b} = 24'h101010; // Fundo cinza escuro 
        end 
    end 
 
endmodule