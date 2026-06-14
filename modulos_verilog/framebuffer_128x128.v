// ==============================================================================
// Módulo: framebuffer_128x128
// Descrição: Memória de buffer para armazenar uma imagem de vídeo 128×128
//            (16384 pixels grayscale). Fornece porta de escrita (UART) e
//            porta de leitura dedicada ao controlador VGA.
//            Não possui porta de leitura CNN — frames de vídeo não passam
//            pela inferência.
//
// Uso: Recebe frames de vídeo (webcam sem detecção de rosto) para exibição
//      direta no VGA com upscale 3× (128×3 = 384 pixels).
// ==============================================================================
module framebuffer_128x128 (
    input wire clk,
    input wire rst,

    // Porta de escrita (UART → vídeo)
    input wire        wr_en,
    input wire [13:0] wr_addr,   // 0–16383
    input wire [7:0]  wr_data,

    // Porta de leitura VGA
    input wire        vga_rd_en,
    input wire [13:0] vga_rd_addr,  // 0–16383
    output reg [7:0]  vga_rd_data,

    // Sinalização de frame completo
    output reg frame_ready
);

    // 128×128 = 16384 bytes → inferidos como M9K
    (* ramstyle = "no_rw_check, M9K" *) reg [7:0] mem [0:16383];

    reg [13:0] clear_addr;
    reg        clearing;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            frame_ready <= 1'b0;
            clearing    <= 1'b1;
            clear_addr  <= 14'd0;
        end else begin
            if (wr_en && wr_addr == 14'd16383 && !clearing) begin
                frame_ready <= 1'b1;
            end
            // frame_ready stays high until next frame starts overwriting
            if (wr_en && wr_addr == 14'd0 && !clearing) begin
                frame_ready <= 1'b0;
            end

            // Lógica do contador de clear
            if (clearing) begin
                if (clear_addr == 14'd16383) begin
                    clearing <= 1'b0;
                end else begin
                    clear_addr <= clear_addr + 14'd1;
                end
            end
        end
    end

    // Multiplexadores de controle de escrita para limpar ou receber UART
    wire        actual_wr_en   = clearing | wr_en;
    wire [13:0] actual_wr_addr = clearing ? clear_addr : wr_addr;
    wire [7:0]  actual_wr_data = clearing ? 8'd0 : wr_data;

    // Bloco síncrono puro (sem reset) para inferir M9K
    always @(posedge clk) begin
        if (actual_wr_en) begin
            mem[actual_wr_addr] <= actual_wr_data;
        end

        if (vga_rd_en) begin
            vga_rd_data <= mem[vga_rd_addr];
        end
    end

endmodule
