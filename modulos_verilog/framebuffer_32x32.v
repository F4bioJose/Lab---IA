// ==============================================================================
// Módulo: framebuffer_32x32
// Descrição: Memória de buffer para armazenar uma imagem de entrada 32x32 
//            (1024 pixels). Fornece portas de leitura independentes para
//            a inferência (CNN) e para o controlador de vídeo (VGA).
// ==============================================================================
module framebuffer_32x32 (
    input wire clk,
    input wire rst,

    // ==========================================
    // [PONTO DE ENTRADA DO UART]
    // A Máquina de Estados (FSM) do UART deverá enviar os dados recebidos 
    // do computador por estas portas de escrita para popular o buffer.
    // ==========================================
    input wire wr_en,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,

    input wire rd_en,
    input wire [9:0] rd_addr,
    output reg [7:0] rd_data,

    // ==========================================
    // [PONTO DE SAÍDA PARA O CONTROLADOR VGA]
    // O Controlador de Vídeo solicitará a leitura dos endereços em 60Hz
    // através destas portas para desenhar a face na tela paralelamente à CNN.
    // (Na fase de Integração Final, este módulo todo será substituído pela SRAM Externa)
    // ==========================================
    input wire vga_rd_en,
    input wire [9:0] vga_rd_addr,
    output reg [7:0] vga_rd_data,

    input wire frame_clear,
    output reg frame_ready
);

    // Utiliza duas memórias espelhadas (mem_a e mem_b) para permitir duas 
    // leituras assíncronas simultâneas sem instanciar uma RAM de 3 portas complexa.
    reg [7:0] mem_a [0:1023];
    reg [7:0] mem_b [0:1023];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd_data <= 8'd0;
            vga_rd_data <= 8'd0;
            frame_ready <= 1'b0;
        end else begin
            if (frame_clear) begin
                frame_ready <= 1'b0; // Limpa a flag de frame pronto
            end else if (wr_en && wr_addr == 10'd1023) begin
                // Quando o último pixel (1023) é escrito, a imagem completa está pronta
                frame_ready <= 1'b1;
            end

            // Escrita simultânea e espelhada em ambas as memórias
            if (wr_en) begin
                mem_a[wr_addr] <= wr_data;
                mem_b[wr_addr] <= wr_data;
            end

            // Leitura dedicada ao processamento da CNN
            if (rd_en) begin
                rd_data <= mem_a[rd_addr];
            end

            // Leitura dedicada ao fluxo de saída de vídeo VGA
            if (vga_rd_en) begin
                vga_rd_data <= mem_b[vga_rd_addr];
            end
        end
    end

endmodule
