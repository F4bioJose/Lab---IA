`timescale 1ns/1ps

// ==============================================================================
// Módulo: tb_cnn_top (Testbench)
// Descrição: Ambiente de simulação para testar a integração do top-level da CNN.
//            Carrega uma imagem em memória, injeta no framebuffer de entrada, 
//            inicia o processo e monitora as predições e escores de saída.
// ==============================================================================
module tb_cnn_top;

    reg clk;
    reg rst;
    reg start_system;

    reg fb_wr_en;
    reg [9:0] fb_wr_addr;
    reg [7:0] fb_wr_data;

    reg vga_rd_en;
    reg [9:0] vga_rd_addr;
    wire [7:0] vga_rd_data;

    wire [15:0] final_result;
    wire [2:0] class_id;
    wire unknown;
    wire access_done;
    wire frame_ready;

    reg [7:0] img_mem [0:1023];

    integer i;
    integer c;

    cnn_top uut (
        .clk(clk),
        .rst(rst),
        .start_system(start_system),
        .fb_wr_en(fb_wr_en),
        .fb_wr_addr(fb_wr_addr),
        .fb_wr_data(fb_wr_data),
        .vga_rd_en(vga_rd_en),
        .vga_rd_addr(vga_rd_addr),
        .vga_rd_data(vga_rd_data),
        .final_result(final_result),
        .class_id(class_id),
        .unknown(unknown),
        .access_done(access_done),
        .frame_ready(frame_ready)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst = 1'b1;
        start_system = 1'b0;
        fb_wr_en = 1'b0;
        fb_wr_addr = 10'd0;
        fb_wr_data = 8'd0;
        vga_rd_en = 1'b0;
        vga_rd_addr = 10'd0;

        // Carrega a imagem de teste externa (arquivo texto com dados hexa) para a RAM local
        $readmemh("inputs/teste2.txt", img_mem);

        #20;
        rst = 1'b0;

        // Etapa 1: Injeta sequencialmente a imagem completa no framebuffer (1024 pixels)
        for (i = 0; i < 1024; i = i + 1) begin
            @(negedge clk);
            fb_wr_en = 1'b1;
            fb_wr_addr = i[9:0];
            fb_wr_data = img_mem[i];
        end

        @(negedge clk);
        fb_wr_en = 1'b0;
        fb_wr_addr = 10'd0;
        fb_wr_data = 8'd0;

        wait (frame_ready == 1'b1);

        // Etapa 2: Sinaliza o início da execução da rede neural
        @(negedge clk);
        start_system = 1'b1;
        @(negedge clk);
        start_system = 1'b0;

        // Etapa 3: Aguarda a conclusão e exibe detalhadamente os resultados no console
        wait (access_done == 1'b1);
        $display("Final result (Q2.14) = %04h", final_result);
        $display("Class ID = %0d, Unknown = %0d", class_id, unknown);
        $display("Dense scores (Q2.14):");
        for (c = 0; c < 7; c = c + 1) begin
            $display("  score[%0d] = %0d", c, $signed(uut.dense_scores[c]));
        end

        #40;
        $stop;
    end

    // DEBUG PRINTS
    reg [9:0] conv_cnt = 0;
    always @(posedge clk) begin
        if (uut.conv_valid) begin
            if (conv_cnt < 4) begin
                $display("VERILOG DEBUG: out_f0[%0d] = %0d", conv_cnt, uut.conv_out_f0);
            end
            conv_cnt = conv_cnt + 1;
        end
    end

    reg [9:0] pool_cnt = 0;
    reg signed [47:0] pool_sum = 0;
    always @(posedge clk) begin
        if (uut.pool_valid) begin
            if (pool_cnt < 4) begin
                $display("VERILOG DEBUG: pool_data[%0d] = %0d", pool_cnt, uut.pool_data);
            end
            if (pool_cnt >= 896) begin
                $display("VERILOG DEBUG: pool_data[%0d] = %0d", pool_cnt, uut.pool_data);
            end
            pool_cnt = pool_cnt + 1;
            pool_sum = pool_sum + uut.pool_data;
            if (pool_cnt == 900) begin
                $display("VERILOG DEBUG: POOL SUM = %0d", pool_sum);
            end
        end
    end

endmodule
