`timescale 1ns/1ps

// ==============================================================================
// Módulo: tb_cnn_top (Testbench)
// Descrição: Ambiente de simulação para testar a integração do top-level da CNN.
//            Carrega uma imagem em memória, injeta no framebuffer de entrada, 
//            inicia o processo e monitora as predições e escores de saída.
// ==============================================================================
module tb_cnn_top;

    localparam integer CLK_HALF_PERIOD_NS = 10; // 50 MHz clock
    localparam integer BAUD_RATE = 1000000;
    localparam integer BIT_PERIOD_NS = 1000000000 / BAUD_RATE;
    localparam integer USE_UART = 1;

    string img_file;

    reg clk;
    reg rst;
    reg start_system;
    reg rx_pin;

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
        .rx_pin(rx_pin),
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

    defparam uut.uart_rx_inst.BAUD_RATE = BAUD_RATE;

    initial begin
        clk = 1'b0;
        forever #(CLK_HALF_PERIOD_NS) clk = ~clk;
    end

    task send_uart_byte(input [7:0] data);
        integer bit_idx;
        begin
            rx_pin = 1'b0;
            #(BIT_PERIOD_NS);
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                rx_pin = data[bit_idx];
                #(BIT_PERIOD_NS);
            end
            rx_pin = 1'b1;
            #(BIT_PERIOD_NS);
        end
    endtask

    initial begin
        rst = 1'b1;
        start_system = 1'b0;
        rx_pin = 1'b1;
        fb_wr_en = 1'b0;
        fb_wr_addr = 10'd0;
        fb_wr_data = 8'd0;
        vga_rd_en = 1'b0;
        vga_rd_addr = 10'd0;

        img_file = "inputs/teste2.txt";
        if ($value$plusargs("IMG=%s", img_file)) begin
            $display("Using IMG file: %s", img_file);
        end else begin
            $display("Using default IMG file: %s", img_file);
        end

        // Carrega a imagem de teste externa (arquivo texto com dados hexa) para a RAM local
        $readmemh(img_file, img_mem);

        #20;
        rst = 1'b0;

        if (USE_UART) begin
            #(BIT_PERIOD_NS * 2);
            for (i = 0; i < 1024; i = i + 1) begin
                send_uart_byte(img_mem[i]);
            end
        end else begin
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
        end

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
