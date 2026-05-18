`timescale 1ns/1ps

module tb_uart_debug;

    localparam integer CLK_HALF_PERIOD_NS = 10;
    localparam integer BAUD_RATE = 1000000;
    localparam integer BIT_PERIOD_NS = 1000000000 / BAUD_RATE;

    reg clk, rst, start_system, rx_pin;
    reg fb_wr_en;
    reg [9:0] fb_wr_addr;
    reg [7:0] fb_wr_data;
    reg vga_rd_en;
    reg [9:0] vga_rd_addr;
    wire [7:0] vga_rd_data;
    wire [15:0] final_result;
    wire [2:0] class_id;
    wire unknown, access_done, frame_ready;
    reg [7:0] img_mem [0:1023];
    integer i, c, uart_byte_count;

    cnn_top uut (
        .clk(clk), .rst(rst), .rx_pin(rx_pin),
        .start_system(start_system),
        .fb_wr_en(fb_wr_en), .fb_wr_addr(fb_wr_addr), .fb_wr_data(fb_wr_data),
        .vga_rd_en(vga_rd_en), .vga_rd_addr(vga_rd_addr), .vga_rd_data(vga_rd_data),
        .final_result(final_result), .class_id(class_id),
        .unknown(unknown), .access_done(access_done), .frame_ready(frame_ready)
    );

    defparam uut.uart_rx_inst.BAUD_RATE = BAUD_RATE;

    initial begin clk = 0; forever #(CLK_HALF_PERIOD_NS) clk = ~clk; end

    task send_uart_byte(input [7:0] data);
        integer bit_idx;
        begin
            rx_pin = 1'b0; #(BIT_PERIOD_NS);
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                rx_pin = data[bit_idx]; #(BIT_PERIOD_NS);
            end
            rx_pin = 1'b1; #(BIT_PERIOD_NS);
        end
    endtask

    // Monitor de transição de estado
    reg [1:0] prev_state;
    always @(posedge clk) begin
        prev_state <= uut.state;
        if (uut.state != prev_state)
            $display("[%0t] FSM: %0d -> %0d (fr=%b pend=%b rd_req=%0d rd_val=%0d dense_done=%b)",
                $time, prev_state, uut.state, frame_ready,
                uut.uart_frame_pending, uut.rd_req_count, uut.rd_val_count, uut.dense_done);
    end

    // Contadores de eventos no pipeline
    integer conv_count = 0, pool_count = 0, flat_count = 0;
    always @(posedge clk) begin
        if (uut.conv_valid) begin
            conv_count = conv_count + 1;
            if (conv_count <= 2 || conv_count == 900)
                $display("[%0t] conv_valid #%0d", $time, conv_count);
        end
        if (uut.pool_valid) begin
            pool_count = pool_count + 1;
            if (pool_count <= 2 || pool_count == 900)
                $display("[%0t] pool_valid #%0d", $time, pool_count);
        end
        if (uut.flat_valid) begin
            flat_count = flat_count + 1;
            if (flat_count <= 2 || flat_count == 900)
                $display("[%0t] flat_valid #%0d", $time, flat_count);
        end
        if (uut.dense_done)
            $display("[%0t] >>> dense_done!", $time);
        if (access_done)
            $display("[%0t] >>> access_done!", $time);
    end

    // Timeout
    initial begin
        #15000000; // 15ms timeout
        $display("TIMEOUT! conv=%0d pool=%0d flat=%0d state=%0d dense_done=%b",
            conv_count, pool_count, flat_count, uut.state, uut.dense_done);
        $finish;
    end

    initial begin
        rst=1; start_system=0; rx_pin=1; fb_wr_en=0; fb_wr_addr=0; fb_wr_data=0;
        vga_rd_en=0; vga_rd_addr=0; uart_byte_count=0;
        $readmemh("inputs/teste2.txt", img_mem);
        #100; rst=0;
        #(BIT_PERIOD_NS * 2);

        for (i = 0; i < 1024; i = i + 1) send_uart_byte(img_mem[i]);

        $display("[%0t] Envio completo. state=%0d", $time, uut.state);

        wait (access_done == 1'b1 || $time > 14000000);

        if (access_done) begin
            $display("=== SUCESSO ===");
            $display("Final result (Q2.14) = %04h", final_result);
            $display("Class ID = %0d, Unknown = %0d", class_id, unknown);
            for (c = 0; c < 7; c = c + 1)
                $display("  score[%0d] = %0d", c, $signed(uut.dense_scores[c]));
        end else begin
            $display("=== FALHA: access_done nao subiu ===");
            $display("conv=%0d pool=%0d flat=%0d state=%0d", conv_count, pool_count, flat_count, uut.state);
        end

        #40;
        $finish;
    end
endmodule
