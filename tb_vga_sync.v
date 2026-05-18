`timescale 1ns / 1ps

module tb_vga_sync();

    reg clk_25mhz;
    reg rst;
    wire hsync;
    wire vsync;
    wire [9:0] pixel_x;
    wire [9:0] pixel_y;
    wire video_on;

    vga_sync uut (
        .clk(clk_25mhz),
        .rst(rst),
        .hsync(hsync),
        .vsync(vsync),
        .pixel_x(pixel_x),
        .pixel_y(pixel_y),
        .video_on(video_on)
    );

    always #20 clk_25mhz = ~clk_25mhz; // 20ns * 2 = 40ns -> f = 1/40 = 25Mhz

    initial begin
        clk_25mhz = 0;
        rst = 1; 
        #100;     // tempo para estabilizar
        rst = 0; 
        #16800000;  // tempo de execucao
        $stop; 
    end

endmodule