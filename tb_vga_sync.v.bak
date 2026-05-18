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

    always #20 clk_25mhz = ~clk_25mhz;

    initial begin
        clk_25mhz = 0;
        rst = 1; 
        #100;
        rst = 0; 
        #64000;
        $stop; 
    end

endmodule