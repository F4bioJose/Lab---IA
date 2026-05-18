`timescale 1ns / 1ps

module tb_top_vga();
    // Injeção de sinais (Inputs)
    reg CLOCK_50;
    reg [0:0] KEY;
    reg [7:0] sram_data_out;
    reg access_granted;

    // Captura de sinais (Outputs)
    wire VGA_HS, VGA_VS, VGA_CLK, VGA_BLANK_N, VGA_SYNC_N;
    wire [7:0] VGA_R, VGA_G, VGA_B;
    wire [19:0] sram_address;

    // Instanciando a sua "Placa-Mãe"
    top_vga uut (
        .CLOCK_50(CLOCK_50),
        .KEY(KEY),
        .sram_data_out(sram_data_out),
        .access_granted(access_granted),
        .VGA_HS(VGA_HS),
        .VGA_VS(VGA_VS),
        .VGA_CLK(VGA_CLK),
        .VGA_BLANK_N(VGA_BLANK_N),
        .VGA_SYNC_N(VGA_SYNC_N),
        .VGA_R(VGA_R),
        .VGA_G(VGA_G),
        .VGA_B(VGA_B),
        .sram_address(sram_address)
    );

    
    always #10 CLOCK_50 = ~CLOCK_50;

    initial begin
       
        CLOCK_50 = 0;
        KEY[0] = 0;             
        sram_data_out = 8'h80;  
        access_granted = 1;     

        
        #100;
        KEY[0] = 1; 

       
		  #16800000;
        $stop;
    end
endmodule