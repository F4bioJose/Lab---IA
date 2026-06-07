`timescale 1ns / 1ps
// ==============================================================================
// Módulo: weights_shared_rom
// Descrição: Inicializa estritamente via weights_all.mif e usa um bootloader 
//            de hardware para descompactar os 17.159 bytes para registradores
//            e sub-blocos M9K das 19 classes para inferência ultra-rápida.
// ==============================================================================
module weights_shared_rom #(
    parameter MEM_FILE      = "../modulos_verilog/weights_all.mif", // ATENÇÃO: caminho a partir de quartus_cnn/
    parameter integer CONV_FILTERS  = 4,
    parameter integer CONV_KERNEL   = 9,
    parameter integer DENSE_SIZE    = 900,
    parameter integer DENSE_CLASSES = 19,
    parameter integer OFF_CONV_W    = 0,
    parameter integer OFF_CONV_B    = CONV_FILTERS * CONV_KERNEL,
    parameter integer OFF_DENSE_W   = OFF_CONV_B + CONV_FILTERS,
    parameter integer OFF_DENSE_B   = OFF_DENSE_W + DENSE_SIZE * DENSE_CLASSES,
    parameter integer TOTAL_WORDS   = OFF_DENSE_B + DENSE_CLASSES
)(
    input  wire                     clk,
    input  wire                     rst,
    output wire                     boot_done,      // HIGH quando inicialização terminar
    input  wire [13:0]              dense_addr,
    
    output wire signed [7:0]        conv_w0 [0:CONV_KERNEL-1],
    output wire signed [7:0]        conv_w1 [0:CONV_KERNEL-1],
    output wire signed [7:0]        conv_w2 [0:CONV_KERNEL-1],
    output wire signed [7:0]        conv_w3 [0:CONV_KERNEL-1],
    
    output wire signed [7:0]        conv_b0,
    output wire signed [7:0]        conv_b1,
    output wire signed [7:0]        conv_b2,
    output wire signed [7:0]        conv_b3,
    
    output reg  signed [7:0]        dense_w [0:DENSE_CLASSES-1],
    output wire signed [7:0]        dense_b [0:DENSE_CLASSES-1]
);

    // =========================================================================
    // REGISTRADORES ESTÁTICOS (Populados no boot)
    // =========================================================================
    reg signed [7:0] r_conv_w0 [0:CONV_KERNEL-1];
    reg signed [7:0] r_conv_w1 [0:CONV_KERNEL-1];
    reg signed [7:0] r_conv_w2 [0:CONV_KERNEL-1];
    reg signed [7:0] r_conv_w3 [0:CONV_KERNEL-1];
    
    reg signed [7:0] r_conv_b0, r_conv_b1, r_conv_b2, r_conv_b3;
    reg signed [7:0] r_dense_b [0:DENSE_CLASSES-1];

    assign conv_w0[0] = r_conv_w0[0]; assign conv_w0[1] = r_conv_w0[1]; assign conv_w0[2] = r_conv_w0[2];
    assign conv_w0[3] = r_conv_w0[3]; assign conv_w0[4] = r_conv_w0[4]; assign conv_w0[5] = r_conv_w0[5];
    assign conv_w0[6] = r_conv_w0[6]; assign conv_w0[7] = r_conv_w0[7]; assign conv_w0[8] = r_conv_w0[8];

    assign conv_w1[0] = r_conv_w1[0]; assign conv_w1[1] = r_conv_w1[1]; assign conv_w1[2] = r_conv_w1[2];
    assign conv_w1[3] = r_conv_w1[3]; assign conv_w1[4] = r_conv_w1[4]; assign conv_w1[5] = r_conv_w1[5];
    assign conv_w1[6] = r_conv_w1[6]; assign conv_w1[7] = r_conv_w1[7]; assign conv_w1[8] = r_conv_w1[8];

    assign conv_w2[0] = r_conv_w2[0]; assign conv_w2[1] = r_conv_w2[1]; assign conv_w2[2] = r_conv_w2[2];
    assign conv_w2[3] = r_conv_w2[3]; assign conv_w2[4] = r_conv_w2[4]; assign conv_w2[5] = r_conv_w2[5];
    assign conv_w2[6] = r_conv_w2[6]; assign conv_w2[7] = r_conv_w2[7]; assign conv_w2[8] = r_conv_w2[8];

    assign conv_w3[0] = r_conv_w3[0]; assign conv_w3[1] = r_conv_w3[1]; assign conv_w3[2] = r_conv_w3[2];
    assign conv_w3[3] = r_conv_w3[3]; assign conv_w3[4] = r_conv_w3[4]; assign conv_w3[5] = r_conv_w3[5];
    assign conv_w3[6] = r_conv_w3[6]; assign conv_w3[7] = r_conv_w3[7]; assign conv_w3[8] = r_conv_w3[8];

    assign conv_b0 = r_conv_b0; assign conv_b1 = r_conv_b1;
    assign conv_b2 = r_conv_b2; assign conv_b3 = r_conv_b3;

    assign dense_b[0] = r_dense_b[0]; assign dense_b[1] = r_dense_b[1]; assign dense_b[2] = r_dense_b[2];
    assign dense_b[3] = r_dense_b[3]; assign dense_b[4] = r_dense_b[4]; assign dense_b[5] = r_dense_b[5];
    assign dense_b[6] = r_dense_b[6]; assign dense_b[7] = r_dense_b[7]; assign dense_b[8] = r_dense_b[8];
    assign dense_b[9] = r_dense_b[9]; assign dense_b[10]= r_dense_b[10];assign dense_b[11]= r_dense_b[11];
    assign dense_b[12]= r_dense_b[12];assign dense_b[13]= r_dense_b[13];assign dense_b[14]= r_dense_b[14];
    assign dense_b[15]= r_dense_b[15];assign dense_b[16]= r_dense_b[16];assign dense_b[17]= r_dense_b[17];
    assign dense_b[18]= r_dense_b[18];

    // =========================================================================
    // SUB-BLOCOS M9K DA CAMADA DENSA
    // =========================================================================
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d0 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d1 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d2 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d3 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d4 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d5 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d6 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d7 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d8 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d9 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d10 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d11 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d12 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d13 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d14 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d15 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d16 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d17 [0:DENSE_SIZE-1];
    (* ramstyle = "M9K" *) reg signed [7:0] mem_d18 [0:DENSE_SIZE-1];

    // =========================================================================
    // ROM MESTRE (Inicializada nativamente pelo Quartus via .mif)
    // =========================================================================
    (* romstyle = "M9K", ram_init_file = MEM_FILE *) reg signed [7:0] master_rom [0:TOTAL_WORDS-1];
    
    // synthesis translate_off
    initial begin
        // O ModelSim RTL ignora atributos do Quartus, então carregamos o .hex 
        // apenas no testbench. A tag acima diz ao Quartus para ignorar isso na FPGA!
        $readmemh("modulos_verilog/weights_all.hex", master_rom);
    end
    // synthesis translate_on
    
    reg [14:0] boot_addr;
    reg [14:0] boot_addr_d1;
    reg signed [7:0] boot_data;

    always @(posedge clk) begin
        boot_data <= master_rom[boot_addr];
        boot_addr_d1 <= boot_addr;
    end

    // =========================================================================
    // BOOTLOADER DE HARDWARE
    // =========================================================================
    reg r_boot_done;
    assign boot_done = r_boot_done;

    reg [4:0] copy_class;
    reg [9:0] copy_idx;
    
    reg boot_reading;
    reg boot_reading_d1;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            boot_addr       <= 0;
            r_boot_done     <= 1'b0;
            copy_class      <= 0;
            copy_idx        <= 0;
            boot_reading    <= 1'b0;
            boot_reading_d1 <= 1'b0;
        end else begin
            // Pipeline para saber quando o dado na saída da master_rom é válido
            boot_reading_d1 <= boot_reading;

            // Avanço de endereço de leitura da Mestre
            if (!r_boot_done) begin
                boot_reading <= 1'b1;
                if (boot_reading) begin
                    if (boot_addr < TOTAL_WORDS) begin
                        boot_addr <= boot_addr + 1;
                    end
                end
                
                // Finaliza o boot quando o último endereço foi lido e processado
                if (boot_addr_d1 == TOTAL_WORDS - 1 && boot_reading_d1) begin
                    r_boot_done <= 1'b1;
                end
            end else begin
                boot_reading <= 1'b0;
            end

            // Processamento do dado lido (ciclo atual é boot_addr_d1)
            if (!r_boot_done && boot_reading_d1) begin
                
                // Convolução (0 a 35)
                if (boot_addr_d1 >= OFF_CONV_W && boot_addr_d1 < OFF_CONV_B) begin
                    case (boot_addr_d1)
                        0: r_conv_w0[0] <= boot_data;  1: r_conv_w0[1] <= boot_data;  2: r_conv_w0[2] <= boot_data;
                        3: r_conv_w0[3] <= boot_data;  4: r_conv_w0[4] <= boot_data;  5: r_conv_w0[5] <= boot_data;
                        6: r_conv_w0[6] <= boot_data;  7: r_conv_w0[7] <= boot_data;  8: r_conv_w0[8] <= boot_data;
                        9: r_conv_w1[0] <= boot_data; 10: r_conv_w1[1] <= boot_data; 11: r_conv_w1[2] <= boot_data;
                       12: r_conv_w1[3] <= boot_data; 13: r_conv_w1[4] <= boot_data; 14: r_conv_w1[5] <= boot_data;
                       15: r_conv_w1[6] <= boot_data; 16: r_conv_w1[7] <= boot_data; 17: r_conv_w1[8] <= boot_data;
                       18: r_conv_w2[0] <= boot_data; 19: r_conv_w2[1] <= boot_data; 20: r_conv_w2[2] <= boot_data;
                       21: r_conv_w2[3] <= boot_data; 22: r_conv_w2[4] <= boot_data; 23: r_conv_w2[5] <= boot_data;
                       24: r_conv_w2[6] <= boot_data; 25: r_conv_w2[7] <= boot_data; 26: r_conv_w2[8] <= boot_data;
                       27: r_conv_w3[0] <= boot_data; 28: r_conv_w3[1] <= boot_data; 29: r_conv_w3[2] <= boot_data;
                       30: r_conv_w3[3] <= boot_data; 31: r_conv_w3[4] <= boot_data; 32: r_conv_w3[5] <= boot_data;
                       33: r_conv_w3[6] <= boot_data; 34: r_conv_w3[7] <= boot_data; 35: r_conv_w3[8] <= boot_data;
                    endcase
                end
                
                // Vieses Convolução (36 a 39)
                else if (boot_addr_d1 >= OFF_CONV_B && boot_addr_d1 < OFF_DENSE_W) begin
                    case (boot_addr_d1)
                        36: r_conv_b0 <= boot_data;
                        37: r_conv_b1 <= boot_data;
                        38: r_conv_b2 <= boot_data;
                        39: r_conv_b3 <= boot_data;
                    endcase
                end
                
                // Pesos Densos (40 a 17139)
                else if (boot_addr_d1 >= OFF_DENSE_W && boot_addr_d1 < OFF_DENSE_B) begin
                    case (copy_class)
                        0: mem_d0[copy_idx] <= boot_data;  1: mem_d1[copy_idx] <= boot_data;
                        2: mem_d2[copy_idx] <= boot_data;  3: mem_d3[copy_idx] <= boot_data;
                        4: mem_d4[copy_idx] <= boot_data;  5: mem_d5[copy_idx] <= boot_data;
                        6: mem_d6[copy_idx] <= boot_data;  7: mem_d7[copy_idx] <= boot_data;
                        8: mem_d8[copy_idx] <= boot_data;  9: mem_d9[copy_idx] <= boot_data;
                       10: mem_d10[copy_idx] <= boot_data; 11: mem_d11[copy_idx] <= boot_data;
                       12: mem_d12[copy_idx] <= boot_data; 13: mem_d13[copy_idx] <= boot_data;
                       14: mem_d14[copy_idx] <= boot_data; 15: mem_d15[copy_idx] <= boot_data;
                       16: mem_d16[copy_idx] <= boot_data; 17: mem_d17[copy_idx] <= boot_data;
                       18: mem_d18[copy_idx] <= boot_data;
                    endcase

                    if (copy_idx == DENSE_SIZE - 1) begin
                        copy_idx <= 0;
                        copy_class <= copy_class + 1;
                    end else begin
                        copy_idx <= copy_idx + 1;
                    end
                end
                
                // Vieses Densos (17140 a 17158)
                else if (boot_addr_d1 >= OFF_DENSE_B && boot_addr_d1 < TOTAL_WORDS) begin
                    r_dense_b[boot_addr_d1 - OFF_DENSE_B] <= boot_data;
                end
            end
        end
    end

    // =========================================================================
    // INFERÊNCIA PÓS-BOOT: Leitura síncrona das memórias paralelas
    // =========================================================================
    always @(posedge clk) begin
        dense_w[ 0] <= mem_d0[dense_addr];
        dense_w[ 1] <= mem_d1[dense_addr];
        dense_w[ 2] <= mem_d2[dense_addr];
        dense_w[ 3] <= mem_d3[dense_addr];
        dense_w[ 4] <= mem_d4[dense_addr];
        dense_w[ 5] <= mem_d5[dense_addr];
        dense_w[ 6] <= mem_d6[dense_addr];
        dense_w[ 7] <= mem_d7[dense_addr];
        dense_w[ 8] <= mem_d8[dense_addr];
        dense_w[ 9] <= mem_d9[dense_addr];
        dense_w[10] <= mem_d10[dense_addr];
        dense_w[11] <= mem_d11[dense_addr];
        dense_w[12] <= mem_d12[dense_addr];
        dense_w[13] <= mem_d13[dense_addr];
        dense_w[14] <= mem_d14[dense_addr];
        dense_w[15] <= mem_d15[dense_addr];
        dense_w[16] <= mem_d16[dense_addr];
        dense_w[17] <= mem_d17[dense_addr];
        dense_w[18] <= mem_d18[dense_addr];
    end

endmodule
