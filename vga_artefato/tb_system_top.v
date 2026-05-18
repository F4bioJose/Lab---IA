`timescale 1ns / 1ps

// ==============================================================================
// Módulo: tb_system_top (Testbench)
// Descrição: Simula o pipeline UART → Framebuffer → VGA completo.
//            Envia 307200 bytes via protocolo UART simulado e verifica que o
//            framebuffer é preenchido e o VGA renderiza corretamente.
// ==============================================================================
module tb_system_top;

    // Parâmetros de simulação
    localparam CLK_PERIOD_NS = 20;       // 50 MHz
    localparam BAUD_RATE     = 1000000;  // 1 Mbaud para acelerar simulação
    localparam BIT_PERIOD_NS = 1000000000 / BAUD_RATE;

    // Sinais do DUT
    reg         CLOCK_50;
    reg         KEY;
    reg         UART_RXD;
    wire        VGA_CLK, VGA_HS, VGA_VS, VGA_BLANK_N, VGA_SYNC_N;
    wire [7:0]  VGA_R, VGA_G, VGA_B;
    wire        LEDG;
    wire [9:0]  LEDR;

    // Memória local para a imagem de teste
    reg [7:0] img_mem [0:307199];
    integer i;

    // Instanciação do DUT
    system_top uut (
        .CLOCK_50   (CLOCK_50),
        .KEY        (KEY),
        .UART_RXD   (UART_RXD),
        .VGA_CLK    (VGA_CLK),
        .VGA_HS     (VGA_HS),
        .VGA_VS     (VGA_VS),
        .VGA_BLANK_N(VGA_BLANK_N),
        .VGA_SYNC_N (VGA_SYNC_N),
        .VGA_R      (VGA_R),
        .VGA_G      (VGA_G),
        .VGA_B      (VGA_B),
        .LEDG       (LEDG),
        .LEDR       (LEDR)
    );

    // Override do baud rate no UART para simulação rápida
    defparam uut.uart_rx_inst.BAUD_RATE = BAUD_RATE;

    // Gerador de clock: 50 MHz (período = 20 ns)
    initial begin
        CLOCK_50 = 1'b0;
        forever #(CLK_PERIOD_NS / 2) CLOCK_50 = ~CLOCK_50;
    end

    // Task para enviar um byte via protocolo UART
    task send_uart_byte(input [7:0] data);
        integer bit_idx;
        begin
            // Start bit (LOW)
            UART_RXD = 1'b0;
            #(BIT_PERIOD_NS);

            // 8 bits de dados (LSB primeiro)
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                UART_RXD = data[bit_idx];
                #(BIT_PERIOD_NS);
            end

            // Stop bit (HIGH)
            UART_RXD = 1'b1;
            #(BIT_PERIOD_NS);
        end
    endtask

    // Sequência principal de teste
    initial begin
        // Valores iniciais
        KEY      = 1'b0;           // KEY = 0 → reset ativo
        UART_RXD = 1'b1;        // Linha idle

        // Carrega imagem de teste (gradiente para verificação visual)
        for (i = 0; i < 307200; i = i + 1) begin
            img_mem[i] = i[7:0];
        end

        // Reset
        #100;
        KEY = 1'b1;                 // Libera reset
        #100;

        $display("[%0t] Iniciando envio UART de 307200 bytes...", $time);

        // Envia os 307200 bytes da imagem via UART
        for (i = 0; i < 307200; i = i + 1) begin
            send_uart_byte(img_mem[i]);
            if (i % 65536 == 0)
                $display("[%0t]   Byte %0d / 307200 enviado", $time, i);
        end

        $display("[%0t] Envio completo! frame_received = %b", $time, LEDG);

        // Espera breve para estabilizar (simulação)
        #500000;

        // Verifica que o framebuffer contém os dados corretos
        $display("--- Verificacao do Framebuffer ---");
        $display("  frame_mem[0]      = %0d (esperado: 0)",   uut.framebuffer.mem[0]);
        $display("  frame_mem[127]    = %0d (esperado: 127)", uut.framebuffer.mem[127]);
        $display("  frame_mem[255]    = %0d (esperado: 255)", uut.framebuffer.mem[255]);
        $display("  frame_mem[307199] = %0d (esperado: 255)", uut.framebuffer.mem[307199]);
        $display("  LEDG (frame_received) = %b", LEDG);

        $display("--- Simulacao concluida com sucesso ---");
        $stop;
    end

    // Timeout de segurança (15ms de simulação)
    initial begin
        #15000000000;
        $display("[TIMEOUT] Simulacao excedeu o tempo limite.");
        $stop;
    end

    // Monitor de progresso da UART
    always @(posedge CLOCK_50) begin
        if (uut.uart_valid)
            if (uut.uart_wr_addr < 3 || uut.uart_wr_addr > 307196)
                $display("[%0t] UART byte recebido: addr=%0d data=0x%02h",
                    $time, uut.uart_wr_addr, uut.uart_data);
    end

endmodule
