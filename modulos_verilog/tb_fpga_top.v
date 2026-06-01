`timescale 1ns/1ps

// ==============================================================================
// Módulo: tb_fpga_top (Testbench)
// Descrição: Testbench de simulação do wrapper fpga_top_de2115.
//            Simula o envio de 1024 bytes via UART serial (como faria o
//            script send_image_32x32.py) e verifica que a inferência
//            completa corretamente, com resultado nos LEDs.
// ==============================================================================
module tb_fpga_top;

    // =========================================================================
    // PARÂMETROS
    // =========================================================================
    localparam integer CLK_HALF_PERIOD_NS = 10;     // 50 MHz clock
    localparam integer BAUD_RATE = 1000000;          // Baud rate acelerado para simulação
    localparam integer BIT_PERIOD_NS = 1000000000 / BAUD_RATE;

    // =========================================================================
    // SINAIS
    // =========================================================================
    reg         clk_50;
    reg  [1:0]  key;
    reg         uart_rxd;
    wire [7:0]  ledg;

    // Memória local para a imagem de teste
    reg [7:0] img_mem [0:1023];
    string img_file;
    integer i;

    // =========================================================================
    // DUT (Design Under Test)
    // =========================================================================
    fpga_top_de2115 dut (
        .CLOCK_50  (clk_50),
        .KEY       (key),
        .UART_RXD  (uart_rxd),
        .LEDG      (ledg)
    );

    // Sobrescreve o baud rate do uart_rx interno para acelerar a simulação
    defparam dut.cnn_inst.uart_rx_inst.BAUD_RATE = BAUD_RATE;

    // =========================================================================
    // GERAÇÃO DE CLOCK
    // =========================================================================
    initial begin
        clk_50 = 1'b0;
        forever #(CLK_HALF_PERIOD_NS) clk_50 = ~clk_50;
    end

    // =========================================================================
    // TASK: Enviar 1 byte via protocolo UART (start + 8 data + stop)
    // =========================================================================
    task send_uart_byte(input [7:0] data);
        integer bit_idx;
        begin
            // Start bit (LOW)
            uart_rxd = 1'b0;
            #(BIT_PERIOD_NS);
            // 8 data bits (LSB first)
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                uart_rxd = data[bit_idx];
                #(BIT_PERIOD_NS);
            end
            // Stop bit (HIGH)
            uart_rxd = 1'b1;
            #(BIT_PERIOD_NS);
        end
    endtask

    // =========================================================================
    // ESTÍMULO PRINCIPAL
    // =========================================================================
    initial begin
        // Inicialização
        key = 2'b11;            // Ambos os botões soltos (active-low)
        uart_rxd = 1'b1;        // Linha UART idle = HIGH

        // Carrega a imagem de teste
        img_file = "inputs/teste2.txt";
        if ($value$plusargs("IMG=%s", img_file)) begin
            $display("[TB] Using IMG file: %s", img_file);
        end else begin
            $display("[TB] Using default IMG file: %s", img_file);
        end
        $readmemh(img_file, img_mem);

        // =====================================================================
        // FASE 1: Reset
        // =====================================================================
        $display("[TB] ===== FASE 1: Reset =====");
        key[0] = 1'b0;         // Pressiona KEY[0] = reset ativo
        #100;
        key[0] = 1'b1;         // Solta KEY[0] = reset liberado
        #100;
        $display("[TB] Reset concluído. LEDs = %08b", ledg);

        // =====================================================================
        // FASE 2: Envio da imagem via UART (1024 bytes)
        // =====================================================================
        $display("[TB] ===== FASE 2: Enviando imagem via UART (%0d bytes) =====",
                 1024);

        #(BIT_PERIOD_NS * 2);   // Pausa antes do primeiro byte

        for (i = 0; i < 1024; i = i + 1) begin
            send_uart_byte(img_mem[i]);
            if (i == 0)
                $display("[TB] Primeiro byte enviado: 0x%02h", img_mem[i]);
            if (i == 1023)
                $display("[TB] Último byte enviado: 0x%02h", img_mem[i]);
        end

        $display("[TB] Todos os 1024 bytes enviados via UART.");
        $display("[TB] frame_ready interno = %0b", dut.cnn_inst.frame_ready);

        // =====================================================================
        // FASE 3: Aguardar inferência concluir
        // =====================================================================
        $display("[TB] ===== FASE 3: Aguardando inferência =====");

        // A FSM do cnn_top deve auto-iniciar a inferência quando frame_ready=1
        // (via uart_start_pulse gerado internamente)
        wait (ledg[6] == 1'b1);

        $display("[TB] ===== INFERÊNCIA CONCLUÍDA =====");
        $display("[TB] LEDs finais:   %08b", ledg);
        $display("[TB]   class_id   = %0d (LEDG[2:0])", ledg[2:0]);
        $display("[TB]   unknown    = %0b (LEDG[7])", ledg[7]);
        $display("[TB]   done       = %0b (LEDG[6])", ledg[6]);

        // Exibe os scores internos da camada densa
        $display("[TB] Dense scores (Q2.14):");
        for (i = 0; i < 7; i = i + 1) begin
            $display("[TB]   score[%0d] = %0d", i,
                     $signed(dut.cnn_inst.dense_scores[i]));
        end

        $display("[TB] final_result (Q2.14) = %04h",
                 dut.cnn_inst.final_result);

        #100;
        $display("[TB] ===== SIMULAÇÃO CONCLUÍDA COM SUCESSO =====");
        $stop;
    end

    // =========================================================================
    // MONITOR: Exibe progresso de recepção UART
    // =========================================================================
    reg [9:0] uart_progress_prev = 0;
    always @(posedge clk_50) begin
        if (dut.cnn_inst.uart_wr_addr != uart_progress_prev) begin
            if (dut.cnn_inst.uart_wr_addr % 256 == 0 && dut.cnn_inst.uart_wr_addr > 0) begin
                $display("[TB] UART progresso: %0d/1024 bytes recebidos",
                         dut.cnn_inst.uart_wr_addr);
            end
            uart_progress_prev = dut.cnn_inst.uart_wr_addr;
        end
    end

    // =========================================================================
    // MONITOR: Detecta transições de estado da FSM
    // =========================================================================
    reg [1:0] prev_state = 0;
    always @(posedge clk_50) begin
        if (dut.cnn_inst.state != prev_state) begin
            case (dut.cnn_inst.state)
                2'd0: $display("[TB] FSM -> ST_IDLE  (t=%0t)", $time);
                2'd1: $display("[TB] FSM -> ST_READ  (t=%0t)", $time);
                2'd2: $display("[TB] FSM -> ST_WAIT  (t=%0t)", $time);
                2'd3: $display("[TB] FSM -> ST_DONE  (t=%0t)", $time);
            endcase
            prev_state = dut.cnn_inst.state;
        end
    end

endmodule
