// ==============================================================================
// Módulo: fpga_top_de2115
// Descrição: Top-level sintetizável para inferência CNN na DE2-115.
//            Recebe 1 imagem 32×32 grayscale via UART serial (115200 baud),
//            executa a inferência completa pelo pipeline cnn_top, e exibe
//            o resultado da classificação nos LEDs verdes (LEDG).
//
// Mapeamento dos LEDs:
//   LEDG[2:0] = class_id (0..6 = classes, 7 = desconhecido)
//   LEDG[6]   = access_done (inferência concluída)
//   LEDG[7]   = unknown (score abaixo do threshold)
//   LEDG[5:3] = (reservado, apagados)
//
// Controles:
//   KEY[0] = Reset global (active-low, com Schmitt trigger na placa)
//   KEY[1] = Start manual da inferência (active-low) — alternativa ao
//            auto-start via UART. Pressionar após carregar imagem pelo
//            testbench ou por escrita direta no framebuffer.
//
// Placa: DE2-115 (EP4CE115F29C7, Cyclone IV E)
// ==============================================================================
module fpga_top_de2115 (
    // Clock
    input  wire        CLOCK_50,

    // Botões (active-low, debounce por Schmitt Trigger na placa)
    input  wire [1:0]  KEY,

    // UART (RS-232 RXD via transceiver MAX3232 da DE2-115)
    input  wire        UART_RXD,

    // LEDs Verdes (resultado da classificação)
    output reg  [7:0]  LEDG
);

    // =========================================================================
    // SINAIS INTERNOS
    // =========================================================================
    wire reset = ~KEY[0];           // KEY[0] pressionado = reset ativo

    // Sinal de start manual via KEY[1] (detecção de borda de descida)
    reg  key1_sync_1, key1_sync_2, key1_prev;
    wire key1_pressed;

    // Saídas do cnn_top
    wire [15:0] final_result;
    wire [2:0]  class_id;
    wire        unknown;
    wire        access_done;
    wire        frame_ready;

    // Portas não utilizadas nesta fase (VGA)
    wire [7:0]  vga_rd_data;

    // =========================================================================
    // 1. SINCRONIZAÇÃO DO KEY[1] (anti-metaestabilidade + detecção de borda)
    // =========================================================================
    always @(posedge CLOCK_50 or posedge reset) begin
        if (reset) begin
            key1_sync_1 <= 1'b1;
            key1_sync_2 <= 1'b1;
            key1_prev   <= 1'b1;
        end else begin
            key1_sync_1 <= KEY[1];
            key1_sync_2 <= key1_sync_1;
            key1_prev   <= key1_sync_2;
        end
    end

    // Borda de descida do KEY[1] (active-low): gera pulso de 1 ciclo
    assign key1_pressed = key1_prev & ~key1_sync_2;

    // =========================================================================
    // 2. CNN PIPELINE COMPLETO
    //    O cnn_top já possui UART RX integrado, controle de framebuffer,
    //    e auto-start quando 1024 bytes são recebidos via UART.
    // =========================================================================
    cnn_top cnn_inst (
        .clk           (CLOCK_50),
        .rst           (reset),

        // UART — conectado diretamente ao pino físico
        .rx_pin        (UART_RXD),

        // Start manual via botão KEY[1] (complementar ao auto-start UART)
        .start_system  (key1_pressed),

        // Portas de escrita direta no framebuffer (não usadas nesta fase)
        .fb_wr_en      (1'b0),
        .fb_wr_addr    (10'd0),
        .fb_wr_data    (8'd0),

        // Portas VGA (não usadas nesta fase)
        .vga_rd_en     (1'b0),
        .vga_rd_addr   (10'd0),
        .vga_rd_data   (vga_rd_data),

        // Saídas de resultado
        .final_result  (final_result),
        .class_id      (class_id),
        .unknown       (unknown),
        .access_done   (access_done),
        .frame_ready   (frame_ready)
    );

    // =========================================================================
    // 3. LATCH DOS LEDs — Mantém o resultado visível após a inferência
    //    Os sinais class_id/unknown do cnn_top são válidos apenas durante
    //    o pulso de access_done (1 ciclo). Este bloco captura e retém
    //    os valores nos LEDs até o próximo reset ou nova inferência.
    // =========================================================================
    always @(posedge CLOCK_50 or posedge reset) begin
        if (reset) begin
            LEDG <= 8'd0;
        end else begin
            if (access_done) begin
                // Captura o resultado no instante exato do access_done
                LEDG[2:0] <= class_id;
                LEDG[5:3] <= 3'b000;        // Reservados
                LEDG[6]   <= 1'b1;          // Sinaliza inferência concluída
                LEDG[7]   <= unknown;
            end
            // Indicação visual de frame recebido (pisca brevemente)
            // Quando um novo frame é recebido, apaga LEDG[6] para indicar
            // que uma nova inferência será realizada
            if (frame_ready && !access_done) begin
                LEDG[6] <= 1'b0;
            end
        end
    end

endmodule
