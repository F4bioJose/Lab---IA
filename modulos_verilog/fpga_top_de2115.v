// ==============================================================================
// Módulo: fpga_top_de2115
// Descrição: Top-level sintetizável para inferência CNN na DE2-115.
//            Recebe 1 imagem 32×32 grayscale via UART serial,
//            executa a inferência completa pelo pipeline cnn_top (19 classes),
//            e exibe o resultado da classificação nos LEDs verdes (LEDG).
//
// Mapeamento dos LEDs:
//   LEDG[4:0] = class_id (0 = Desconhecido; 1–18 = pessoa identificada)
//   LEDG[5]   = debug_frame_nonzero (frame recebido com pixels não-nulos)
//   LEDG[6]   = access_done (inferência concluída — latchado)
//   LEDG[7]   = unknown (1 quando a rede prediz a classe 0 = Desconhecido)
//
// Controles:
//   KEY[0] = Reset global (active-low, com Schmitt trigger na placa)
//   KEY[1] = Start manual da inferência (active-low)
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

    // LEDs Verdes (resultado da classificação — 8 LEDs disponíveis)
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
    wire [4:0]  class_id;           // 0 = Desconhecido; 1–18 = pessoa identificada
    wire        unknown;
    wire        access_done;
    wire        frame_ready;
    wire        debug_weights_nonzero;
    wire        debug_frame_nonzero;

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
    // 2. CNN PIPELINE COMPLETO (19 classes)
    // =========================================================================
    cnn_top cnn_inst (
        .clk           (CLOCK_50),
        .rst           (reset),

        // UART — conectado diretamente ao pino físico
        .rx_pin        (UART_RXD),

        // Start manual via botão KEY[1]
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
        .frame_ready   (frame_ready),
        .debug_weights_nonzero (debug_weights_nonzero),
        .debug_frame_nonzero   (debug_frame_nonzero)
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
                LEDG[4:0] <= class_id;          // 0 = Desconhecido; 1–18 = pessoa
                LEDG[5]   <= debug_frame_nonzero;
                LEDG[6]   <= 1'b1;              // Sinaliza inferência concluída
                LEDG[7]   <= unknown;            // 1 = classe 0 predita (Desconhecido)
            end
            // Apaga LEDG[6] quando novo frame começa a ser processado
            if (frame_ready && !access_done) begin
                LEDG[6] <= 1'b0;
            end
        end
    end

endmodule
