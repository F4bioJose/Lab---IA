// ==============================================================================
// Módulo: system_top
// Descrição: Top-level sintetizável do Artefato 3 — Pipeline UART → VGA.
//            Recebe imagem 32×32 grayscale via UART serial, armazena em 
//            framebuffer interno (altsyncram dual-port) e exibe no monitor VGA 
//            com scaling 8× (256×256 pixels centrados em 640×480).
// Placa: DE2-115 (EP4CE115F29C7)
// ==============================================================================
module system_top (
    // Clock & Reset
    input  wire        CLOCK_50,
    input  wire        KEY,         // KEY[0] na placa = reset (active-low)

    // UART (RS-232 RXD via transceiver MAX3232 da DE2-115)
    input  wire        UART_RXD,

    // VGA DAC (ADV7123 na DE2-115)
    output wire        VGA_CLK,
    output wire        VGA_HS,
    output wire        VGA_VS,
    output wire        VGA_BLANK_N,
    output wire        VGA_SYNC_N,
    output reg  [7:0]  VGA_R,
    output reg  [7:0]  VGA_G,
    output reg  [7:0]  VGA_B,

    // LEDs de depuração
    output wire        LEDG,        // LEDG[0] na placa = frame recebido
    output wire [9:0]  LEDR         // LEDR[9:0] = progresso da escrita UART
);

    // =========================================================================
    // SINAIS INTERNOS
    // =========================================================================
    wire reset = ~KEY;              // Botão KEY[0] pressionado = reset ativo

    // Domínio de clock VGA (25.175 MHz)
    wire clk_25mhz;

    // UART
    wire [7:0] uart_data;
    wire       uart_valid;
    reg  [1:0] rx_sync;             // Double-sync anti-metaestabilidade
    reg [13:0] uart_wr_addr;        // Endereço de escrita auto-incrementado
    reg        frame_received;      // Flag: pelo menos 1 frame completo recebido

    // VGA timing
    wire       hsync_raw, vsync_raw, video_on_raw;
    wire [9:0] pixel_x, pixel_y;

    // Framebuffer read
    wire [7:0] fb_rd_data;

    // =========================================================================
    // 1. PLL: 50 MHz → 25.175 MHz (pixel clock VGA 640×480 @ 60Hz)
    // =========================================================================
    wire pll_locked;

    vga_pll pll_inst (
        .areset (reset),
        .inclk0 (CLOCK_50),
        .c0     (clk_25mhz),
        .locked (pll_locked)
    );

    // VGA_CLK: Usa DDR output register para conduzir o pixel clock ao pino
    // VGA_CLK por roteamento dedicado de clock (evita roteamento genérico).
    // datain_h=1 na borda de subida, datain_l=0 na borda de descida → reproduz o clock.
    altddio_out #(
        .extend_oe_disable      ("OFF"),
        .intended_device_family ("Cyclone IV E"),
        .invert_output          ("OFF"),
        .lpm_hint               ("UNUSED"),
        .lpm_type               ("altddio_out"),
        .oe_reg                 ("UNREGISTERED"),
        .power_up_high          ("OFF"),
        .width                  (1)
    ) vga_clk_ddr (
        .datain_h    (1'b1),
        .datain_l    (1'b0),
        .outclock    (clk_25mhz),
        .dataout     (VGA_CLK),
        .aclr        (1'b0),
        .aset        (1'b0),
        .oe          (1'b1),
        .outclocken  (1'b1),
        .sclr        (1'b0),
        .sset        (1'b0)
    );

    // VGA_SYNC_N: Registrador preservado para evitar que o sintetizador
    // otimize a constante e reporte "stuck at GND".
    // Valor funcional: sempre 0 (sync-on-green desativado, padrão do DAC).
    (* preserve *) reg vga_sync_n_reg;
    always @(posedge clk_25mhz or posedge reset) begin
        if (reset)
            vga_sync_n_reg <= 1'b0;
        else
            vga_sync_n_reg <= 1'b0;
    end
    assign VGA_SYNC_N = vga_sync_n_reg;

    // =========================================================================
    // 2. UART RX: Recebe bytes seriais do PC a 115200 baud
    // =========================================================================
    // Double-sync do sinal RX para evitar metaestabilidade
    always @(posedge CLOCK_50) begin
        if (reset) begin
            rx_sync <= 2'b11;       // Linha idle = HIGH
        end else begin
            rx_sync <= {rx_sync[0], UART_RXD};
        end
    end

    uart_rx #(
        .CLK_FREQ  (50000000),
        .BAUD_RATE (115200)
    ) uart_rx_inst (
        .clk       (CLOCK_50),
        .rst       (reset),
        .rx_pin    (rx_sync[1]),     // Sinal sincronizado
        .data_out  (uart_data),
        .data_valid(uart_valid)
    );

    // =========================================================================
    // 3. FRAMEBUFFER: altsyncram dual-port explícito (16384 × 8 bits)
    //    - Porta A (escrita): domínio CLOCK_50 (UART)
    //    - Porta B (leitura): domínio clk_25mhz (VGA)
    // =========================================================================
    reg [13:0] fb_rd_addr;

    altsyncram #(
        .operation_mode                 ("DUAL_PORT"),
        .width_a                        (8),
        .widthad_a                      (14),
        .numwords_a                     (16384),
        .width_b                        (8),
        .widthad_b                      (14),
        .numwords_b                     (16384),
        .address_reg_b                  ("CLOCK1"),
        .outdata_reg_b                  ("CLOCK1"),
        .intended_device_family         ("Cyclone IV E"),
        .lpm_type                       ("altsyncram"),
        .read_during_write_mode_mixed_ports ("DONT_CARE")
    ) framebuffer (
        // Porta A — Escrita (50 MHz, UART)
        .clock0      (CLOCK_50),
        .address_a   (uart_wr_addr),
        .data_a      (uart_data),
        .wren_a      (uart_valid),

        // Porta B — Leitura (25 MHz, VGA)
        .clock1      (clk_25mhz),
        .address_b   (fb_rd_addr),
        .q_b         (fb_rd_data)
    );

    // Controle de endereço de escrita UART
    always @(posedge CLOCK_50) begin
        if (reset) begin
            uart_wr_addr   <= 14'd0;
            frame_received <= 1'b0;
        end else if (uart_valid) begin
            if (uart_wr_addr == 14'd16383) begin // 128*128 - 1
                uart_wr_addr   <= 14'd0;
                frame_received <= 1'b1;
            end else begin
                uart_wr_addr <= uart_wr_addr + 1'b1;
            end
        end
    end

    // =========================================================================
    // 4. VGA SYNC: Gerador de temporização 640×480 @ 60Hz
    // =========================================================================
    vga_sync vga_sync_inst (
        .clk     (clk_25mhz),
        .rst     (reset),
        .hsync   (hsync_raw),
        .vsync   (vsync_raw),
        .pixel_x (pixel_x),
        .pixel_y (pixel_y),
        .video_on(video_on_raw)
    );

    // =========================================================================
    // 5. MAPEAMENTO DE COORDENADAS — Imagem 128x128 (sem scaling)
    // =========================================================================
    // A imagem original tem 128x128 pixels e é exibida 1:1.
    // A imagem é centralizada na tela de 640x480.
    localparam IMG_WIDTH  = 128;
    localparam IMG_HEIGHT = 128;
    localparam H_OFFSET   = (640 - IMG_WIDTH) / 2;  // 256
    localparam V_OFFSET   = (480 - IMG_HEIGHT) / 2; // 176

    wire [6:0] fb_x;
    wire [6:0] fb_y;
    
    // Mapeia as coordenadas da tela para as do framebuffer (subtraindo o offset)
    assign fb_x = pixel_x - H_OFFSET;
    assign fb_y = pixel_y - V_OFFSET;

    wire in_image = (pixel_x >= H_OFFSET) && (pixel_x < (H_OFFSET + IMG_WIDTH)) &&
                    (pixel_y >= V_OFFSET) && (pixel_y < (V_OFFSET + IMG_HEIGHT));

    // Endereço linear no framebuffer 128x128
    // Endereço = y * 128 + x  (Otimização: y << 7)
    always @(*) begin
        if (in_image)
            fb_rd_addr = (fb_y << 7) + fb_x;
        else
            fb_rd_addr = 14'd0;
    end

    // =========================================================================
    // 6. COMPENSAÇÃO DE LATÊNCIA (1 ciclo de pipeline)
    //    A leitura do altsyncram possui 1 ciclo de latência registrada.
    //    Todos os sinais de controle devem ser atrasados igualmente.
    // =========================================================================
    reg hs_d, vs_d, video_on_d, in_image_d;

    always @(posedge clk_25mhz or posedge reset) begin
        if (reset) begin
            hs_d       <= 1'b1;
            vs_d       <= 1'b1;
            video_on_d <= 1'b0;
            in_image_d <= 1'b0;
        end else begin
            hs_d       <= hsync_raw;
            vs_d       <= vsync_raw;
            video_on_d <= video_on_raw;
            in_image_d <= in_image;
        end
    end

    assign VGA_HS      = hs_d;
    assign VGA_VS      = vs_d;
    assign VGA_BLANK_N = video_on_d;

    // =========================================================================
    // 7. RENDERIZAÇÃO VGA — Composição final dos pixels
    // =========================================================================
    always @(*) begin
        if (!video_on_d) begin
            // Fora da área visível: preto obrigatório
            VGA_R = 8'd0;
            VGA_G = 8'd0;
            VGA_B = 8'd0;
        end
        else if (in_image_d) begin
            // Área da imagem: escala de cinza do framebuffer
            VGA_R = fb_rd_data;
            VGA_G = fb_rd_data;
            VGA_B = fb_rd_data;
        end
        else begin
            // Fundo: preto
            VGA_R = 8'd0;
            VGA_G = 8'd0;
            VGA_B = 8'd0;
        end
    end

    // =========================================================================
    // 8. LEDs DE DEPURAÇÃO
    // =========================================================================
    assign LEDG       = frame_received;
    assign LEDR       = uart_wr_addr;

endmodule
