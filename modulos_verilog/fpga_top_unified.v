// ==============================================================================
// Módulo: fpga_top_unified
// Descrição: Top-level unificado — Inferência CNN + Exibição VGA.
//            Recebe 1 imagem 32×32 grayscale via UART serial (115200 baud),
//            exibe no monitor VGA (scaling 12×: 384×384 centrado em 640×480),
//            executa inferência CNN (19 classes) e exibe o nome da classe
//            predita via sprite de texto na parte inferior da tela.
//
// Comportamento do sprite de nome:
//   - Antes/durante inferência: exibe "Desconhecido" (classe 0, vermelho)
//   - Após inferência concluída:
//       Se reconhecido (class_id 1–18) → nome da pessoa (verde)
//       Se desconhecido (class_id 0)   → "Desconhecido" (vermelho)
//   - Ao receber nova imagem (frame_ready): reseta para classe 0
//
// Mapeamento de Classes (ordem Keras):
//   0 = Desconhecido | 1 = Igor | 2 = Joao | 3 = Jose Henrique | 4 = Julia
//   5 = Lucio | 6 = Naira | 7 = Rafael | 8 = Samuel | 9 = Yuri
//   10 = Anna Carol | 11 = Bruno | 12 = Diego | 13 = Eduardo | 14 = Fabio
//   15 = Felipe | 16 = Gabriel | 17 = Horacio | 18 = Hugo
//
// Controles:
//   KEY[0] = Reset global (active-low, com Schmitt trigger na placa)
//
// LEDs:
//   LEDG[4:0] = class_id latched (0–18)
//   LEDG[5]   = debug_frame_nonzero
//   LEDG[6]   = inferência concluída (latched)
//   LEDG[7]   = unknown (classe 0 predita)
//
// Placa: DE2-115 (EP4CE115F29C7, Cyclone IV E)
// ==============================================================================
module fpga_top_unified (
    // Clock principal
    input  wire        CLOCK_50,

    // Botão de reset (active-low, debounce por Schmitt Trigger na placa)
    input  wire        KEY,

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

    // LEDs Verdes (resultado da classificação)
    output reg  [7:0]  LEDG
);

    // =========================================================================
    // 1. SINAIS INTERNOS
    // =========================================================================
    wire reset = ~KEY;              // KEY pressionado = reset ativo

    // --- Domínio VGA (25 MHz) ---
    wire clk_25mhz;
    wire pll_locked;

    // --- Saídas do cnn_top ---
    wire [15:0] final_result;
    wire [4:0]  class_id;           // 0 = Desconhecido; 1–18 = pessoa
    wire        unknown;
    wire        access_done;
    wire        frame_ready;
    wire        debug_weights_nonzero;
    wire        debug_frame_nonzero;
    wire        frame_mode;         // 0 = vídeo (128×128), 1 = rosto (32×32)

    // --- Portas de escrita do video framebuffer (geradas pelo cnn_top) ---
    wire        video_fb_wr_en;
    wire [13:0] video_fb_wr_addr;
    wire [7:0]  video_fb_wr_data;

    // --- Porta VGA do framebuffer 32×32 (dentro do cnn_top) ---
    wire [7:0]  vga_rd_data;        // Pixel lido do framebuffer 32×32
    reg  [9:0]  vga_rd_addr;        // Endereço de leitura VGA (32×32 = 10 bits)

    // --- Porta VGA do framebuffer 128×128 ---
    wire [7:0]  video_vga_rd_data;  // Pixel lido do framebuffer 128×128
    reg  [13:0] video_vga_rd_addr;  // Endereço de leitura VGA (128×128 = 14 bits)

    // --- Modo de exibição VGA (latched no domínio 25 MHz) ---
    reg         display_mode;       // 0 = vídeo 128×128, 1 = rosto 32×32

    // --- VGA timing ---
    wire        hsync_raw, vsync_raw, video_on_raw;
    wire [9:0]  pixel_x, pixel_y;

    // --- Orquestrador: classe exibida no sprite ---
    reg  [4:0]  display_class_id;   // Classe atualmente exibida (0 = Desconhecido)
    wire        access_granted;     // Cor do sprite: verde (reconhecido) ou vermelho

    // =========================================================================
    // 2. PLL: 50 MHz → 25 MHz (pixel clock VGA 640×480 @ 60Hz)
    // =========================================================================
    vga_pll pll_inst (
        .areset (reset),
        .inclk0 (CLOCK_50),
        .c0     (clk_25mhz),
        .locked (pll_locked)
    );

    // =========================================================================
    // 3. VGA_CLK: DDR output register para conduzir o pixel clock ao pino
    //    VGA_CLK por roteamento dedicado de clock.
    //    datain_h=1 na borda de subida, datain_l=0 na descida → reproduz clock.
    // =========================================================================
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

    // =========================================================================
    // 4. VGA_SYNC_N: Registrador preservado para evitar otimização indesejada.
    //    Valor funcional: sempre 0 (sync-on-green desativado, padrão do DAC).
    // =========================================================================
    (* preserve *) reg vga_sync_n_reg;
    always @(posedge clk_25mhz or posedge reset) begin
        if (reset)
            vga_sync_n_reg <= 1'b0;
        else
            vga_sync_n_reg <= 1'b0;
    end
    assign VGA_SYNC_N = vga_sync_n_reg;

    // =========================================================================
    // 5. CNN PIPELINE COMPLETO (19 classes)
    //    Recebe imagem via UART interna, processa e gera class_id.
    //    A porta VGA do framebuffer 32×32 é conectada ao barramento VGA.
    //    As portas do video framebuffer 128×128 são roteadas ao módulo externo.
    // =========================================================================
    cnn_top cnn_inst (
        .clk           (CLOCK_50),
        .rst           (reset),

        // UART — conectado ao pino físico
        .rx_pin        (UART_RXD),

        // Start manual desabilitado — inferência é 100% automática via UART
        .start_system  (1'b0),

        // Portas de escrita externa (não usadas — UART interna ao cnn_top)
        .fb_wr_en      (1'b0),
        .fb_wr_addr    (10'd0),
        .fb_wr_data    (8'd0),

        // Portas VGA do framebuffer 32×32 — conectadas ao barramento VGA
        .vga_rd_en     (1'b1),              // Leitura contínua habilitada
        .vga_rd_addr   (vga_rd_addr),       // Endereço gerado pela lógica VGA
        .vga_rd_data   (vga_rd_data),       // Pixel para renderização

        // Portas de escrita do video framebuffer 128×128
        .video_fb_wr_en   (video_fb_wr_en),
        .video_fb_wr_addr (video_fb_wr_addr),
        .video_fb_wr_data (video_fb_wr_data),

        // Modo do frame atual
        .frame_mode    (frame_mode),

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
    // 5b. VIDEO FRAMEBUFFER 128×128 (apenas para exibição VGA)
    // =========================================================================
    framebuffer_128x128 video_fb_inst (
        .clk          (CLOCK_50),
        .rst          (reset),
        .wr_en        (video_fb_wr_en),
        .wr_addr      (video_fb_wr_addr),
        .wr_data      (video_fb_wr_data),
        .vga_rd_en    (1'b1),
        .vga_rd_addr  (video_vga_rd_addr),
        .vga_rd_data  (video_vga_rd_data),
        .frame_ready  ()                   // Não utilizado
    );

    // =========================================================================
    // 6. VGA SYNC: Gerador de temporização 640×480 @ 60Hz
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
    // 7. MAPEAMENTO DE COORDENADAS — Dual mode:
    //    Modo rosto:  32×32 × 12 = 384×384 centrada em 640×480
    //    Modo vídeo: 128×128 × 3 = 384×384 centrada em 640×480
    // =========================================================================
    // Offsets e span são idênticos em ambos os modos (mesma região 384×384).
    //   H_OFFSET = (640 - 384) / 2 = 128
    //   V_OFFSET = (480 - 384) / 2 = 48
    //
    // Modo rosto (÷12): ×5462 >> 16
    //   Constante 5462 verificada: zero erros para 0–383.
    //
    // Modo vídeo (÷3): ×21845 >> 16
    //   21845 = ceil(65536/3) → 383 × 21845 >> 16 = 127 ✓

    localparam H_OFFSET   = 10'd128;
    localparam V_OFFSET   = 10'd48;
    localparam IMG_SPAN   = 10'd384;    // Ambos os modos usam 384×384

    // Sprite de texto (ROM): 256×32 posicionado abaixo da imagem
    localparam H_START_SPRITE = 10'd192;
    localparam V_START_SPRITE = 10'd440;
    localparam W_SPRITE       = 10'd256;
    localparam H_SPRITE       = 10'd32;

    // --- Sincronização do display_mode (50 MHz → 25 MHz) ---
    reg frame_mode_sync;
    always @(posedge clk_25mhz or posedge reset) begin
        if (reset) begin
            frame_mode_sync <= 1'b0;
            display_mode    <= 1'b0;
        end else begin
            frame_mode_sync <= frame_mode;
            display_mode    <= frame_mode_sync;
        end
    end

    // --- Detecção de área da imagem ---
    wire in_image = (pixel_x >= H_OFFSET) && (pixel_x < (H_OFFSET + IMG_SPAN)) &&
                    (pixel_y >= V_OFFSET) && (pixel_y < (V_OFFSET + IMG_SPAN));

    // --- Detecção de área do sprite (oculto em modo vídeo) ---
    wire in_sprite_window = display_mode &&
                            (pixel_x >= H_START_SPRITE) && (pixel_x < (H_START_SPRITE + W_SPRITE)) &&
                            (pixel_y >= V_START_SPRITE) && (pixel_y < (V_START_SPRITE + H_SPRITE));

    // --- Coordenadas locais dentro da área 384×384 ---
    wire [9:0]  local_x = pixel_x - H_OFFSET;   // 0–383 dentro da imagem
    wire [9:0]  local_y = pixel_y - V_OFFSET;    // 0–383 dentro da imagem

    // --- Modo rosto: Divisão por 12 → ×5462 >> 16 → 5 bits (0–31) ---
    wire [25:0] fb_x_full = local_x * 16'd5462;
    wire [25:0] fb_y_full = local_y * 16'd5462;
    wire [4:0]  fb_x = fb_x_full[20:16];
    wire [4:0]  fb_y = fb_y_full[20:16];

    // --- Modo vídeo: Divisão por 3 → ×21845 >> 16 → 7 bits (0–127) ---
    wire [25:0] vid_x_full = local_x * 16'd21845;
    wire [25:0] vid_y_full = local_y * 16'd21845;
    wire [6:0]  vid_x = vid_x_full[22:16];
    wire [6:0]  vid_y = vid_y_full[22:16];

    // --- Endereço linear no framebuffer 32×32 ---
    always @(*) begin
        if (in_image)
            vga_rd_addr = {fb_y, fb_x};  // 10 bits: fb_y * 32 + fb_x
        else
            vga_rd_addr = 10'd0;
    end

    // --- Endereço linear no framebuffer 128×128 ---
    always @(*) begin
        if (in_image)
            video_vga_rd_addr = {vid_y, vid_x};  // 14 bits: vid_y * 128 + vid_x
        else
            video_vga_rd_addr = 14'd0;
    end

    // =========================================================================
    // 8. MAPEAMENTO DE COORDENADAS DO SPRITE (ROM de nomes)
    // =========================================================================
    // Cada classe ocupa um slot de 256×32 = 8192 pixels (1 bit/pixel) na ROM.
    // Endereço = {display_class_id, 13'd0} + (sprite_y × 256 + sprite_x)

    wire [7:0]  sprite_x = (pixel_x - H_START_SPRITE);  // 0–255
    wire [4:0]  sprite_y = (pixel_y - V_START_SPRITE);   // 0–31
    wire [12:0] pixel_atual_offset = (sprite_y * 10'd256) + sprite_x;
    wire [17:0] endereco_base_aluno = {display_class_id, 13'd0};
    wire [17:0] endereco_mega_rom = endereco_base_aluno + pixel_atual_offset;
    wire        pixel_do_sprite;

    rom_sprites rom_sprites_inst (
        .clock   (clk_25mhz),
        .address (endereco_mega_rom),
        .q       (pixel_do_sprite)
    );

    // =========================================================================
    // 9. COMPENSAÇÃO DE LATÊNCIA (1 ciclo de pipeline a 25 MHz)
    //    A leitura do framebuffer (50 MHz) e da ROM de sprites possuem
    //    latência registrada. Os sinais de controle são atrasados igualmente
    //    para manter o alinhamento de dados com a renderização.
    // =========================================================================
    reg hs_d, vs_d, video_on_d, in_image_d, in_sprite_window_d, display_mode_d;

    always @(posedge clk_25mhz or posedge reset) begin
        if (reset) begin
            hs_d               <= 1'b1;
            vs_d               <= 1'b1;
            video_on_d         <= 1'b0;
            in_image_d         <= 1'b0;
            in_sprite_window_d <= 1'b0;
            display_mode_d     <= 1'b0;
        end else begin
            hs_d               <= hsync_raw;
            vs_d               <= vsync_raw;
            video_on_d         <= video_on_raw;
            in_image_d         <= in_image;
            in_sprite_window_d <= in_sprite_window;
            display_mode_d     <= display_mode;
        end
    end

    assign VGA_HS      = hs_d;
    assign VGA_VS      = vs_d;
    assign VGA_BLANK_N = video_on_d;

    // --- MUX de pixel: seleciona entre framebuffer 32×32 e 128×128 ---
    wire [7:0] pixel_data = display_mode_d ? vga_rd_data : video_vga_rd_data;

    // =========================================================================
    // 10. RENDERIZAÇÃO VGA — Composição final dos pixels
    // =========================================================================
    // Prioridade: fora da tela > imagem > sprite > fundo preto
    // Sprite: verde se reconhecido (access_granted), vermelho se desconhecido
    // Sprite oculto em modo vídeo (in_sprite_window já inclui display_mode)

    always @(*) begin
        if (!video_on_d) begin
            // Fora da área visível: preto obrigatório
            VGA_R = 8'd0;
            VGA_G = 8'd0;
            VGA_B = 8'd0;
        end
        else if (in_image_d) begin
            // Área da imagem: escala de cinza (fonte selecionada por display_mode)
            VGA_R = pixel_data;
            VGA_G = pixel_data;
            VGA_B = pixel_data;
        end
        else if (in_sprite_window_d) begin
            if (pixel_do_sprite == 1'b1) begin
                // Pixel do texto: verde (reconhecido) ou vermelho (desconhecido)
                VGA_R = (access_granted) ? 8'h00 : 8'hFF;
                VGA_G = (access_granted) ? 8'hFF : 8'h00;
                VGA_B = 8'h00;
            end else begin
                // Fundo do sprite: cinza escuro
                VGA_R = 8'h10;
                VGA_G = 8'h10;
                VGA_B = 8'h10;
            end
        end
        else begin
            // Fundo: preto
            VGA_R = 8'd0;
            VGA_G = 8'd0;
            VGA_B = 8'd0;
        end
    end

    // =========================================================================
    // 11. FSM ORQUESTRADORA — Controle do sprite de nome
    // =========================================================================
    // Lógica central de integração CNN ↔ VGA.
    //
    // Regras:
    //   - No reset: display_class_id = 0 ("Desconhecido")
    //   - Quando frame_ready pulsa (nova imagem completa, inferência prestes
    //     a iniciar): display_class_id volta a 0 ("Desconhecido")
    //   - Quando access_done pulsa (inferência concluída): display_class_id
    //     captura o class_id da CNN
    //   - access_granted é derivado automaticamente: 1 se reconhecido (≠0)
    //
    // O registrador mantém o último resultado até nova imagem ser recebida.

    assign access_granted = (display_class_id != 5'd0);

    always @(posedge CLOCK_50 or posedge reset) begin
        if (reset) begin
            display_class_id <= 5'd0;
        end else begin
            // Ao receber nova imagem completa: reseta para "Desconhecido"
            // (frame_ready pulsa brevemente quando o 1024° byte é escrito)
            if (frame_ready) begin
                display_class_id <= 5'd0;
            end

            // Ao concluir a inferência: captura o resultado da CNN
            // (access_done pulsa 1 ciclo quando o pipeline termina)
            // Prioridade implícita: access_done não coincide com frame_ready
            if (access_done) begin
                display_class_id <= class_id;
            end
        end
    end

    // =========================================================================
    // 12. LATCH DOS LEDs — Mantém o resultado visível após a inferência
    // =========================================================================
    // Idêntico ao comportamento original do fpga_top_de2115.v.
    // Os sinais class_id/unknown do cnn_top são válidos durante o pulso
    // de access_done (1 ciclo). Este bloco captura e retém os valores
    // nos LEDs até o próximo reset ou nova inferência.

    always @(posedge CLOCK_50 or posedge reset) begin
        if (reset) begin
            LEDG <= 8'd0;
        end else begin
            if (access_done) begin
                LEDG[4:0] <= class_id;
                LEDG[5]   <= debug_frame_nonzero;
                LEDG[6]   <= 1'b1;              // Inferência concluída
                LEDG[7]   <= unknown;
            end
            // Apaga LEDG[6] quando novo frame começa
            if (frame_ready && !access_done) begin
                LEDG[6] <= 1'b0;
            end
        end
    end

endmodule
