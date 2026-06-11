// ==============================================================================
// Módulo: system_top
// Descrição: Top-level sintetizável — Pipeline UART → VGA.
//            Recebe imagem 64×64 grayscale via UART serial a 230400 baud,
//            armazena em framebuffer interno (altsyncram dual-port) e exibe
//            no monitor VGA com scaling 6× (384×384 centrada em 640×480).
// Placa: DE2-115 (EP4CE115F29C7) 
// ==============================================================================
module system_top (
    // Clock & Reset
    input  wire        CLOCK_50,
    input  wire        KEY,         // KEY[0] na placa = reset (active-low)

    // UART (RS-232 RXD via transceiver MAX3232 da DE2-115)
    input  wire        UART_RXD,
	 
	 input  wire [17:0] SW,
	 
	 // ID das Classes (rostos)
	 //wire         class_id, 

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
    reg [11:0] uart_wr_addr;        // Endereço de escrita auto-incrementado (64*64 = 4096)
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
    // 2. UART RX: Recebe bytes seriais do PC a 230400 baud
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
        .BAUD_RATE (230400)
    ) uart_rx_inst (
        .clk       (CLOCK_50),
        .rst       (reset),
        .rx_pin    (rx_sync[1]),     // Sinal sincronizado
        .data_out  (uart_data),
        .data_valid(uart_valid)
    );

    // =========================================================================
    // 3. FRAMEBUFFER: altsyncram dual-port explícito (4096 × 8 bits)
    //    - Porta A (escrita): domínio CLOCK_50 (UART)
    //    - Porta B (leitura): domínio clk_25mhz (VGA)
    // =========================================================================
    reg [11:0] fb_rd_addr;

    altsyncram #(
        .operation_mode                 ("DUAL_PORT"),
        .width_a                        (8),
        .widthad_a                      (12),
        .numwords_a                     (4096),
        .width_b                        (8),
        .widthad_b                      (12),
        .numwords_b                     (4096),
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
            uart_wr_addr   <= 12'd0;
            frame_received <= 1'b0;
        end else if (uart_valid) begin
            if (uart_wr_addr == 12'd4095) begin // 64*64 - 1
                uart_wr_addr   <= 12'd0;
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
    // 5. MAPEAMENTO DE COORDENADAS — Scaling 6×: 64×64 → 384×384 centrada
    // =========================================================================
    // A imagem 64×64 é ampliada 6× = 384×384 pixels, centrada em 640×480.
    //   H_OFFSET = (640 - 384) / 2 = 128
    //   V_OFFSET = (480 - 384) / 2 = 48
    //   fb_x = (pixel_x - 128) / 6  ≈  (pixel_x - 128) * 10923 >> 16
    //   fb_y = (pixel_y -  48) / 6  ≈  (pixel_y -  48) * 10923 >> 16
    //
    // Constante 10923 verificada: zero erros para 0–383 (teste exaustivo).

    localparam H_OFFSET = 10'd128;
    localparam V_OFFSET = 10'd48;
    localparam IMG_SPAN = 10'd384;  // 64 * 6
	 
	 // 1. Sprite de texto (ROM): 256x32 
	 localparam H_START_SPRITE = 10'd192;
	 localparam V_START_SPRITE = 10'd440;
	 localparam W_SPRITE = 10'd256;
	 localparam H_SPRITE = 10'd32; 
	 
	 // -=-=-=-=-=-=-=-=-=-

    wire in_image = (pixel_x >= H_OFFSET) && (pixel_x < (H_OFFSET + IMG_SPAN)) &&
                    (pixel_y >= V_OFFSET) && (pixel_y < (V_OFFSET + IMG_SPAN));
						  
	 // mesma logica do in_image
	 wire in_sprite_window = (pixel_x >= H_START_SPRITE) && (pixel_x < (H_START_SPRITE + W_SPRITE)) &&
                            (pixel_y >= V_START_SPRITE) && (pixel_y < (V_START_SPRITE + H_SPRITE));

    wire [9:0]  local_x = pixel_x - H_OFFSET;   // 0–383 dentro da imagem
    wire [9:0]  local_y = pixel_y - V_OFFSET;    // 0–383 dentro da imagem
    wire [25:0] fb_x_full = local_x * 16'd10923; // 10 bits × 16 bits = 26 bits
    wire [25:0] fb_y_full = local_y * 16'd10923;
    wire [5:0]  fb_x = fb_x_full[21:16];         // >> 16, pegamos 6 bits (0–63)
    wire [5:0]  fb_y = fb_y_full[21:16];

    // Endereço linear no framebuffer 64×64
    // Endereço = fb_y * 64 + fb_x  (Otimização: fb_y << 6)
    always @(*) begin
        if (in_image)
            fb_rd_addr = {fb_y, fb_x};  // Concatenação = fb_y * 64 + fb_x
        else
            fb_rd_addr = 12'd0;
    end
	 
	 
	 // 5.1 Mapeamento de coordenadas para nomes 
	 
	 wire access_granted = SW[17]; 
	 // Fios dos Switches
	 wire [4:0] class_id_simulado = SW[4:0]; // total de nomes e 19 - necessario 5 bits
	 
	 wire [7:0] sprite_x = (pixel_x - H_START_SPRITE); // representar 256 
	 wire [4:0] sprite_y = (pixel_y - V_START_SPRITE); // representar 32 
	 wire [12:0] pixel_atual_offset = (sprite_y * 10'd256) + sprite_x; 
	 wire [17:0] endereco_base_aluno = {class_id_simulado, 13'd0}; //big shift de 13 casas
	 wire [17:0] endereco_mega_rom = endereco_base_aluno + pixel_atual_offset; 
	 wire pixel_do_sprite;
	  
	  rom_sprites rom_sprites_inst (
        .clock   (clk_25mhz),
        .address (endereco_mega_rom),
        .q       (pixel_do_sprite)
    );

    // =========================================================================
    // 6. COMPENSAÇÃO DE LATÊNCIA (1 ciclo de pipeline)
    //    A leitura do altsyncram possui 1 ciclo de latência registrada.
    //    Todos os sinais de controle devem ser atrasados igualmente.
    // =========================================================================
    reg hs_d, vs_d, video_on_d, in_image_d, in_sprite_window_d; 

    always @(posedge clk_25mhz or posedge reset) begin
        if (reset) begin
            hs_d       <= 1'b1;
            vs_d       <= 1'b1;
            video_on_d <= 1'b0;
            in_image_d <= 1'b0;
				in_sprite_window_d <= 1'b0;
        end else begin
            hs_d       <= hsync_raw;
            vs_d       <= vsync_raw;
            video_on_d <= video_on_raw;
            in_image_d <= in_image;
				in_sprite_window_d <= in_sprite_window; 
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
		  else if (in_sprite_window_d) begin
				if (pixel_do_sprite == 1'b1) begin
					
					VGA_R = (access_granted) ? 8'h00 : 8'hFF;
					VGA_G = (access_granted) ? 8'hFF : 8'h00;
					VGA_B = 8'h00;
				end else begin
					VGA_R = 8'h10; VGA_G = 8'h10; VGA_B = 8'h10; 
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
    // 8. LEDs DE DEPURAÇÃO
    // =========================================================================
    assign LEDG       = frame_received;
    assign LEDR       = uart_wr_addr[11:2];

endmodule
