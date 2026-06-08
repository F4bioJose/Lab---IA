// ==============================================================================
// Módulo: cnn_top
// Descrição: Top-level da arquitetura Tiny-CNN. Instancia, conecta e orquestra
//            todos os submódulos da rede: Framebuffer, Line Buffer, Convolução,
//            Max Pooling, Flatten, Camada Densa (19 classes) e Decisão
//            (Argmax puro — sem threshold, desconhecido = classe 0 nativa).
//
// Mapeamento de Classes (ordem Keras — string sort das pastas do dataset):
//   0 = Desconhecido | 1 = Igor | 2 = Joao | 3 = Jose Henrique | 4 = Julia
//   5 = Lucio | 6 = Naira | 7 = Rafael | 8 = Samuel | 9 = Yuri
//   10 = Anna Carol | 11 = Bruno | 12 = Diego | 13 = Eduardo | 14 = Fabio
//   15 = Felipe | 16 = Gabriel | 17 = Horacio | 18 = Hugo
// ==============================================================================
module cnn_top (
    input wire clk,
    input wire rst,
    // [PONTO DE INTEGRAÇÃO - UART]
    // RX serial vindo do conversor USB/TTL.
    input wire rx_pin,
    // [SINAL DE CONTROLE EXTERNO]
    // O sinal de início será acionado por um handshake via UART ou registrador de comando.
    input wire start_system,

    // [PONTO DE INTEGRAÇÃO - UART]
    // O módulo Controlador UART deverá ser conectado nessas três portas abaixo.
    // O script em Python enviará os bytes da imagem serialmente, a FSM da UART
    // agrupará em 8 bits e ativará o `fb_wr_en`, incrementando o `fb_wr_addr`
    // a cada ciclo de escrita válido. Atingindo o limite, a inferência dispara.
    input wire fb_wr_en,
    input wire [9:0] fb_wr_addr,
    input wire [7:0] fb_wr_data,

    // [PONTO DE INTEGRAÇÃO - VGA E SRAM EXTERNA]
    // A varredura de exibição de vídeo se conectará aqui.
    input wire vga_rd_en,
    input wire [9:0] vga_rd_addr,
    output wire [7:0] vga_rd_data,

    output wire [15:0] final_result,
    output wire [4:0]  class_id,       // 0 = Desconhecido; 1-18 = pessoa identificada
    output wire        unknown,        // 1 quando class_id == 0 (rede prediz desconhecido)
    output reg         access_done,
    output wire        frame_ready,
    output wire        debug_weights_nonzero,
    output wire        debug_frame_nonzero
);

    // UART RX sincronizado para o clock interno
    reg rx_sync_1;
    reg rx_sync_2;

    wire [7:0] uart_data;
    wire uart_valid;
    reg [9:0] uart_wr_addr;
    reg uart_start_pulse;
    reg uart_frame_pending;

    // Sinal combinacional para escrita imediata no framebuffer
    wire uart_wr_en_comb;

    wire fb_wr_en_int;
    wire [9:0] fb_wr_addr_int;
    wire [7:0] fb_wr_data_int;
    wire start_system_int;

    wire [7:0] fb_rd_data;
    reg [9:0] fb_rd_addr;
    reg fb_rd_en;
    reg fb_rd_en_d;
    reg frame_clear;

    wire window_valid;
    wire [7:0] win [0:8];

    wire conv_valid;
    wire signed [15:0] conv_out_f0;
    wire signed [15:0] conv_out_f1;
    wire signed [15:0] conv_out_f2;
    wire signed [15:0] conv_out_f3;

    wire pool_valid;
    wire signed [15:0] pool_data;

    wire flat_valid;
    wire signed [15:0] flat_data;

    // Atraso de 1 ciclo para sincronizar com a latência da ROM Densa M9K
    reg flat_valid_d;
    reg signed [15:0] flat_data_d;

    // Pesos e biases — 19 classes
    wire signed [7:0] dense_w [0:18];
    wire signed [7:0] dense_b [0:18];
    localparam integer DENSE_ADDR_WIDTH = 14;
    reg [DENSE_ADDR_WIDTH-1:0] dense_addr;

    // Pesos convolucionais
    wire signed [7:0] conv_w0 [0:8];
    wire signed [7:0] conv_w1 [0:8];
    wire signed [7:0] conv_w2 [0:8];
    wire signed [7:0] conv_w3 [0:8];
    wire signed [7:0] conv_b0;
    wire signed [7:0] conv_b1;
    wire signed [7:0] conv_b2;
    wire signed [7:0] conv_b3;

    wire dense_done;
    wire dense_valid;
    wire signed [15:0] dense_scores [0:18];
    wire weights_boot_done;
    wire argmax_valid;

    reg [10:0] rd_req_count;
    reg [10:0] rd_val_count;
    reg [7:0] debug_or_acc;

    // Maquina de estados (FSM) principal para controle do pipeline
    localparam ST_IDLE  = 2'd0;
    localparam ST_READ  = 2'd1;
    localparam ST_WAIT  = 2'd2;
    localparam ST_DONE  = 2'd3;

    reg [1:0] state;

    // Evita o atraso de 1 ciclo que causaria off-by-one no endereço
    assign uart_wr_en_comb = uart_valid && (state == ST_IDLE) && !frame_ready && !uart_frame_pending;

    assign debug_weights_nonzero = |{conv_b0, conv_b1, conv_b2, conv_b3,
                                    dense_b[ 0], dense_b[ 1], dense_b[ 2], dense_b[ 3],
                                    dense_b[ 4], dense_b[ 5], dense_b[ 6], dense_b[ 7],
                                    dense_b[ 8], dense_b[ 9], dense_b[10], dense_b[11],
                                    dense_b[12], dense_b[13], dense_b[14], dense_b[15],
                                    dense_b[16], dense_b[17], dense_b[18]};
    assign debug_frame_nonzero = |debug_or_acc;

    // UART RX: converte serial em byte + pulso de dado valido
    uart_rx uart_rx_inst (
        .clk(clk),
        .rst(rst),
        .rx_pin(rx_sync_2),
        .data_out(uart_data),
        .data_valid(uart_valid)
    );

    assign fb_wr_en_int    = uart_wr_en_comb | fb_wr_en;
    assign fb_wr_addr_int  = uart_wr_en_comb ? uart_wr_addr : fb_wr_addr;
    assign fb_wr_data_int  = uart_wr_en_comb ? uart_data    : fb_wr_data;
    assign start_system_int = start_system | uart_start_pulse;

    // =========================================================================
    // Instanciação e Interconexão dos Componentes do Hardware CNN
    // =========================================================================

    // 1. Framebuffer: Armazena a imagem a ser processada
    framebuffer_32x32 framebuffer_inst (
        .clk(clk),
        .rst(rst),
        .wr_en(fb_wr_en_int),
        .wr_addr(fb_wr_addr_int),
        .wr_data(fb_wr_data_int),
        .rd_en(fb_rd_en),
        .rd_addr(fb_rd_addr),
        .rd_data(fb_rd_data),
        .vga_rd_en(vga_rd_en),
        .vga_rd_addr(vga_rd_addr),
        .vga_rd_data(vga_rd_data),
        .frame_clear(frame_clear),
        .frame_ready(frame_ready)
    );

    // 2. Line Buffer: Converte fluxo contínuo de pixels em janelas 3x3
    line_buffer_32x32 lb_inst (
        .clk(clk),
        .rst(rst),
        .pixel_in(fb_rd_data),
        .shift_en(fb_rd_en_d),
        .win(win),
        .window_valid(window_valid)
    );

    // 3. Convolução: Aplica 4 filtros independentes + bias + ReLU
    conv_4_filters_relu_window conv_inst (
        .clk(clk),
        .rst(rst),
        .window_valid(window_valid),
        .win_data(win),
        .w_f0(conv_w0),
        .w_f1(conv_w1),
        .w_f2(conv_w2),
        .w_f3(conv_w3),
        .b_f0(conv_b0),
        .b_f1(conv_b1),
        .b_f2(conv_b2),
        .b_f3(conv_b3),
        .valid_out(conv_valid),
        .out_f0(conv_out_f0),
        .out_f1(conv_out_f1),
        .out_f2(conv_out_f2),
        .out_f3(conv_out_f3)
    );

    // 4. Max Pooling: Redução de dimensionalidade espacial 2x2
    max_pooling_design pool_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(conv_valid),
        .data_in_f0(conv_out_f0),
        .data_in_f1(conv_out_f1),
        .data_in_f2(conv_out_f2),
        .data_in_f3(conv_out_f3),
        .valid_out(pool_valid),
        .data_out(pool_data)
    );

    // 5. Flatten: Serialização dos mapas 2D para array 1D
    flatten flat_inst (
        .clk(clk),
        .rst(rst),
        .data_in(pool_data),
        .valid_in(pool_valid),
        .data_out(flat_data),
        .valid_out(flat_valid),
        .done()
    );

    // 6. Memória ROM Compartilhada: Pesos pré-treinados (19 classes)
    weights_shared_rom weights_inst (
        .clk(clk),
        .rst(rst),
        .boot_done(weights_boot_done),
        .dense_addr(dense_addr),
        .conv_w0(conv_w0),
        .conv_w1(conv_w1),
        .conv_w2(conv_w2),
        .conv_w3(conv_w3),
        .conv_b0(conv_b0),
        .conv_b1(conv_b1),
        .conv_b2(conv_b2),
        .conv_b3(conv_b3),
        .dense_w(dense_w),
        .dense_b(dense_b)
    );

    // 7. Camada Densa: Calcula os logits (scores brutos) para as 19 classes
    dense_900x19_scores dense_inst (
        .clk(clk),
        .rst(rst),
        .x_in(flat_data_d),
        .w_in(dense_w),
        .bias_in(dense_b),
        .valid_in(flat_valid_d),
        .scores(dense_scores),
        .valid_out(dense_valid),
        .done(dense_done)
    );

    // 8. Argmax: Identifica a predição dominante entre as 19 classes.
    //    SEM threshold — a classe 0 (Desconhecido) é nativa da rede (Softmax).
    //    unknown = 1 apenas quando a rede prediz a classe 0.
    argmax_19 argmax_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(dense_valid),
        .scores(dense_scores),
        .valid_out(argmax_valid),
        .class_id(class_id),
        .unknown(unknown),
        .max_score(final_result)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rx_sync_1       <= 1'b1;
            rx_sync_2       <= 1'b1;
            uart_wr_addr    <= 10'd0;
            uart_start_pulse <= 1'b0;
            uart_frame_pending <= 1'b0;
            state           <= ST_IDLE;
            fb_rd_en        <= 1'b0;
            fb_rd_en_d      <= 1'b0;
            fb_rd_addr      <= 10'd0;
            rd_req_count    <= 11'd0;
            rd_val_count    <= 11'd0;
            dense_addr      <= 14'd0;
            debug_or_acc    <= 8'd0;
            access_done     <= 1'b0;
            frame_clear     <= 1'b0;
            flat_valid_d    <= 1'b0;
            flat_data_d     <= 16'sd0;
        end else begin
            rx_sync_1 <= rx_pin;
            rx_sync_2 <= rx_sync_1;

            flat_valid_d <= flat_valid;
            flat_data_d  <= flat_data;

            uart_start_pulse <= 1'b0;
            access_done      <= 1'b0;
            frame_clear      <= 1'b0;
            fb_rd_en_d       <= fb_rd_en;

            // Incremento de endereço UART: avança APÓS a escrita combinacional
            if (uart_wr_en_comb) begin
                if (uart_wr_addr == 10'd1023) begin
                    uart_wr_addr       <= 10'd0;
                    uart_frame_pending <= 1'b1;
                end else begin
                    uart_wr_addr <= uart_wr_addr + 10'd1;
                end
            end

            if (uart_frame_pending && frame_ready && state == ST_IDLE) begin
                uart_start_pulse   <= 1'b1;
                uart_frame_pending <= 1'b0;
            end

            if (frame_clear) begin
                uart_wr_addr <= 10'd0;
                debug_or_acc <= 8'd0;
            end

            if (flat_valid) begin
                if (dense_addr == 14'd899) begin
                    dense_addr <= 14'd0;
                end else begin
                    dense_addr <= dense_addr + 14'd1;
                end
            end

            if (fb_rd_en_d && rd_val_count < 11'd1024) begin
                rd_val_count <= rd_val_count + 11'd1;
                debug_or_acc <= debug_or_acc | fb_rd_data;
            end

            case (state)
                ST_IDLE: begin
                    fb_rd_en     <= 1'b0;
                    fb_rd_addr   <= 10'd0;
                    rd_req_count <= 11'd0;
                    rd_val_count <= 11'd0;
                    if (start_system_int && frame_ready && weights_boot_done) begin
                        state       <= ST_READ;
                        frame_clear <= 1'b1;
                        dense_addr  <= 14'd0;
                    end
                end

                ST_READ: begin
                    if (rd_req_count < 11'd1024) begin
                        fb_rd_en   <= 1'b1;
                        fb_rd_addr <= rd_req_count[9:0];
                        rd_req_count <= rd_req_count + 11'd1;
                    end else begin
                        fb_rd_en <= 1'b0;
                    end

                    if (rd_val_count == 11'd1024) begin
                        fb_rd_en <= 1'b0;
                        state    <= ST_WAIT;
                    end
                end

                ST_WAIT: begin
                    fb_rd_en <= 1'b0;
                    if (dense_done) begin
                        state <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    access_done <= 1'b1;
                    state       <= ST_IDLE;
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule