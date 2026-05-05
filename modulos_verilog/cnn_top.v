// ==============================================================================
// Módulo: cnn_top
// Descrição: Top-level da arquitetura Tiny-CNN. Instancia, conecta e orquestra 
//            todos os submódulos da rede: Framebuffer, Line Buffer, Convolução, 
//            Max Pooling, Flatten, Camada Densa e Decisão (Argmax/Threshold).
// ==============================================================================
module cnn_top (
    input wire clk,
    input wire rst,
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
    // A varredura de exibição de vídeo se conectará aqui. O Controlador VGA
    // ativará o `vga_rd_en` e fará o mapeamento de pixels X,Y em `vga_rd_addr`.
    // NOTA PARA ARTEFATO 3: Na versão final, estas portas e o módulo Framebuffer 
    // interno devem ser substituídos pelo Controlador da SRAM externa de 512 KB da DE2-115.
    input wire vga_rd_en,
    input wire [9:0] vga_rd_addr,
    output wire [7:0] vga_rd_data,

    output wire [15:0] final_result,
    output wire [2:0] class_id,
    output wire unknown,
    output reg access_done,
    output wire frame_ready
);

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

    wire signed [7:0] dense_w [0:6];
    wire signed [7:0] dense_b [0:6];
    reg [9:0] dense_addr;

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
    wire signed [15:0] dense_scores [0:6];
    wire argmax_valid;

    reg [10:0] rd_req_count;
    reg [10:0] rd_val_count;

    // Maquina de estados (FSM) principal para controle do pipeline
    localparam ST_IDLE  = 2'd0; // Estado inativo aguardando frame_ready
    localparam ST_READ  = 2'd1; // Varredura de memória para alimentação da rede
    localparam ST_WAIT  = 2'd2; // Aguardando as últimas camadas concluírem (densa)
    localparam ST_DONE  = 2'd3; // Sinaliza fim da inferência e resultado válido

    reg [1:0] state;

    // [MAPEAMENTO DE MEMÓRIA FUTURO (SRAM)]
    // NOTA PARA ARTEFATO 3: O Framebuffer deve ser substituído pela interface da SRAM.
    // Mapeamento sugerido para a SRAM de 512KB:
    // - Banco A: 640x480 (Escala de cinza) reservado para a saída VGA.
    // - Banco B: 32x32 (Escala de cinza) reservado para a entrada da CNN.
    // =====================================================================
    // Instanciação e Interconexão dos Componentes do Hardware CNN
    // =====================================================================

    // 1. Framebuffer: Armazena a imagem a ser processada
    framebuffer_32x32 framebuffer_inst (
        .clk(clk),
        .rst(rst),
        .wr_en(fb_wr_en),
        .wr_addr(fb_wr_addr),
        .wr_data(fb_wr_data),
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

    // 6. Memória ROM Compartilhada: Pesos pré-treinados
    weights_shared_rom weights_inst (
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

    // 7. Camada Densa: Calcula os logits (scores brutos) para as 7 classes
    dense_900x7_scores dense_inst (
        .clk(clk),
        .rst(rst),
        .x_in(flat_data),
        .w_in(dense_w),
        .bias_in(dense_b),
        .valid_in(flat_valid),
        .scores(dense_scores),
        .valid_out(dense_valid),
        .done(dense_done)
    );

    // 8. Argmax + Threshold: Identifica a predição dominante com limiar de confiança
    argmax_threshold_7 argmax_inst (
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
            state <= ST_IDLE;
            fb_rd_en <= 1'b0;
            fb_rd_en_d <= 1'b0;
            fb_rd_addr <= 10'd0;
            rd_req_count <= 11'd0;
            rd_val_count <= 11'd0;
            dense_addr <= 10'd0;
            access_done <= 1'b0;
            frame_clear <= 1'b0;
        end else begin
            access_done <= 1'b0;
            frame_clear <= 1'b0;
            fb_rd_en_d <= fb_rd_en;

            if (flat_valid) begin
                if (dense_addr == 10'd899) begin
                    dense_addr <= 10'd0;
                end else begin
                    dense_addr <= dense_addr + 10'd1;
                end
            end

            if (fb_rd_en_d && rd_val_count < 11'd1024) begin
                rd_val_count <= rd_val_count + 11'd1;
            end

            case (state)
                ST_IDLE: begin
                    fb_rd_en <= 1'b0;
                    fb_rd_addr <= 10'd0;
                    rd_req_count <= 11'd0;
                    rd_val_count <= 11'd0;
                    if (start_system && frame_ready) begin
                        state <= ST_READ;
                        frame_clear <= 1'b1;
                        dense_addr <= 10'd0;
                    end
                end

                ST_READ: begin
                    // Varre sequencialmente todo o buffer de memória da imagem (1024 endereços)
                    if (rd_req_count < 11'd1024) begin
                        fb_rd_en <= 1'b1;
                        fb_rd_addr <= rd_req_count[9:0];
                        rd_req_count <= rd_req_count + 11'd1;
                    end else begin
                        fb_rd_en <= 1'b0;
                    end

                    if (rd_val_count == 11'd1024) begin
                        fb_rd_en <= 1'b0;
                        state <= ST_WAIT;
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
                    state <= ST_IDLE;
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule