`timescale 1ns/1ps

// ==============================================================================
// Módulo: tb_cnn_layer_capture
// Descrição: Testbench de simulação do cnn_top (19 classes).
//            Para cada imagem processada, captura a saída de cada camada da
//            CNN em um arquivo .txt separado, permitindo comparação ponto a
//            ponto com a implementação Python de referência.
//
// Mapeamento de Classes (ordem Keras — string sort das pastas do dataset):
//   0 = Desconhecido | 1 = Igor | 2 = Joao | 3 = Jose Henrique
//   4 = Julia | 5 = Lucio | 6 = Naira | 7 = Rafael
//   8 = Samuel | 9 = Yuri | 10 = Anna Carol | 11 = Bruno | 12 = Diego
//   13 = Eduardo | 14 = Fabio | 15 = Felipe | 16 = Gabriel | 17 = Horacio | 18 = Hugo
//
// Arquivos gerados (por padrão em "comparacao/hw/"):
//   conv_out.txt   — Saídas dos 4 filtros conv após ReLU (uma linha por janela)
//   pool_out.txt   — Saídas do Max Pooling serializadas (uma linha por amostra)
//   flat_out.txt   — Saídas do Flatten (uma linha por elemento, 900 total)
//   dense_out.txt  — Scores finais das 19 classes (uma linha com 19 valores)
//
// Formato de cada linha:
//   conv_out.txt : "f0 f1 f2 f3\n"          (4 valores Q2.14 com sinal)
//   pool_out.txt : "val\n"                   (1 valor Q2.14 por canal serializado)
//   flat_out.txt : "val\n"                   (1 valor Q2.14)
//   dense_out.txt: "s0 s1 ... s18\n"         (19 scores Q2.14 com sinal)
//
// Uso no ModelSim/Questa:
//   vsim tb_cnn_layer_capture +IMG=inputs/teste2.txt +OUTDIR=comparacao/hw/
// ==============================================================================
module tb_cnn_layer_capture;

    // =========================================================================
    // PARÂMETROS
    // =========================================================================
    localparam integer CLK_HALF_PERIOD_NS = 10;          // 50 MHz
    localparam integer BAUD_RATE          = 1000000;     // Acelerado para simulação
    localparam integer BIT_PERIOD_NS      = 1000000000 / BAUD_RATE;

    // =========================================================================
    // SINAIS
    // =========================================================================
    reg        clk;
    reg        rst;
    reg        start_system;
    reg        fb_wr_en;
    reg [9:0]  fb_wr_addr;
    reg [7:0]  fb_wr_data;

    wire [15:0] final_result;
    wire [4:0]  class_id;
    wire        unknown;
    wire        access_done;
    wire        frame_ready;
    wire        debug_weights_nonzero;
    wire        debug_frame_nonzero;
    wire [7:0]  vga_rd_data;

    // Linha UART para envio de imagem
    reg uart_rxd;

    // Memória local para a imagem de teste
    reg [7:0] img_mem [0:1023];

    // Strings para arquivos e diretório de saída
    string img_file;
    string out_dir;
    string fname_conv, fname_pool, fname_flat, fname_dense;

    // Descritores de arquivo
    integer fd_conv, fd_pool, fd_flat, fd_dense;
    integer i;

    // =========================================================================
    // DUT — Instancia cnn_top diretamente para acesso a todos os sinais
    // =========================================================================
    cnn_top dut (
        .clk                  (clk),
        .rst                  (rst),
        .rx_pin               (uart_rxd),
        .start_system         (start_system),
        .fb_wr_en             (fb_wr_en),
        .fb_wr_addr           (fb_wr_addr),
        .fb_wr_data           (fb_wr_data),
        .vga_rd_en            (1'b0),
        .vga_rd_addr          (10'd0),
        .vga_rd_data          (vga_rd_data),
        .final_result         (final_result),
        .class_id             (class_id),
        .unknown              (unknown),
        .access_done          (access_done),
        .frame_ready          (frame_ready),
        .debug_weights_nonzero(debug_weights_nonzero),
        .debug_frame_nonzero  (debug_frame_nonzero)
    );

    // Sobrescreve o baud rate do uart_rx interno para acelerar simulação
    defparam dut.uart_rx_inst.BAUD_RATE = BAUD_RATE;

    // =========================================================================
    // GERAÇÃO DE CLOCK
    // =========================================================================
    initial begin
        clk = 1'b0;
        forever #(CLK_HALF_PERIOD_NS) clk = ~clk;
    end

    // =========================================================================
    // TASK: Enviar 1 byte via UART (start + 8 data bits LSB-first + stop)
    // =========================================================================
    task send_uart_byte(input [7:0] data);
        integer bit_idx;
        begin
            uart_rxd = 1'b0;                    // Start bit
            #(BIT_PERIOD_NS);
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                uart_rxd = data[bit_idx];
                #(BIT_PERIOD_NS);
            end
            uart_rxd = 1'b1;                    // Stop bit
            #(BIT_PERIOD_NS);
        end
    endtask

    // =========================================================================
    // CAPTURA — Conv: salva f0 f1 f2 f3 a cada pulso conv_valid
    // =========================================================================
    always @(posedge clk) begin
        if (dut.conv_valid) begin
            $fwrite(fd_conv, "%0d %0d %0d %0d\n",
                $signed(dut.conv_out_f0),
                $signed(dut.conv_out_f1),
                $signed(dut.conv_out_f2),
                $signed(dut.conv_out_f3));
        end
    end

    // =========================================================================
    // CAPTURA — Pool: salva um valor por pulso pool_valid
    // =========================================================================
    always @(posedge clk) begin
        if (dut.pool_valid) begin
            $fwrite(fd_pool, "%0d\n", $signed(dut.pool_data));
        end
    end

    // =========================================================================
    // CAPTURA — Flatten: salva um valor por pulso flat_valid
    // =========================================================================
    always @(posedge clk) begin
        if (dut.flat_valid) begin
            $fwrite(fd_flat, "%0d\n", $signed(dut.flat_data));
        end
    end

    // =========================================================================
    // CAPTURA — Dense: salva os 19 scores em uma única linha ao final
    // =========================================================================
    always @(posedge clk) begin
        if (dut.dense_valid) begin
            $fwrite(fd_dense, "%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                $signed(dut.dense_scores[ 0]),
                $signed(dut.dense_scores[ 1]),
                $signed(dut.dense_scores[ 2]),
                $signed(dut.dense_scores[ 3]),
                $signed(dut.dense_scores[ 4]),
                $signed(dut.dense_scores[ 5]),
                $signed(dut.dense_scores[ 6]),
                $signed(dut.dense_scores[ 7]),
                $signed(dut.dense_scores[ 8]),
                $signed(dut.dense_scores[ 9]),
                $signed(dut.dense_scores[10]),
                $signed(dut.dense_scores[11]),
                $signed(dut.dense_scores[12]),
                $signed(dut.dense_scores[13]),
                $signed(dut.dense_scores[14]),
                $signed(dut.dense_scores[15]),
                $signed(dut.dense_scores[16]),
                $signed(dut.dense_scores[17]),
                $signed(dut.dense_scores[18]));
        end
    end

    // =========================================================================
    // ESTÍMULO PRINCIPAL
    // =========================================================================
    initial begin
        // -- Inicialização dos sinais de controle --
        rst          = 1'b0;
        start_system = 1'b0;
        fb_wr_en     = 1'b0;
        fb_wr_addr   = 10'd0;
        fb_wr_data   = 8'd0;
        uart_rxd     = 1'b1;    // UART idle = HIGH

        // -- Resolve parâmetros de linha de comando --
        img_file = "inputs/teste2.txt";
        out_dir  = "comparacao/hw/";
        if ($value$plusargs("IMG=%s",    img_file)) $display("[TB] IMG    = %s", img_file);
        if ($value$plusargs("OUTDIR=%s", out_dir))  $display("[TB] OUTDIR = %s", out_dir);

        // -- Monta nomes dos arquivos de saída --
        fname_conv  = {out_dir, "conv_out.txt"};
        fname_pool  = {out_dir, "pool_out.txt"};
        fname_flat  = {out_dir, "flat_out.txt"};
        fname_dense = {out_dir, "dense_out.txt"};

        // -- Abre arquivos de captura --
        fd_conv  = $fopen(fname_conv,  "w");
        fd_pool  = $fopen(fname_pool,  "w");
        fd_flat  = $fopen(fname_flat,  "w");
        fd_dense = $fopen(fname_dense, "w");

        if (!fd_conv || !fd_pool || !fd_flat || !fd_dense) begin
            $display("[TB] ERRO: Não foi possível criar arquivos de saída em '%s'", out_dir);
            $finish;
        end

        // -- Cabeçalhos informativos --
        $fwrite(fd_conv,  "# Layer: conv | Image: %s | Format: f0 f1 f2 f3 (Q2.14 signed)\n", img_file);
        $fwrite(fd_pool,  "# Layer: pool | Image: %s | Format: val (Q2.14 signed)\n",          img_file);
        $fwrite(fd_flat,  "# Layer: flat | Image: %s | Format: val (Q2.14 signed)\n",           img_file);
        $fwrite(fd_dense, "# Layer: dense | Image: %s | Format: s0..s18 (Q2.14 signed) | Classe 0=Desconhecido, 1..18=Pessoas\n", img_file);

        // -- Carrega imagem --
        $readmemh(img_file, img_mem);

        // =====================================================================
        // FASE 1: Reset
        // =====================================================================
        $display("[TB] ===== FASE 1: Reset =====");
        rst = 1'b1;
        #100;
        rst = 1'b0;
        #100;
        $display("[TB] Reset concluído.");

        // =====================================================================
        // FASE 2: Envio da imagem via UART (1024 bytes)
        // =====================================================================
        $display("[TB] ===== FASE 2: Enviando imagem via UART (1024 bytes) =====");
        #(BIT_PERIOD_NS * 2);

        for (i = 0; i < 1024; i = i + 1) begin
            send_uart_byte(img_mem[i]);
        end

        $display("[TB] Todos os 1024 bytes enviados via UART.");
        $display("[TB] frame_ready = %0b", dut.frame_ready);

        // =====================================================================
        // FASE 3: Aguardar inferência concluir (FSM auto-inicia via uart_start_pulse)
        // =====================================================================
        $display("[TB] ===== FASE 3: Aguardando inferência =====");
        wait (dut.access_done == 1'b1);

        // =====================================================================
        // FASE 4: Exibir resultado e fechar arquivos
        // =====================================================================
        $display("[TB] ===== INFERÊNCIA CONCLUÍDA =====");
        $display("[TB] class_id   = %0d", class_id);
        $display("[TB] unknown    = %0b (classe 0 = Desconhecido nativo)", unknown);
        $display("[TB] max_score  = %0d (Q2.14 = %f)", $signed(final_result),
                 $signed(final_result) / 16384.0);

        $display("[TB] Dense scores (Q2.14 → float):");
        $display("[TB]   s[ 0]=%-6d [Desconhecido]",    $signed(dut.dense_scores[ 0]));
        $display("[TB]   s[ 1]=%-6d  s[ 2]=%-6d  s[ 3]=%-6d",
            $signed(dut.dense_scores[ 1]),
            $signed(dut.dense_scores[ 2]),
            $signed(dut.dense_scores[ 3]));
        $display("[TB]   s[ 4]=%-6d  s[ 5]=%-6d  s[ 6]=%-6d",
            $signed(dut.dense_scores[ 4]),
            $signed(dut.dense_scores[ 5]),
            $signed(dut.dense_scores[ 6]));
        $display("[TB]   s[ 7]=%-6d  s[ 8]=%-6d  s[ 9]=%-6d",
            $signed(dut.dense_scores[ 7]),
            $signed(dut.dense_scores[ 8]),
            $signed(dut.dense_scores[ 9]));
        $display("[TB]   s[10]=%-6d  s[11]=%-6d  s[12]=%-6d",
            $signed(dut.dense_scores[10]),
            $signed(dut.dense_scores[11]),
            $signed(dut.dense_scores[12]));
        $display("[TB]   s[13]=%-6d  s[14]=%-6d  s[15]=%-6d",
            $signed(dut.dense_scores[13]),
            $signed(dut.dense_scores[14]),
            $signed(dut.dense_scores[15]));
        $display("[TB]   s[16]=%-6d  s[17]=%-6d  s[18]=%-6d",
            $signed(dut.dense_scores[16]),
            $signed(dut.dense_scores[17]),
            $signed(dut.dense_scores[18]));

        // -- Fecha arquivos de captura --
        $fclose(fd_conv);
        $fclose(fd_pool);
        $fclose(fd_flat);
        $fclose(fd_dense);

        $display("[TB] Arquivos de captura salvos em '%s'", out_dir);
        $display("[TB] ===== SIMULAÇÃO CONCLUÍDA COM SUCESSO =====");
        $stop;
    end

    // =========================================================================
    // MONITOR: Progresso de recepção UART
    // =========================================================================
    reg [9:0] uart_progress_prev = 0;
    always @(posedge clk) begin
        if (dut.uart_wr_addr != uart_progress_prev) begin
            if ((dut.uart_wr_addr % 256 == 0) && (dut.uart_wr_addr > 0)) begin
                $display("[TB] UART: %0d/1024 bytes recebidos", dut.uart_wr_addr);
            end
            uart_progress_prev = dut.uart_wr_addr;
        end
    end

    // =========================================================================
    // MONITOR: Transições de estado da FSM
    // =========================================================================
    reg [1:0] prev_state = 0;
    always @(posedge clk) begin
        if (dut.state != prev_state) begin
            case (dut.state)
                2'd0: $display("[TB] FSM -> ST_IDLE  (t=%0t)", $time);
                2'd1: $display("[TB] FSM -> ST_READ  (t=%0t)", $time);
                2'd2: $display("[TB] FSM -> ST_WAIT  (t=%0t)", $time);
                2'd3: $display("[TB] FSM -> ST_DONE  (t=%0t)", $time);
            endcase
            prev_state = dut.state;
        end
    end

endmodule
