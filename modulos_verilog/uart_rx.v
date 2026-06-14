module uart_rx #(
    parameter CLK_FREQ = 50000000,  // 50 MHz
    parameter BAUD_RATE = 2000000    // Velocidade de comunicacao
)(
    input wire clk,
    input wire rst,
    input wire rx_pin,          // O pino fisico conectado ao cabo do PC

    output reg [7:0] data_out,  // O byte montado (liga no uart_pixel do Frame Buffer)
    output reg data_valid       // O aviso (liga no uart_rx_valid do Frame Buffer)
);

    // Calculo do tempo de cada bit
    localparam CYCLES_PER_BIT = CLK_FREQ / BAUD_RATE;

    // Estados da Maquina (FSM)
    localparam IDLE  = 2'b00;
    localparam START = 2'b01;
    localparam DATA  = 2'b10;
    localparam STOP  = 2'b11;

    reg [1:0] state = IDLE;
    reg [15:0] clock_count = 16'd0;
    reg [2:0] bit_index = 3'd0;   // Conta qual bit estamos lendo (0 a 7)
    reg [7:0] shift_reg = 8'd0;   // Guarda temporariamente os bits recebidos

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            clock_count <= 16'd0;
            bit_index <= 3'd0;
            data_valid <= 1'b0;
            data_out <= 8'h00;
        end else begin
            // aviso de dado valido dura apenas 1 ciclo de clock
            data_valid <= 1'b0;

            case (state)
                IDLE: begin
                    clock_count <= 16'd0;
                    bit_index <= 3'd0;
                    // A linha serial fica em 1 quando ociosa.
                    // 0 indica o Start Bit (inicio da transmissao)
                    if (rx_pin == 1'b0) begin
                        state <= START;
                    end
                end

                START: begin
                    // Espera chegar no meio do Start Bit para confirmar que nao e ruido
                    if (clock_count == (CYCLES_PER_BIT / 2)) begin
                        if (rx_pin == 1'b0) begin
                            clock_count <= 16'd0;
                            state <= DATA;
                        end else begin
                            state <= IDLE; // Foi falso alarme (ruido)
                        end
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end

                DATA: begin
                    // Espera o tempo de 1 bit inteiro para ler no meio do sinal
                    if (clock_count == CYCLES_PER_BIT - 1) begin
                        clock_count <= 16'd0;
                        shift_reg[bit_index] <= rx_pin; // Salva o bit lido

                        if (bit_index == 3'd7) begin
                            state <= STOP;
                            bit_index <= 3'd0;
                        end else begin
                            bit_index <= bit_index + 3'd1;
                        end
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end

                STOP: begin
                    // Espera terminar o Stop Bit
                    if (clock_count == CYCLES_PER_BIT - 1) begin
                        data_out <= shift_reg; // Transfere o byte completo para a saida
                        data_valid <= 1'b1;    // PRO FRAME BUFFER: PIXEL PRONTO
                        state <= IDLE;
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end
            endcase
        end
    end
endmodule
