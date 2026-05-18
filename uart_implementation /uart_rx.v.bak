module uart_rx #(
    parameter CLK_FREQ = 50000000,  // 50 MHz
    parameter BAUD_RATE = 115200    // Velocidade de comunicação
)(
    input wire clk,
    input wire rst,
    input wire rx_pin,          // O pino físico conectado ao cabo do PC

    output reg [7:0] data_out,  // O byte montado (liga no uart_pixel do Frame Buffer)
    output reg data_valid       // O aviso (liga no uart_rx_valid do Frame Buffer)
);

    // Cálculo do tempo de cada bit
    localparam CYCLES_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    // Estados da Máquina (FSM)
    localparam IDLE  = 2'b00;
    localparam START = 2'b01;
    localparam DATA  = 2'b10;
    localparam STOP  = 2'b11;

    reg [1:0] state = IDLE;
    reg [15:0] clock_count = 0;
    reg [2:0] bit_index = 0;   // Conta qual bit estamos lendo (0 a 7)
    reg [7:0] shift_reg = 0;   // Guarda temporariamente os bits recebidos

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            clock_count <= 0;
            bit_index <= 0;
            data_valid <= 1'b0;
            data_out <= 8'h00;
        end else begin
            // Por padrão, o aviso de "dado válido" dura apenas 1 ciclo de clock
            data_valid <= 1'b0; 

            case (state)
                IDLE: begin
                    clock_count <= 0;
                    bit_index <= 0;
                    // A linha serial fica em '1' quando ociosa. 
                    // Um '0' indica o Start Bit (início da transmissão)
                    if (rx_pin == 1'b0) begin 
                        state <= START;
                    end
                end

                START: begin
                    // Espera chegar no MEIO do Start Bit para confirmar que não é ruído
                    if (clock_count == (CYCLES_PER_BIT / 2)) begin
                        if (rx_pin == 1'b0) begin
                            clock_count <= 0;
                            state <= DATA;
                        end else begin
                            state <= IDLE; // Foi falso alarme (ruído)
                        end
                    end else begin
                        clock_count <= clock_count + 1;
                    end
                end

                DATA: begin
                    // Espera o tempo de 1 bit inteiro para ler no meio do sinal
                    if (clock_count == CYCLES_PER_BIT - 1) begin
                        clock_count <= 0;
                        shift_reg[bit_index] <= rx_pin; // Salva o bit lido
                        
                        if (bit_index == 7) begin
                            state <= STOP;
                            bit_index <= 0;
                        end else begin
                            bit_index <= bit_index + 1;
                        end
                    end else begin
                        clock_count <= clock_count + 1;
                    end
                end

                STOP: begin
                    // Espera terminar o Stop Bit
                    if (clock_count == CYCLES_PER_BIT - 1) begin
                        data_out <= shift_reg; // Transfere o byte completo para a saída
                        data_valid <= 1'b1;    // GRITA PRO FRAME BUFFER: "PIXEL PRONTO!"
                        state <= IDLE;
                    end else begin
                        clock_count <= clock_count + 1;
                    end
                end
            endcase
        end
    end
endmodule