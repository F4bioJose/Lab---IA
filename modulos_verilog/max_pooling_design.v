// ==============================================================================
// Módulo: max_pooling_design
// Descrição: Realiza o Max Pooling 2x2. Recebe o fluxo contínuo de 4 canais da 
//            convolução, consolida as linhas e as colunas consecutivas e 
//            repasse o valor máximo de uma vizinhança 2x2.
// ==============================================================================
module max_pooling_design (
    input wire clk,
    input wire rst,
    input wire valid_in,
    input wire signed [15:0] data_in_f0,
    input wire signed [15:0] data_in_f1,
    input wire signed [15:0] data_in_f2,
    input wire signed [15:0] data_in_f3,
    output reg valid_out,
    output reg signed [15:0] data_out
);

    reg signed [15:0] row1_f0 [0:29];
    reg signed [15:0] row1_f1 [0:29];
    reg signed [15:0] row1_f2 [0:29];
    reg signed [15:0] row1_f3 [0:29];

    reg signed [15:0] prev_row_prev_f0;
    reg signed [15:0] prev_row_prev_f1;
    reg signed [15:0] prev_row_prev_f2;
    reg signed [15:0] prev_row_prev_f3;
    reg signed [15:0] curr_prev_f0;
    reg signed [15:0] curr_prev_f1;
    reg signed [15:0] curr_prev_f2;
    reg signed [15:0] curr_prev_f3;

    reg [5:0] x;
    reg [5:0] y;

    // Filas FIFO e Buffer (Agrupa e gerencia as saídas multiplexadas dos 4 canais)
    reg [63:0] fifo [0:31];
    reg [4:0] wr_ptr;
    reg [4:0] rd_ptr;
    reg [5:0] fifo_count;
    reg [63:0] bundle_reg;

    reg [1:0] chan_idx;
    reg pending;

    wire signed [15:0] pool_f0;
    wire signed [15:0] pool_f1;
    wire signed [15:0] pool_f2;
    wire signed [15:0] pool_f3;

    wire pool_event;
    wire fifo_full;
    wire fifo_empty;
    wire fifo_push;
    wire fifo_pop;

    assign pool_f0 = pooling(prev_row_prev_f0, row1_f0[x], curr_prev_f0, data_in_f0);
    assign pool_f1 = pooling(prev_row_prev_f1, row1_f1[x], curr_prev_f1, data_in_f1);
    assign pool_f2 = pooling(prev_row_prev_f2, row1_f2[x], curr_prev_f2, data_in_f2);
    assign pool_f3 = pooling(prev_row_prev_f3, row1_f3[x], curr_prev_f3, data_in_f3);

    assign pool_event = valid_in && (x >= 6'd1) && (y >= 6'd1) && x[0] && y[0];
    assign fifo_full = (fifo_count == 6'd32);
    assign fifo_empty = (fifo_count == 6'd0);
    assign fifo_push = pool_event && !fifo_full;
    assign fifo_pop = (!pending) && !fifo_empty;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            x <= 6'd0;
            y <= 6'd0;
            prev_row_prev_f0 <= 16'sd0;
            prev_row_prev_f1 <= 16'sd0;
            prev_row_prev_f2 <= 16'sd0;
            prev_row_prev_f3 <= 16'sd0;
            curr_prev_f0 <= 16'sd0;
            curr_prev_f1 <= 16'sd0;
            curr_prev_f2 <= 16'sd0;
            curr_prev_f3 <= 16'sd0;
            wr_ptr <= 5'd0;
            rd_ptr <= 5'd0;
            fifo_count <= 6'd0;
            bundle_reg <= 64'd0;
            chan_idx <= 2'd0;
            pending <= 1'b0;
            valid_out <= 1'b0;
            data_out <= 16'sd0;
            for (integer i = 0; i < 30; i = i + 1) begin
                row1_f0[i] <= 16'sd0;
                row1_f1[i] <= 16'sd0;
                row1_f2[i] <= 16'sd0;
                row1_f3[i] <= 16'sd0;
            end
        end else begin
            valid_out <= 1'b0;

            if (valid_in) begin
                row1_f0[x] <= data_in_f0;
                row1_f1[x] <= data_in_f1;
                row1_f2[x] <= data_in_f2;
                row1_f3[x] <= data_in_f3;

                prev_row_prev_f0 <= row1_f0[x];
                prev_row_prev_f1 <= row1_f1[x];
                prev_row_prev_f2 <= row1_f2[x];
                prev_row_prev_f3 <= row1_f3[x];

                curr_prev_f0 <= data_in_f0;
                curr_prev_f1 <= data_in_f1;
                curr_prev_f2 <= data_in_f2;
                curr_prev_f3 <= data_in_f3;

                if (x == 6'd29) begin
                    x <= 6'd0;
                    if (y == 6'd29) begin
                        y <= 6'd0;
                    end else begin
                        y <= y + 6'd1;
                    end
                end else begin
                    x <= x + 6'd1;
                end
            end

            if (fifo_push) begin
                fifo[wr_ptr] <= {pool_f3, pool_f2, pool_f1, pool_f0};
                wr_ptr <= wr_ptr + 5'd1;
            end

            if (fifo_pop) begin
                bundle_reg <= fifo[rd_ptr];
                rd_ptr <= rd_ptr + 5'd1;
                pending <= 1'b1;
                chan_idx <= 2'd1;
                valid_out <= 1'b1;
                data_out <= fifo[rd_ptr][15:0];
            end else if (pending) begin
                valid_out <= 1'b1;
                case (chan_idx)
                    2'd0: data_out <= bundle_reg[15:0];
                    2'd1: data_out <= bundle_reg[31:16];
                    2'd2: data_out <= bundle_reg[47:32];
                    default: data_out <= bundle_reg[63:48];
                endcase

                if (chan_idx == 2'd3) begin
                    chan_idx <= 2'd0;
                    pending <= 1'b0;
                end else begin
                    chan_idx <= chan_idx + 2'd1;
                end
            end

            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + 6'd1;
                2'b01: fifo_count <= fifo_count - 6'd1;
                default: fifo_count <= fifo_count;
            endcase
        end
    end

    // Função utilitária combinacional para extrair o maior valor de um quadrante 2x2
    function signed [15:0] pooling;
        input signed [15:0] a, b, c, d;
        reg signed [15:0] m1, m2;
        begin
            m1 = (a > b) ? a : b;
            m2 = (c > d) ? c : d;
            pooling = (m1 > m2) ? m1 : m2;
        end
    endfunction

endmodule