// ==============================================================================
// Stubs de simulação para IPs Altera (Icarus Verilog)
// Substitui altsyncram e altddio_out por modelos comportamentais simples.
// ==============================================================================

// --- altsyncram: RAM dual-port comportamental ---
module altsyncram (
    clock0, clock1,
    address_a, address_b,
    data_a, data_b,
    wren_a, wren_b,
    q_a, q_b,
    aclr0, aclr1,
    addressstall_a, addressstall_b,
    byteena_a,
    clocken0, clocken1, clocken2, clocken3,
    eccstatus,
    rden_a, rden_b
);
    parameter operation_mode = "DUAL_PORT";
    parameter width_a = 8;
    parameter widthad_a = 10;
    parameter numwords_a = 1024;
    parameter width_b = 8;
    parameter widthad_b = 10;
    parameter numwords_b = 1024;
    parameter address_reg_b = "CLOCK1";
    parameter outdata_reg_b = "CLOCK1";
    parameter clock_enable_input_a = "BYPASS";
    parameter clock_enable_input_b = "BYPASS";
    parameter clock_enable_output_b = "BYPASS";
    parameter intended_device_family = "Cyclone IV E";
    parameter lpm_type = "altsyncram";
    parameter power_up_uninitialized = "FALSE";
    parameter ram_block_type = "AUTO";
    parameter read_during_write_mode_mixed_ports = "DONT_CARE";
    parameter width_byteena_a = 1;

    input  clock0, clock1;
    input  [widthad_a-1:0] address_a, address_b;
    input  [width_a-1:0] data_a, data_b;
    input  wren_a, wren_b;
    output reg [width_a-1:0] q_a;
    output reg [width_b-1:0] q_b;
    input  aclr0, aclr1;
    input  addressstall_a, addressstall_b;
    input  byteena_a;
    input  clocken0, clocken1, clocken2, clocken3;
    output [1:0] eccstatus;
    input  rden_a, rden_b;

    assign eccstatus = 2'b00;

    // Memória interna
    reg [width_a-1:0] mem [0:numwords_a-1];

    integer k;
    initial begin
        for (k = 0; k < numwords_a; k = k + 1)
            mem[k] = {width_a{1'b0}};
    end

    // Porta A — Escrita
    always @(posedge clock0) begin
        if (wren_a)
            mem[address_a] <= data_a;
    end

    // Porta B — Leitura registrada
    always @(posedge clock1) begin
        q_b <= mem[address_b];
    end
endmodule

// --- altddio_out: DDR output comportamental (reproduz clock) ---
module altddio_out (
    datain_h, datain_l,
    outclock,
    dataout,
    aclr, aset,
    oe, outclocken,
    sclr, sset
);
    parameter extend_oe_disable = "OFF";
    parameter intended_device_family = "Cyclone IV E";
    parameter invert_output = "OFF";
    parameter lpm_hint = "UNUSED";
    parameter lpm_type = "altddio_out";
    parameter oe_reg = "UNREGISTERED";
    parameter power_up_high = "OFF";
    parameter width = 1;

    input  [width-1:0] datain_h, datain_l;
    input  outclock;
    output [width-1:0] dataout;
    input  aclr, aset;
    input  oe, outclocken;
    input  sclr, sset;

    // Simulação simplificada: saída = clock (datain_h na borda alta, datain_l na baixa)
    assign dataout = outclock ? datain_h : datain_l;
endmodule
