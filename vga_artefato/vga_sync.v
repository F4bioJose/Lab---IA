module vga_sync (
    input  wire clk,
    input  wire rst,          
    output wire hsync,        // Mudado de reg para wire
    output wire vsync,        // Mudado de reg para wire
    output wire [9:0] pixel_x, // Mudado de reg para wire
    output wire [9:0] pixel_y, // Mudado de reg para wire
    output wire video_on      // Mudado de reg para wire
);

    parameter H_ACTIVE = 640, H_FRONT = 16, H_SYNC = 96, H_BACK = 48, H_TOTAL = H_ACTIVE+H_FRONT+H_SYNC+H_BACK-1; 
    parameter V_ACTIVE = 480, V_FRONT = 10, V_SYNC = 2,  V_BACK = 33, V_TOTAL = V_ACTIVE+V_FRONT+V_SYNC+V_BACK-1; 

    reg [9:0] h_count, v_count;

    // 1. Contador Horizontal
    always @(posedge clk or posedge rst) begin
        if (rst) 
            h_count <= 0;
        else if (h_count == H_TOTAL) 
            h_count <= 0;
        else 
            h_count <= h_count + 1;
    end

    // 2. Contador Vertical
    always @(posedge clk or posedge rst) begin
        if (rst) 
            v_count <= 0;
        else if (h_count == H_TOTAL) begin         
            if (v_count == V_TOTAL) 
                v_count <= 0;
            else 
                v_count <= v_count + 1;            
        end
    end

    
    assign hsync = (h_count >= (H_ACTIVE + H_FRONT) && h_count < (H_ACTIVE + H_FRONT + H_SYNC)) ? 1'b0 : 1'b1;
    
    assign vsync = (v_count >= (V_ACTIVE + V_FRONT) && v_count < (V_ACTIVE + V_FRONT + V_SYNC)) ? 1'b0 : 1'b1;
    
    assign video_on = (h_count < H_ACTIVE && v_count < V_ACTIVE);
    
    
    assign pixel_x = h_count;
    assign pixel_y = v_count;

endmodule