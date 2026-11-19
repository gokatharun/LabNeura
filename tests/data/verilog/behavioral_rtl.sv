`define ID(value) value

module behavioral_rtl(
    input logic clk,
    input logic rst_n,
    input logic [7:0] data,
    output logic [7:0] q,
    output logic flag
);
    always_comb begin : comb
        q[3:0] = data[7:4];
        q[7:4] = data[3:0];
        {q[1:0], flag} = {data[0], flag};
        if (flag) begin : selected
            q = `ID(data);
        end else if (data == 8'h00) begin
            q = '0;
        end else begin
            q = data + 1;
        end
        case (flag)
            1'b0, 1'b1: q = data;
            default: q = '0;
        endcase
        casez (data)
            8'b1???????: q = data;
            default: q = '0;
        endcase
        casex (data)
            8'bx???????: q = data;
            default: q = '0;
        endcase
    end

    always_ff @(posedge clk, negedge rst_n) begin : sequential
        if (!rst_n)
            q <= '0;
        else if (flag)
            q <= data;
    end

    always_latch begin
        if (flag)
            q <= data;
    end

    always @(posedge clk or negedge rst_n)
        if (!rst_n)
            flag <= 1'b0;
        else
            flag <= data[0];

    initial begin
        #5 q = '0;
        @(posedge clk) q = data;
        @* flag = 1'b1;
        @(*) flag = 1'b0;
        wait (flag) q = data;
    end
endmodule
