/// =================== Unsigned, Fixed Point =========================
module std_fp_add #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out
);
  assign out = left + right;
endmodule

module std_fp_sub #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out
);
  assign out = left - right;
endmodule

module std_fp_mult_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    input  logic             go,
    input  logic             clk,
    output logic [WIDTH-1:0] out,
    output logic             done
);
  logic [WIDTH-1:0]          rtmp;
  logic [WIDTH-1:0]          ltmp;
  logic [(WIDTH << 1) - 1:0] out_tmp;
  reg done_buf[1:0];
  always_ff @(posedge clk) begin
    if (go) begin
      rtmp <= right;
      ltmp <= left;
      out_tmp <= ltmp * rtmp;
      out <= out_tmp[(WIDTH << 1) - INT_WIDTH - 1 : WIDTH - INT_WIDTH];

      done <= done_buf[1];
      done_buf[0] <= 1'b1;
      done_buf[1] <= done_buf[0];
    end else begin
      rtmp <= 0;
      ltmp <= 0;
      out_tmp <= 0;
      out <= 0;

      done <= 0;
      done_buf[0] <= 0;
      done_buf[1] <= 0;
    end
  end
endmodule

/* verilator lint_off WIDTH */
module std_fp_div_pipe #(
  parameter WIDTH = 32,
  parameter INT_WIDTH = 16,
  parameter FRAC_WIDTH = 16
) (
    input  logic             go,
    input  logic             clk,
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out_remainder,
    output logic [WIDTH-1:0] out_quotient,
    output logic             done
);
    localparam ITERATIONS = WIDTH + FRAC_WIDTH;

    logic [WIDTH-1:0] quotient, quotient_next;
    logic [WIDTH:0] acc, acc_next;
    logic [$clog2(ITERATIONS)-1:0] idx;
    logic start, running, finished;

    assign start = go && !running;
    assign finished = running && (idx == ITERATIONS - 1);

    always_comb begin
      if (acc >= {1'b0, right}) begin
        acc_next = acc - right;
        {acc_next, quotient_next} = {acc_next[WIDTH-1:0], quotient, 1'b1};
      end else begin
        {acc_next, quotient_next} = {acc, quotient} << 1;
      end
    end

    always_ff @(posedge clk) begin
      if (!go) begin
        running <= 0;
        done <= 0;
        out_remainder <= 0;
        out_quotient <= 0;
      end else if (start && left == 0) begin
        out_remainder <= 0;
        out_quotient <= 0;
        done <= 1;
      end

      if (start) begin
        running <= 1;
        done <= 0;
        idx <= 0;
        {acc, quotient} <= {{WIDTH{1'b0}}, left, 1'b0};
        out_quotient <= 0;
        out_remainder <= left;
      end else if (finished) begin
        running <= 0;
        done <= 1;
        out_quotient <= quotient_next;
      end else begin
        idx <= idx + 1;
        acc <= acc_next;
        quotient <= quotient_next;
        if (right <= out_remainder) begin
          out_remainder <= out_remainder - right;
        end
      end
    end
endmodule

module std_fp_gt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic             out
);
  assign out = left > right;
endmodule

module std_fp_add_dwidth #(
    parameter WIDTH1 = 32,
    parameter WIDTH2 = 32,
    parameter INT_WIDTH1 = 16,
    parameter FRAC_WIDTH1 = 16,
    parameter INT_WIDTH2 = 12,
    parameter FRAC_WIDTH2 = 20,
    parameter OUT_WIDTH = 36
) (
    input  logic [   WIDTH1-1:0] left,
    input  logic [   WIDTH2-1:0] right,
    output logic [OUT_WIDTH-1:0] out
);

  localparam BIG_INT = (INT_WIDTH1 >= INT_WIDTH2) ? INT_WIDTH1 : INT_WIDTH2;
  localparam BIG_FRACT = (FRAC_WIDTH1 >= FRAC_WIDTH2) ? FRAC_WIDTH1 : FRAC_WIDTH2;

  if (BIG_INT + BIG_FRACT != OUT_WIDTH)
    $error("std_fp_add_dwidth: Given output width not equal to computed output width");

  logic [INT_WIDTH1-1:0] left_int;
  logic [INT_WIDTH2-1:0] right_int;
  logic [FRAC_WIDTH1-1:0] left_fract;
  logic [FRAC_WIDTH2-1:0] right_fract;

  logic [BIG_INT-1:0] mod_right_int;
  logic [BIG_FRACT-1:0] mod_left_fract;

  logic [BIG_INT-1:0] whole_int;
  logic [BIG_FRACT-1:0] whole_fract;

  assign {left_int, left_fract} = left;
  assign {right_int, right_fract} = right;

  assign mod_left_fract = left_fract * (2 ** (FRAC_WIDTH2 - FRAC_WIDTH1));

  always_comb begin
    if ((mod_left_fract + right_fract) >= 2 ** FRAC_WIDTH2) begin
      whole_int = left_int + right_int + 1;
      whole_fract = mod_left_fract + right_fract - 2 ** FRAC_WIDTH2;
    end else begin
      whole_int = left_int + right_int;
      whole_fract = mod_left_fract + right_fract;
    end
  end

  assign out = {whole_int, whole_fract};
endmodule

/// =================== Signed, Fixed Point =========================
module std_fp_sadd #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left + right);
endmodule

module std_fp_ssub #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);

  assign out = $signed(left - right);
endmodule

module std_fp_smult_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  signed       [WIDTH-1:0] left,
    input  signed       [WIDTH-1:0] right,
    input  logic                    go,
    input  logic                    clk,
    output logic signed [WIDTH-1:0] out,
    output logic                    done
);
  logic signed [WIDTH-1:0] ltmp;
  logic signed [WIDTH-1:0] rtmp;
  logic signed [(WIDTH << 1) - 1:0] out_tmp;
  reg done_buf[1:0];
  always_ff @(posedge clk) begin
    if (go) begin
      ltmp <= left;
      rtmp <= right;
      // Sign extend by the first bit for the operands.
      out_tmp <= $signed(
                   { {WIDTH{ltmp[WIDTH-1]}}, ltmp} *
                   { {WIDTH{rtmp[WIDTH-1]}}, rtmp}
                 );
      out <= out_tmp[(WIDTH << 1) - INT_WIDTH - 1: WIDTH - INT_WIDTH];

      done <= done_buf[1];
      done_buf[0] <= 1'b1;
      done_buf[1] <= done_buf[0];
    end else begin
      rtmp <= 0;
      ltmp <= 0;
      out_tmp <= 0;
      out <= 0;

      done <= 0;
      done_buf[0] <= 0;
      done_buf[1] <= 0;
    end
  end
endmodule

module std_fp_sdiv_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input                     clk,
    input                     go,
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out_quotient,
    output signed [WIDTH-1:0] out_remainder,
    output logic              done
);

  logic signed [WIDTH-1:0] left_abs;
  logic signed [WIDTH-1:0] right_abs;
  logic signed [WIDTH-1:0] comp_out_q;
  logic signed [WIDTH-1:0] comp_out_r;

  assign right_abs = right[WIDTH-1] ? -right : right;
  assign left_abs = left[WIDTH-1] ? -left : left;
  assign out_quotient = left[WIDTH-1] ^ right[WIDTH-1] ? -comp_out_q : comp_out_q;
  assign out_remainder = (left[WIDTH-1] && comp_out_r) ? $signed(right - comp_out_r) : comp_out_r;

  std_fp_div_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(INT_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
  ) comp (
    .clk(clk),
    .done(done),
    .go(go),
    .left(left_abs),
    .right(right_abs),
    .out_quotient(comp_out_q),
    .out_remainder(comp_out_r)
  );
endmodule

module std_fp_sadd_dwidth #(
    parameter WIDTH1 = 32,
    parameter WIDTH2 = 32,
    parameter INT_WIDTH1 = 16,
    parameter FRAC_WIDTH1 = 16,
    parameter INT_WIDTH2 = 12,
    parameter FRAC_WIDTH2 = 20,
    parameter OUT_WIDTH = 36
) (
    input  logic [   WIDTH1-1:0] left,
    input  logic [   WIDTH2-1:0] right,
    output logic [OUT_WIDTH-1:0] out
);

  logic signed [INT_WIDTH1-1:0] left_int;
  logic signed [INT_WIDTH2-1:0] right_int;
  logic [FRAC_WIDTH1-1:0] left_fract;
  logic [FRAC_WIDTH2-1:0] right_fract;

  localparam BIG_INT = (INT_WIDTH1 >= INT_WIDTH2) ? INT_WIDTH1 : INT_WIDTH2;
  localparam BIG_FRACT = (FRAC_WIDTH1 >= FRAC_WIDTH2) ? FRAC_WIDTH1 : FRAC_WIDTH2;

  logic [BIG_INT-1:0] mod_right_int;
  logic [BIG_FRACT-1:0] mod_left_fract;

  logic [BIG_INT-1:0] whole_int;
  logic [BIG_FRACT-1:0] whole_fract;

  assign {left_int, left_fract} = left;
  assign {right_int, right_fract} = right;

  assign mod_left_fract = left_fract * (2 ** (FRAC_WIDTH2 - FRAC_WIDTH1));

  always_comb begin
    if ((mod_left_fract + right_fract) >= 2 ** FRAC_WIDTH2) begin
      whole_int = $signed(left_int + right_int + 1);
      whole_fract = mod_left_fract + right_fract - 2 ** FRAC_WIDTH2;
    end else begin
      whole_int = $signed(left_int + right_int);
      whole_fract = mod_left_fract + right_fract;
    end
  end

  assign out = {whole_int, whole_fract};
endmodule

module std_fp_sgt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic signed [WIDTH-1:0] left,
    input  logic signed [WIDTH-1:0] right,
    output logic signed             out
);
  assign out = $signed(left > right);
endmodule

module std_fp_slt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
   input logic signed [WIDTH-1:0] left,
   input logic signed [WIDTH-1:0] right,
   output logic signed            out
);
  assign out = $signed(left < right);
endmodule

/// =================== Unsigned, Bitnum =========================
module std_mult_pipe #(
    parameter WIDTH = 32
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    input  logic             go,
    input  logic             clk,
    output logic [WIDTH-1:0] out,
    output logic             done
);
  std_fp_mult_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(WIDTH),
    .FRAC_WIDTH(0)
  ) comp (
    .clk(clk),
    .done(done),
    .go(go),
    .left(left),
    .right(right),
    .out(out)
  );
endmodule

module std_div_pipe #(
    parameter WIDTH = 32
) (
    input                    clk,
    input                    go,
    input        [WIDTH-1:0] left,
    input        [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out_remainder,
    output logic [WIDTH-1:0] out_quotient,
    output logic             done
);

  logic [WIDTH-1:0] dividend;
  logic [(WIDTH-1)*2:0] divisor;
  logic [WIDTH-1:0] quotient;
  logic [WIDTH-1:0] quotient_msk;
  logic start, running, finished;

  assign start = go && !running;
  assign finished = !quotient_msk && running;

  always_ff @(posedge clk) begin
    if (!go) begin
      running <= 0;
      done <= 0;
      out_remainder <= 0;
      out_quotient <= 0;
    end else if (start && left == 0) begin
      out_remainder <= 0;
      out_quotient <= 0;
      done <= 1;
    end

    if (start) begin
      running <= 1;
      dividend <= left;
      divisor <= right << WIDTH - 1;
      quotient <= 0;
      quotient_msk <= 1 << WIDTH - 1;
    end else if (finished) begin
      running <= 0;
      done <= 1;
      out_remainder <= dividend;
      out_quotient <= quotient;
    end else begin
      if (divisor <= dividend) begin
        dividend <= dividend - divisor;
        quotient <= quotient | quotient_msk;
      end
      divisor <= divisor >> 1;
      quotient_msk <= quotient_msk >> 1;
    end
  end

  `ifdef VERILATOR
    // Simulation self test against unsynthesizable implementation.
    always @(posedge clk) begin
      if (finished && dividend != $unsigned(left % right))
        $error(
          "\nstd_div_pipe (Remainder): Computed and golden outputs do not match!\n",
          "left: %0d", $unsigned(left),
          "  right: %0d\n", $unsigned(right),
          "expected: %0d", $unsigned(left % right),
          "  computed: %0d", $unsigned(dividend)
        );
      if (finished && quotient != $unsigned(left / right))
        $error(
          "\nstd_div_pipe (Quotient): Computed and golden outputs do not match!\n",
          "left: %0d", $unsigned(left),
          "  right: %0d\n", $unsigned(right),
          "expected: %0d", $unsigned(left / right),
          "  computed: %0d", $unsigned(quotient)
        );
    end
  `endif
endmodule

/// =================== Signed, Bitnum =========================
module std_sadd #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left + right);
endmodule

module std_ssub #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left - right);
endmodule

module std_smult_pipe #(
    parameter WIDTH = 32
) (
    input  logic                    go,
    input  logic                    clk,
    input  signed       [WIDTH-1:0] left,
    input  signed       [WIDTH-1:0] right,
    output logic signed [WIDTH-1:0] out,
    output logic                    done
);
  std_fp_smult_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(WIDTH),
    .FRAC_WIDTH(0)
  ) comp (
    .clk(clk),
    .done(done),
    .go(go),
    .left(left),
    .right(right),
    .out(out)
  );
endmodule

/* verilator lint_off WIDTH */
module std_sdiv_pipe #(
    parameter WIDTH = 32
) (
    input                     clk,
    input                     go,
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out_quotient,
    output signed [WIDTH-1:0] out_remainder,
    output logic              done
);

  logic signed [WIDTH-1:0] left_abs, right_abs, comp_out_q, comp_out_r;
  logic different_signs;

  assign right_abs = right[WIDTH-1] ? -right : right;
  assign left_abs = left[WIDTH-1] ? -left : left;
  assign different_signs = left[WIDTH-1] ^ right[WIDTH-1];
  assign out_quotient = different_signs ? -comp_out_q : comp_out_q;
  assign out_remainder = (left[WIDTH-1] && comp_out_r) ? $signed(right - comp_out_r) : comp_out_r;

  std_div_pipe #(
    .WIDTH(WIDTH)
  ) comp (
    .clk(clk),
    .done(done),
    .go(go),
    .left(left_abs),
    .right(right_abs),
    .out_quotient(comp_out_q),
    .out_remainder(comp_out_r)
  );

  `ifdef VERILATOR
    // Simulation self test against unsynthesizable implementation.
    always @(posedge clk) begin
      if (done && out_quotient != $signed(left / right))
        $error(
          "\nstd_sdiv_pipe (Quotient): Computed and golden outputs do not match!\n",
          "left: %0d", left,
          "  right: %0d\n", right,
          "expected: %0d", $signed(left / right),
          "  computed: %0d", $signed(out_quotient)
        );
      if (done && out_remainder != $signed(((left % right) + right) % right))
        $error(
          "\nstd_sdiv_pipe (Remainder): Computed and golden outputs do not match!\n",
          "left: %0d", left,
          "  right: %0d\n", right,
          "expected: %0d", $signed(((left % right) + right) % right),
          "  computed: %0d", $signed(out_remainder)
        );
    end
  `endif
endmodule

module std_sgt #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left > right);
endmodule

module std_slt #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left < right);
endmodule

module std_seq #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left == right);
endmodule

module std_sneq #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left != right);
endmodule

module std_sge #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left >= right);
endmodule

module std_sle #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left <= right);
endmodule

module std_slsh #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = left <<< right;
endmodule

module std_srsh #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = left >>> right;
endmodule
/**
 * Core primitives for Calyx.
 * Implements core primitives used by the compiler.
 *
 * Conventions:
 * - All parameter names must be SNAKE_CASE and all caps.
 * - Port names must be snake_case, no caps.
 */
`default_nettype none

module std_const #(
    parameter WIDTH = 32,
    parameter VALUE = 0
) (
   output logic [WIDTH - 1:0] out
);
  assign out = VALUE;
endmodule

module std_slice #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = 32
) (
   input wire                   logic [ IN_WIDTH-1:0] in,
   output logic [OUT_WIDTH-1:0] out
);
  assign out = in[OUT_WIDTH-1:0];

  `ifdef VERILATOR
    always_comb begin
      if (IN_WIDTH < OUT_WIDTH)
        $error(
          "std_slice: Input width less than output width\n",
          "IN_WIDTH: %0d", IN_WIDTH,
          "OUT_WIDTH: %0d", OUT_WIDTH
        );
    end
  `endif
endmodule

module std_pad #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = 32
) (
   input wire logic [IN_WIDTH-1:0]  in,
   output logic     [OUT_WIDTH-1:0] out
);
  localparam EXTEND = OUT_WIDTH - IN_WIDTH;
  assign out = { {EXTEND {1'b0}}, in};

  `ifdef VERILATOR
    always_comb begin
      if (IN_WIDTH > OUT_WIDTH)
        $error(
          "std_pad: Output width less than input width\n",
          "IN_WIDTH: %0d", IN_WIDTH,
          "OUT_WIDTH: %0d", OUT_WIDTH
        );
    end
  `endif
endmodule

module std_not #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] in,
   output logic [WIDTH-1:0] out
);
  assign out = ~in;
endmodule

module std_and #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left & right;
endmodule

module std_or #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left | right;
endmodule

module std_xor #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left ^ right;
endmodule

module std_add #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left + right;
endmodule

module std_sub #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left - right;
endmodule

module std_gt #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left > right;
endmodule

module std_lt #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left < right;
endmodule

module std_eq #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left == right;
endmodule

module std_neq #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left != right;
endmodule

module std_ge #(
    parameter WIDTH = 32
) (
    input wire   logic [WIDTH-1:0] left,
    input wire   logic [WIDTH-1:0] right,
    output logic out
);
  assign out = left >= right;
endmodule

module std_le #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left <= right;
endmodule

module std_lsh #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left << right;
endmodule

module std_rsh #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left >> right;
endmodule

/// this primitive is intended to be used
/// for lowering purposes (not in source programs)
module std_mux #(
    parameter WIDTH = 32
) (
   input wire               logic cond,
   input wire               logic [WIDTH-1:0] tru,
   input wire               logic [WIDTH-1:0] fal,
   output logic [WIDTH-1:0] out
);
  assign out = cond ? tru : fal;
endmodule

/// Memories
module std_reg #(
    parameter WIDTH = 32
) (
   input wire [ WIDTH-1:0]    in,
   input wire                 write_en,
   input wire                 clk,
   input wire                 reset,
    // output
   output logic [WIDTH - 1:0] out,
   output logic               done
);

  always_ff @(posedge clk) begin
    if (reset) begin
       out <= 0;
       done <= 0;
    end else if (write_en) begin
      out <= in;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module std_mem_d1 #(
    parameter WIDTH = 32,
    parameter SIZE = 16,
    parameter IDX_SIZE = 4
) (
   input wire                logic [IDX_SIZE-1:0] addr0,
   input wire                logic [ WIDTH-1:0] write_data,
   input wire                logic write_en,
   input wire                logic clk,
   output logic [ WIDTH-1:0] read_data,
   output logic              done
);

  logic [WIDTH-1:0] mem[SIZE-1:0];

  /* verilator lint_off WIDTH */
  assign read_data = mem[addr0];
  always_ff @(posedge clk) begin
    if (write_en) begin
      mem[addr0] <= write_data;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module std_mem_d2 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4
) (
   input wire                logic [D0_IDX_SIZE-1:0] addr0,
   input wire                logic [D1_IDX_SIZE-1:0] addr1,
   input wire                logic [ WIDTH-1:0] write_data,
   input wire                logic write_en,
   input wire                logic clk,
   output logic [ WIDTH-1:0] read_data,
   output logic              done
);

  /* verilator lint_off WIDTH */
  logic [WIDTH-1:0] mem[D0_SIZE-1:0][D1_SIZE-1:0];

  assign read_data = mem[addr0][addr1];
  always_ff @(posedge clk) begin
    if (write_en) begin
      mem[addr0][addr1] <= write_data;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module std_mem_d3 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D2_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4,
    parameter D2_IDX_SIZE = 4
) (
   input wire                logic [D0_IDX_SIZE-1:0] addr0,
   input wire                logic [D1_IDX_SIZE-1:0] addr1,
   input wire                logic [D2_IDX_SIZE-1:0] addr2,
   input wire                logic [ WIDTH-1:0] write_data,
   input wire                logic write_en,
   input wire                logic clk,
   output logic [ WIDTH-1:0] read_data,
   output logic              done
);

  /* verilator lint_off WIDTH */
  logic [WIDTH-1:0] mem[D0_SIZE-1:0][D1_SIZE-1:0][D2_SIZE-1:0];

  assign read_data = mem[addr0][addr1][addr2];
  always_ff @(posedge clk) begin
    if (write_en) begin
      mem[addr0][addr1][addr2] <= write_data;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module std_mem_d4 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D2_SIZE = 16,
    parameter D3_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4,
    parameter D2_IDX_SIZE = 4,
    parameter D3_IDX_SIZE = 4
) (
   input wire                logic [D0_IDX_SIZE-1:0] addr0,
   input wire                logic [D1_IDX_SIZE-1:0] addr1,
   input wire                logic [D2_IDX_SIZE-1:0] addr2,
   input wire                logic [D3_IDX_SIZE-1:0] addr3,
   input wire                logic [ WIDTH-1:0] write_data,
   input wire                logic write_en,
   input wire                logic clk,
   output logic [ WIDTH-1:0] read_data,
   output logic              done
);

  /* verilator lint_off WIDTH */
  logic [WIDTH-1:0] mem[D0_SIZE-1:0][D1_SIZE-1:0][D2_SIZE-1:0][D3_SIZE-1:0];

  assign read_data = mem[addr0][addr1][addr2][addr3];
  always_ff @(posedge clk) begin
    if (write_en) begin
      mem[addr0][addr1][addr2][addr3] <= write_data;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

`default_nettype wire
module main (
    input logic go,
    input logic clk,
    input logic reset,
    output logic done
);
    import "DPI-C" function string futil_getenv (input string env_var);
    string DATA;
    initial begin
        DATA = futil_getenv("DATA");
        $fdisplay(2, "DATA (path to meminit files): %s", DATA);
        $readmemh({DATA, "/M_W_x.dat"}, M_W_x.mem);
        $readmemh({DATA, "/y.dat"}, y.mem);
        $readmemh({DATA, "/x.dat"}, x.mem);
        $readmemh({DATA, "/W.dat"}, W.mem);
        $readmemh({DATA, "/b.dat"}, b.mem);
    end
    final begin
        $writememh({DATA, "/M_W_x.out"}, M_W_x.mem);
        $writememh({DATA, "/y.out"}, y.mem);
        $writememh({DATA, "/x.out"}, x.mem);
        $writememh({DATA, "/W.out"}, W.mem);
        $writememh({DATA, "/b.out"}, b.mem);
    end
    logic [31:0] M_W_x_addr0;
    logic [31:0] M_W_x_write_data;
    logic M_W_x_write_en;
    logic M_W_x_clk;
    logic [31:0] M_W_x_read_data;
    logic M_W_x_done;
    logic [31:0] y_addr0;
    logic [31:0] y_addr1;
    logic [31:0] y_write_data;
    logic y_write_en;
    logic y_clk;
    logic [31:0] y_read_data;
    logic y_done;
    logic [31:0] x_addr0;
    logic [31:0] x_addr1;
    logic [31:0] x_write_data;
    logic x_write_en;
    logic x_clk;
    logic [31:0] x_read_data;
    logic x_done;
    logic [31:0] W_addr0;
    logic [31:0] W_addr1;
    logic [31:0] W_write_data;
    logic W_write_en;
    logic W_clk;
    logic [31:0] W_read_data;
    logic W_done;
    logic [31:0] b_addr0;
    logic [31:0] b_addr1;
    logic [31:0] b_write_data;
    logic b_write_en;
    logic b_clk;
    logic [31:0] b_read_data;
    logic b_done;
    logic [31:0] MAC_accumulator_left;
    logic [31:0] MAC_accumulator_right;
    logic [31:0] MAC_accumulator_out;
    logic MAC_multiplier_clk;
    logic MAC_multiplier_go;
    logic [31:0] MAC_multiplier_left;
    logic [31:0] MAC_multiplier_right;
    logic [31:0] MAC_multiplier_out;
    logic MAC_multiplier_done;
    logic [31:0] W_idx0_in;
    logic W_idx0_write_en;
    logic W_idx0_clk;
    logic W_idx0_reset;
    logic [31:0] W_idx0_out;
    logic W_idx0_done;
    logic [31:0] M_W_x_idx_in;
    logic M_W_x_idx_write_en;
    logic M_W_x_idx_clk;
    logic M_W_x_idx_reset;
    logic [31:0] M_W_x_idx_out;
    logic M_W_x_idx_done;
    logic [31:0] x_idx1_in;
    logic x_idx1_write_en;
    logic x_idx1_clk;
    logic x_idx1_reset;
    logic [31:0] x_idx1_out;
    logic x_idx1_done;
    logic [31:0] W_idx1_lt_left;
    logic [31:0] W_idx1_lt_right;
    logic W_idx1_lt_out;
    logic [1:0] fsm_in;
    logic fsm_write_en;
    logic fsm_clk;
    logic fsm_reset;
    logic [1:0] fsm_out;
    logic fsm_done;
    logic [1:0] incr_left;
    logic [1:0] incr_right;
    logic [1:0] incr_out;
    logic [1:0] fsm0_in;
    logic fsm0_write_en;
    logic fsm0_clk;
    logic fsm0_reset;
    logic [1:0] fsm0_out;
    logic fsm0_done;
    logic cond_stored_in;
    logic cond_stored_write_en;
    logic cond_stored_clk;
    logic cond_stored_reset;
    logic cond_stored_out;
    logic cond_stored_done;
    logic [1:0] incr0_left;
    logic [1:0] incr0_right;
    logic [1:0] incr0_out;
    logic cs_wh_in;
    logic cs_wh_write_en;
    logic cs_wh_clk;
    logic cs_wh_reset;
    logic cs_wh_out;
    logic cs_wh_done;
    logic cs_wh0_in;
    logic cs_wh0_write_en;
    logic cs_wh0_clk;
    logic cs_wh0_reset;
    logic cs_wh0_out;
    logic cs_wh0_done;
    logic cs_wh1_in;
    logic cs_wh1_write_en;
    logic cs_wh1_clk;
    logic cs_wh1_reset;
    logic cs_wh1_out;
    logic cs_wh1_done;
    logic [4:0] fsm1_in;
    logic fsm1_write_en;
    logic fsm1_clk;
    logic fsm1_reset;
    logic [4:0] fsm1_out;
    logic fsm1_done;
    initial begin
        M_W_x_addr0 = 32'd0;
        M_W_x_write_data = 32'd0;
        M_W_x_write_en = 1'd0;
        M_W_x_clk = 1'd0;
        y_addr0 = 32'd0;
        y_addr1 = 32'd0;
        y_write_data = 32'd0;
        y_write_en = 1'd0;
        y_clk = 1'd0;
        x_addr0 = 32'd0;
        x_addr1 = 32'd0;
        x_write_data = 32'd0;
        x_write_en = 1'd0;
        x_clk = 1'd0;
        W_addr0 = 32'd0;
        W_addr1 = 32'd0;
        W_write_data = 32'd0;
        W_write_en = 1'd0;
        W_clk = 1'd0;
        b_addr0 = 32'd0;
        b_addr1 = 32'd0;
        b_write_data = 32'd0;
        b_write_en = 1'd0;
        b_clk = 1'd0;
        MAC_accumulator_left = 32'd0;
        MAC_accumulator_right = 32'd0;
        MAC_multiplier_clk = 1'd0;
        MAC_multiplier_go = 1'd0;
        MAC_multiplier_left = 32'd0;
        MAC_multiplier_right = 32'd0;
        W_idx0_in = 32'd0;
        W_idx0_write_en = 1'd0;
        W_idx0_clk = 1'd0;
        W_idx0_reset = 1'd0;
        M_W_x_idx_in = 32'd0;
        M_W_x_idx_write_en = 1'd0;
        M_W_x_idx_clk = 1'd0;
        M_W_x_idx_reset = 1'd0;
        x_idx1_in = 32'd0;
        x_idx1_write_en = 1'd0;
        x_idx1_clk = 1'd0;
        x_idx1_reset = 1'd0;
        W_idx1_lt_left = 32'd0;
        W_idx1_lt_right = 32'd0;
        fsm_in = 2'd0;
        fsm_write_en = 1'd0;
        fsm_clk = 1'd0;
        fsm_reset = 1'd0;
        incr_left = 2'd0;
        incr_right = 2'd0;
        fsm0_in = 2'd0;
        fsm0_write_en = 1'd0;
        fsm0_clk = 1'd0;
        fsm0_reset = 1'd0;
        cond_stored_in = 1'd0;
        cond_stored_write_en = 1'd0;
        cond_stored_clk = 1'd0;
        cond_stored_reset = 1'd0;
        incr0_left = 2'd0;
        incr0_right = 2'd0;
        cs_wh_in = 1'd0;
        cs_wh_write_en = 1'd0;
        cs_wh_clk = 1'd0;
        cs_wh_reset = 1'd0;
        cs_wh0_in = 1'd0;
        cs_wh0_write_en = 1'd0;
        cs_wh0_clk = 1'd0;
        cs_wh0_reset = 1'd0;
        cs_wh1_in = 1'd0;
        cs_wh1_write_en = 1'd0;
        cs_wh1_clk = 1'd0;
        cs_wh1_reset = 1'd0;
        fsm1_in = 5'd0;
        fsm1_write_en = 1'd0;
        fsm1_clk = 1'd0;
        fsm1_reset = 1'd0;
    end
    std_mem_d1 # (
        .IDX_SIZE(32),
        .SIZE(6),
        .WIDTH(32)
    ) M_W_x (
        .addr0(M_W_x_addr0),
        .clk(M_W_x_clk),
        .done(M_W_x_done),
        .read_data(M_W_x_read_data),
        .write_data(M_W_x_write_data),
        .write_en(M_W_x_write_en)
    );
    std_mem_d2 # (
        .D0_IDX_SIZE(32),
        .D0_SIZE(4),
        .D1_IDX_SIZE(32),
        .D1_SIZE(5),
        .WIDTH(32)
    ) y (
        .addr0(y_addr0),
        .addr1(y_addr1),
        .clk(y_clk),
        .done(y_done),
        .read_data(y_read_data),
        .write_data(y_write_data),
        .write_en(y_write_en)
    );
    std_mem_d2 # (
        .D0_IDX_SIZE(32),
        .D0_SIZE(6),
        .D1_IDX_SIZE(32),
        .D1_SIZE(5),
        .WIDTH(32)
    ) x (
        .addr0(x_addr0),
        .addr1(x_addr1),
        .clk(x_clk),
        .done(x_done),
        .read_data(x_read_data),
        .write_data(x_write_data),
        .write_en(x_write_en)
    );
    std_mem_d2 # (
        .D0_IDX_SIZE(32),
        .D0_SIZE(4),
        .D1_IDX_SIZE(32),
        .D1_SIZE(6),
        .WIDTH(32)
    ) W (
        .addr0(W_addr0),
        .addr1(W_addr1),
        .clk(W_clk),
        .done(W_done),
        .read_data(W_read_data),
        .write_data(W_write_data),
        .write_en(W_write_en)
    );
    std_mem_d2 # (
        .D0_IDX_SIZE(32),
        .D0_SIZE(4),
        .D1_IDX_SIZE(32),
        .D1_SIZE(5),
        .WIDTH(32)
    ) b (
        .addr0(b_addr0),
        .addr1(b_addr1),
        .clk(b_clk),
        .done(b_done),
        .read_data(b_read_data),
        .write_data(b_write_data),
        .write_en(b_write_en)
    );
    std_add # (
        .WIDTH(32)
    ) MAC_accumulator (
        .left(MAC_accumulator_left),
        .out(MAC_accumulator_out),
        .right(MAC_accumulator_right)
    );
    std_mult_pipe # (
        .WIDTH(32)
    ) MAC_multiplier (
        .clk(MAC_multiplier_clk),
        .done(MAC_multiplier_done),
        .go(MAC_multiplier_go),
        .left(MAC_multiplier_left),
        .out(MAC_multiplier_out),
        .right(MAC_multiplier_right)
    );
    std_reg # (
        .WIDTH(32)
    ) W_idx0 (
        .clk(W_idx0_clk),
        .done(W_idx0_done),
        .in(W_idx0_in),
        .out(W_idx0_out),
        .reset(W_idx0_reset),
        .write_en(W_idx0_write_en)
    );
    std_reg # (
        .WIDTH(32)
    ) M_W_x_idx (
        .clk(M_W_x_idx_clk),
        .done(M_W_x_idx_done),
        .in(M_W_x_idx_in),
        .out(M_W_x_idx_out),
        .reset(M_W_x_idx_reset),
        .write_en(M_W_x_idx_write_en)
    );
    std_reg # (
        .WIDTH(32)
    ) x_idx1 (
        .clk(x_idx1_clk),
        .done(x_idx1_done),
        .in(x_idx1_in),
        .out(x_idx1_out),
        .reset(x_idx1_reset),
        .write_en(x_idx1_write_en)
    );
    std_lt # (
        .WIDTH(32)
    ) W_idx1_lt (
        .left(W_idx1_lt_left),
        .out(W_idx1_lt_out),
        .right(W_idx1_lt_right)
    );
    std_reg # (
        .WIDTH(2)
    ) fsm (
        .clk(fsm_clk),
        .done(fsm_done),
        .in(fsm_in),
        .out(fsm_out),
        .reset(fsm_reset),
        .write_en(fsm_write_en)
    );
    std_add # (
        .WIDTH(2)
    ) incr (
        .left(incr_left),
        .out(incr_out),
        .right(incr_right)
    );
    std_reg # (
        .WIDTH(2)
    ) fsm0 (
        .clk(fsm0_clk),
        .done(fsm0_done),
        .in(fsm0_in),
        .out(fsm0_out),
        .reset(fsm0_reset),
        .write_en(fsm0_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cond_stored (
        .clk(cond_stored_clk),
        .done(cond_stored_done),
        .in(cond_stored_in),
        .out(cond_stored_out),
        .reset(cond_stored_reset),
        .write_en(cond_stored_write_en)
    );
    std_add # (
        .WIDTH(2)
    ) incr0 (
        .left(incr0_left),
        .out(incr0_out),
        .right(incr0_right)
    );
    std_reg # (
        .WIDTH(1)
    ) cs_wh (
        .clk(cs_wh_clk),
        .done(cs_wh_done),
        .in(cs_wh_in),
        .out(cs_wh_out),
        .reset(cs_wh_reset),
        .write_en(cs_wh_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cs_wh0 (
        .clk(cs_wh0_clk),
        .done(cs_wh0_done),
        .in(cs_wh0_in),
        .out(cs_wh0_out),
        .reset(cs_wh0_reset),
        .write_en(cs_wh0_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cs_wh1 (
        .clk(cs_wh1_clk),
        .done(cs_wh1_done),
        .in(cs_wh1_in),
        .out(cs_wh1_out),
        .reset(cs_wh1_reset),
        .write_en(cs_wh1_write_en)
    );
    std_reg # (
        .WIDTH(5)
    ) fsm1 (
        .clk(fsm1_clk),
        .done(fsm1_done),
        .in(fsm1_in),
        .out(fsm1_out),
        .reset(fsm1_reset),
        .write_en(fsm1_write_en)
    );
    assign MAC_accumulator_left =
     ~M_W_x_idx_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd7 & go | fsm_out == 2'd1 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? M_W_x_idx_out :
     ~W_idx0_done & cs_wh_out & fsm1_out == 5'd14 & go ? W_idx0_out :
     ~x_idx1_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd12 & go ? x_idx1_out :
     fsm_out == 2'd0 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | ~y_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd11 & go ? y_read_data : 32'd0;
    assign MAC_accumulator_right =
     fsm_out == 2'd0 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? M_W_x_read_data :
     ~W_idx0_done & cs_wh_out & fsm1_out == 5'd14 & go | ~M_W_x_idx_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd7 & go | ~x_idx1_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd12 & go | fsm_out == 2'd1 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? 32'd1 :
     ~y_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd11 & go ? b_read_data : 32'd0;
    assign MAC_multiplier_clk =
     1'b1 ? clk : 1'd0;
    assign MAC_multiplier_go =
     ~MAC_multiplier_done & ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? 1'd1 : 1'd0;
    assign MAC_multiplier_left =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? W_read_data : 32'd0;
    assign MAC_multiplier_right =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? x_read_data : 32'd0;
    assign M_W_x_addr0 =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go | fsm_out == 2'd0 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? M_W_x_idx_out : 32'd0;
    assign M_W_x_clk =
     1'b1 ? clk : 1'd0;
    assign M_W_x_write_data =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? MAC_multiplier_out : 32'd0;
    assign M_W_x_write_en =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? MAC_multiplier_done : 1'd0;
    assign M_W_x_idx_clk =
     1'b1 ? clk : 1'd0;
    assign M_W_x_idx_in =
     ~M_W_x_idx_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd7 & go | fsm_out == 2'd1 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? MAC_accumulator_out :
     ~M_W_x_idx_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd4 & go | ~M_W_x_idx_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd9 & go ? 32'd0 : 32'd0;
    assign M_W_x_idx_write_en =
     ~M_W_x_idx_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd7 & go | fsm_out == 2'd1 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | ~M_W_x_idx_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd4 & go | ~M_W_x_idx_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd9 & go ? 1'd1 : 1'd0;
    assign W_addr0 =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? W_idx0_out : 32'd0;
    assign W_addr1 =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? M_W_x_idx_out : 32'd0;
    assign W_clk =
     1'b1 ? clk : 1'd0;
    assign W_idx0_clk =
     1'b1 ? clk : 1'd0;
    assign W_idx0_in =
     ~W_idx0_done & cs_wh_out & fsm1_out == 5'd14 & go ? MAC_accumulator_out :
     ~W_idx0_done & fsm1_out == 5'd0 & go ? 32'd0 : 32'd0;
    assign W_idx0_write_en =
     ~W_idx0_done & cs_wh_out & fsm1_out == 5'd14 & go | ~W_idx0_done & fsm1_out == 5'd0 & go ? 1'd1 : 1'd0;
    assign W_idx1_lt_left =
     cs_wh0_out & cs_wh_out & fsm1_out == 5'd5 & go | fsm0_out < 2'd1 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? M_W_x_idx_out :
     fsm1_out == 5'd1 & go ? W_idx0_out :
     cs_wh_out & fsm1_out == 5'd3 & go ? x_idx1_out : 32'd0;
    assign W_idx1_lt_right =
     fsm1_out == 5'd1 & go ? 32'd4 :
     cs_wh_out & fsm1_out == 5'd3 & go ? 32'd5 :
     cs_wh0_out & cs_wh_out & fsm1_out == 5'd5 & go | fsm0_out < 2'd1 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? 32'd6 : 32'd0;
    assign done =
     fsm1_out == 5'd16 ? 1'd1 : 1'd0;
    assign b_addr0 =
     ~y_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd11 & go ? W_idx0_out : 32'd0;
    assign b_addr1 =
     ~y_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd11 & go ? x_idx1_out : 32'd0;
    assign b_clk =
     1'b1 ? clk : 1'd0;
    assign cond_stored_clk =
     1'b1 ? clk : 1'd0;
    assign cond_stored_in =
     fsm0_out < 2'd1 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? W_idx1_lt_out : 1'd0;
    assign cond_stored_reset =
     1'b1 ? reset : 1'd0;
    assign cond_stored_write_en =
     fsm0_out < 2'd1 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? 1'd1 : 1'd0;
    assign cs_wh_clk =
     1'b1 ? clk : 1'd0;
    assign cs_wh_in =
     fsm1_out == 5'd1 & go ? W_idx1_lt_out :
     fsm1_out == 5'd16 & go ? 1'd0 : 1'd0;
    assign cs_wh_reset =
     1'b1 ? reset : 1'd0;
    assign cs_wh_write_en =
     fsm1_out == 5'd1 & go | fsm1_out == 5'd16 & go ? 1'd1 : 1'd0;
    assign cs_wh0_clk =
     1'b1 ? clk : 1'd0;
    assign cs_wh0_in =
     cs_wh_out & fsm1_out == 5'd3 & go ? W_idx1_lt_out :
     cs_wh_out & fsm1_out == 5'd14 & go ? 1'd0 : 1'd0;
    assign cs_wh0_reset =
     1'b1 ? reset : 1'd0;
    assign cs_wh0_write_en =
     cs_wh_out & fsm1_out == 5'd3 & go | cs_wh_out & fsm1_out == 5'd14 & go ? 1'd1 : 1'd0;
    assign cs_wh1_clk =
     1'b1 ? clk : 1'd0;
    assign cs_wh1_in =
     cs_wh0_out & cs_wh_out & fsm1_out == 5'd5 & go ? W_idx1_lt_out :
     cs_wh0_out & cs_wh_out & fsm1_out == 5'd9 & go ? 1'd0 : 1'd0;
    assign cs_wh1_reset =
     1'b1 ? reset : 1'd0;
    assign cs_wh1_write_en =
     cs_wh0_out & cs_wh_out & fsm1_out == 5'd5 & go | cs_wh0_out & cs_wh_out & fsm1_out == 5'd9 & go ? 1'd1 : 1'd0;
    assign fsm_clk =
     1'b1 ? clk : 1'd0;
    assign fsm_in =
     fsm_out == 2'd2 ? 2'd0 :
     fsm_out != 2'd2 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? incr_out : 2'd0;
    assign fsm_reset =
     1'b1 ? reset : 1'd0;
    assign fsm_write_en =
     fsm_out != 2'd2 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | fsm_out == 2'd2 ? 1'd1 : 1'd0;
    assign fsm0_clk =
     1'b1 ? clk : 1'd0;
    assign fsm0_in =
     fsm0_out == 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | fsm0_out == 2'd1 & ~cond_stored_out ? 2'd0 :
     fsm0_out != 2'd3 & (cond_stored_out | fsm0_out < 2'd1) & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? incr0_out : 2'd0;
    assign fsm0_reset =
     1'b1 ? reset : 1'd0;
    assign fsm0_write_en =
     fsm0_out != 2'd3 & (cond_stored_out | fsm0_out < 2'd1) & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | fsm0_out == 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | fsm0_out == 2'd1 & ~cond_stored_out ? 1'd1 : 1'd0;
    assign fsm1_clk =
     1'b1 ? clk : 1'd0;
    assign fsm1_in =
     fsm1_out == 5'd16 ? 5'd0 :
     fsm1_out == 5'd9 & M_W_x_idx_done & cs_wh0_out & cs_wh_out & go ? 5'd10 :
     fsm1_out == 5'd10 & fsm0_out == 2'd1 & ~cond_stored_out & cs_wh0_out & cs_wh_out & go ? 5'd11 :
     fsm1_out == 5'd11 & y_done & cs_wh0_out & cs_wh_out & go ? 5'd12 :
     fsm1_out == 5'd12 & x_idx1_done & cs_wh0_out & cs_wh_out & go ? 5'd13 :
     fsm1_out == 5'd4 & ~cs_wh0_out & cs_wh_out & go ? 5'd14 :
     fsm1_out == 5'd14 & W_idx0_done & cs_wh_out & go ? 5'd15 :
     fsm1_out == 5'd2 & ~cs_wh_out & go ? 5'd16 :
     fsm1_out == 5'd0 & W_idx0_done & go | fsm1_out == 5'd15 & cs_wh_out & go ? 5'd1 :
     fsm1_out == 5'd1 & 1'b1 & go ? 5'd2 :
     fsm1_out == 5'd2 & x_idx1_done & cs_wh_out & go | fsm1_out == 5'd13 & cs_wh0_out & cs_wh_out & go ? 5'd3 :
     fsm1_out == 5'd3 & 1'b1 & go ? 5'd4 :
     fsm1_out == 5'd4 & M_W_x_idx_done & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd8 & cs_wh1_out & cs_wh0_out & cs_wh_out & go ? 5'd5 :
     fsm1_out == 5'd5 & 1'b1 & go ? 5'd6 :
     fsm1_out == 5'd6 & M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & go ? 5'd7 :
     fsm1_out == 5'd7 & M_W_x_idx_done & cs_wh1_out & cs_wh0_out & cs_wh_out & go ? 5'd8 :
     fsm1_out == 5'd6 & ~cs_wh1_out & cs_wh0_out & cs_wh_out & go ? 5'd9 : 5'd0;
    assign fsm1_reset =
     1'b1 ? reset : 1'd0;
    assign fsm1_write_en =
     fsm1_out == 5'd0 & W_idx0_done & go | fsm1_out == 5'd1 & 1'b1 & go | fsm1_out == 5'd2 & x_idx1_done & cs_wh_out & go | fsm1_out == 5'd3 & 1'b1 & go | fsm1_out == 5'd4 & M_W_x_idx_done & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd5 & 1'b1 & go | fsm1_out == 5'd6 & M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd7 & M_W_x_idx_done & cs_wh1_out & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd8 & cs_wh1_out & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd6 & ~cs_wh1_out & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd9 & M_W_x_idx_done & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd10 & fsm0_out == 2'd1 & ~cond_stored_out & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd11 & y_done & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd12 & x_idx1_done & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd13 & cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd4 & ~cs_wh0_out & cs_wh_out & go | fsm1_out == 5'd14 & W_idx0_done & cs_wh_out & go | fsm1_out == 5'd15 & cs_wh_out & go | fsm1_out == 5'd2 & ~cs_wh_out & go | fsm1_out == 5'd16 ? 1'd1 : 1'd0;
    assign incr_left =
     cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? 2'd1 : 2'd0;
    assign incr_right =
     cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? fsm_out : 2'd0;
    assign incr0_left =
     ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? fsm0_out : 2'd0;
    assign incr0_right =
     ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go ? 2'd1 : 2'd0;
    assign x_addr0 =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? M_W_x_idx_out : 32'd0;
    assign x_addr1 =
     ~M_W_x_done & cs_wh1_out & cs_wh0_out & cs_wh_out & fsm1_out == 5'd6 & go ? x_idx1_out : 32'd0;
    assign x_clk =
     1'b1 ? clk : 1'd0;
    assign x_idx1_clk =
     1'b1 ? clk : 1'd0;
    assign x_idx1_in =
     ~x_idx1_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd12 & go ? MAC_accumulator_out :
     ~x_idx1_done & cs_wh_out & fsm1_out == 5'd2 & go ? 32'd0 : 32'd0;
    assign x_idx1_write_en =
     ~x_idx1_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd12 & go | ~x_idx1_done & cs_wh_out & fsm1_out == 5'd2 & go ? 1'd1 : 1'd0;
    assign y_addr0 =
     fsm_out == 2'd0 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | ~y_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd11 & go ? W_idx0_out : 32'd0;
    assign y_addr1 =
     fsm_out == 2'd0 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | ~y_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd11 & go ? x_idx1_out : 32'd0;
    assign y_clk =
     1'b1 ? clk : 1'd0;
    assign y_write_data =
     fsm_out == 2'd0 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | ~y_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd11 & go ? MAC_accumulator_out : 32'd0;
    assign y_write_en =
     fsm_out == 2'd0 & cond_stored_out & fsm0_out >= 2'd1 & fsm0_out < 2'd3 & ~(fsm0_out == 2'd1 & ~cond_stored_out) & cs_wh0_out & cs_wh_out & fsm1_out == 5'd10 & go | ~y_done & cs_wh0_out & cs_wh_out & fsm1_out == 5'd11 & go ? 1'd1 : 1'd0;
endmodule