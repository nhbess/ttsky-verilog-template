/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * XOR learner — strict hill-climb only (new_score > old_score).
 * Core: tt_um_xor_learner #(.PLATEAU_ESCAPE(0))
 */

`default_nettype none

module tt_um_xor_learner_strict (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

  tt_um_xor_learner #(
      .PLATEAU_ESCAPE(0)
  ) core (
      .ui_in  (ui_in),
      .uo_out (uo_out),
      .uio_in (uio_in),
      .uio_out(uio_out),
      .uio_oe (uio_oe),
      .ena    (ena),
      .clk    (clk),
      .rst_n  (rst_n)
  );

endmodule
