/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tiny Tapeout wrapper: top module name tt_um_example for template/testbench.
 * Core logic: tt_um_xor_learner. VGA CA step archived in src/legacy/tt_um_vga_ca.v
 */

`default_nettype none

module tt_um_example (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

  tt_um_xor_learner u_core (
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
