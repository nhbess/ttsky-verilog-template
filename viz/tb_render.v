`default_nettype none
`timescale 1ns / 1ps

/* Standalone bench: drive tt_um_example, dump VCD for viz/render_vga.py (no cocotb). */
module tb_render;

  reg clk = 0;
  reg rst_n = 0;
  reg ena = 1;
  reg [7:0] ui_in = 0;
  reg [7:0] uio_in = 0;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;

  // ~25 MHz
  always #20 clk = ~clk;

  tt_um_example dut (
      .ui_in(ui_in),
      .uo_out(uo_out),
      .uio_in(uio_in),
      .uio_out(uio_out),
      .uio_oe(uio_oe),
      .ena(ena),
      .clk(clk),
      .rst_n(rst_n)
  );

  initial begin
    $dumpfile("waves.vcd");
    $dumpvars(0, tb_render);
    #200 rst_n = 1;
    // ~36 ms wall time at 1ns precision: enough for >1 VGA frame at 25 MHz
    #36_000_000;
    $finish;
  end

endmodule
