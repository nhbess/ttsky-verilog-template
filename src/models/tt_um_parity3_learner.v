/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * 3-bit parity: h1=g0(a,b), t=g1(h1,c), y=g2(h1,t). Target a^b^c (XOR chain).
 * 8 training rows, score 0..8. Same trainer FSM as XOR learner.
 *
 * ui_in[0]=a, ui_in[1]=b, ui_in[2]=c, ui_in[3]=train_enable (differs from XOR).
 *
 * PLATEAU_ESCAPE / PLATEAU_AND_MASK: same spirit as XOR; project_parity3 sets
 * .PLATEAU_ESCAPE(1); use 0 in the instance for strict-only compare.
 */

`default_nettype none

module tt_um_parity3_learner #(
    parameter PLATEAU_ESCAPE = 1,
    // Bitwise AND mask for tie test: 3 -> ~1/4, 7 -> ~1/8 (parity default 3).
    parameter [3:0] PLATEAU_AND_MASK = 4'd3
) (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

  localparam S_IDLE      = 4'd1;
  localparam S_INIT_UNIT = 4'd2;
  localparam S_OLD_CLEAR = 4'd3;
  localparam S_OLD_ACC   = 4'd4;
  localparam S_PROPOSE   = 4'd5;
  localparam S_NEW_CLEAR = 4'd6;
  localparam S_NEW_ACC   = 4'd7;
  localparam S_COMPARE   = 4'd8;

  function [15:0] lfsr16_step_i;
    input [15:0] s;
    reg [15:0] t;
    begin
      t = (s == 16'h0000) ? 16'hACE1 : s;
      if (t[0])
        lfsr16_step_i = {1'b0, t[15:1]} ^ 16'hB400;
      else
        lfsr16_step_i = {1'b0, t[15:1]};
    end
  endfunction

  function [27:0] init_chain;
    input unused;
    reg [15:0] t;
    begin
      t = 16'hACE1;
      t = lfsr16_step_i(t);
      init_chain[3:0] = t[3:0];
      t = lfsr16_step_i(t);
      init_chain[7:4] = t[3:0];
      t = lfsr16_step_i(t);
      init_chain[11:8] = t[3:0];
      init_chain[27:12] = t;
    end
  endfunction

  wire [27:0] init_pack = init_chain(1'b0);

  reg [3:0] g0, g1, g2;
  reg [1:0] p0, p1, p2;
  reg [15:0] lfsr;
  reg [3:0] state;
  reg [1:0] unit_sel;
  reg [2:0] sample_idx;
  reg [3:0] old_score;
  reg [3:0] new_score;
  reg [3:0] old_gate;
  reg [3:0] trial_gate;

  wire adv = ena & ui_in[3];

  wire [1:0] ab_ui = {ui_in[0], ui_in[1]};
  wire       h1f = g0[ab_ui];
  wire [1:0] h1c_ui = {h1f, ui_in[2]};
  wire       t_ui = g1[h1c_ui];
  wire       y   = g2[{h1f, t_ui}];

  wire [2:0] sidx = sample_idx;
  wire       ta = sidx[2];
  wire       tb = sidx[1];
  wire       tc = sidx[0];
  wire [1:0] ab_t = {ta, tb};
  wire       h1o = g0[ab_t];
  wire [1:0] h1c_t = {h1o, tc};
  wire       to = g1[h1c_t];
  wire       yo  = g2[{h1o, to}];
  wire       tgt = ta ^ tb ^ tc;
  wire       match_old = yo == tgt;

  wire [3:0] g0t = (unit_sel == 2'd0) ? trial_gate : g0;
  wire [3:0] g1t = (unit_sel == 2'd1) ? trial_gate : g1;
  wire [3:0] g2t = (unit_sel == 2'd2) ? trial_gate : g2;
  wire       h1n = g0t[ab_t];
  wire [1:0] h1c_n = {h1n, tc};
  wire       tn = g1t[h1c_n];
  wire       yn  = g2t[{h1n, tn}];
  wire       match_new = yn == tgt;

  wire [15:0] t1 = lfsr16_step_i(lfsr);
  wire [1:0]  us_next = (t1[1:0] == 2'd3) ? 2'd0 : t1[1:0];
  wire [15:0] t2 = lfsr16_step_i(t1);
  wire [1:0]  pu_r = (us_next == 2'd0) ? p0 : (us_next == 2'd1) ? p1 : p2;
  wire [2:0]  thr = {1'b0, pu_r} + 3'd1;
  wire        allow_in = ({1'b0, t2[1:0]}) < thr;
  wire [3:0]  g_pick = (us_next == 2'd0) ? g0 : (us_next == 2'd1) ? g1 : g2;

  wire [15:0] t3 = lfsr16_step_i(lfsr);
  wire [3:0]  cand_raw = t3[3:0];
  wire [3:0]  cand_fix = (cand_raw == old_gate) ? (cand_raw ^ 4'h1) : cand_raw;

  wire [15:0] t_cmp = lfsr16_step_i(lfsr);
  wire        rare_tie = ((t_cmp[3:0] & PLATEAU_AND_MASK) == 4'd0);
  wire        accept_strict = (new_score > old_score);
  wire        accept_plateau = (new_score > old_score) ||
      ((new_score == old_score) && rare_tie);
  wire        accept_cmp = PLATEAU_ESCAPE ? accept_plateau : accept_strict;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      g0 <= init_pack[3:0];
      g1 <= init_pack[7:4];
      g2 <= init_pack[11:8];
      lfsr <= init_pack[27:12];
      p0 <= 2'd3;
      p1 <= 2'd3;
      p2 <= 2'd3;
      unit_sel <= 2'd0;
      sample_idx <= 3'd0;
      old_score <= 4'd0;
      new_score <= 4'd0;
      old_gate <= 4'd0;
      trial_gate <= 4'd0;
      state <= S_IDLE;
    end else if (adv) begin
      case (state)
        S_IDLE: begin
          state <= S_INIT_UNIT;
        end

        S_INIT_UNIT: begin
          lfsr <= t2;
          unit_sel <= us_next;
          if (!allow_in)
            state <= S_IDLE;
          else begin
            old_gate <= g_pick;
            state <= S_OLD_CLEAR;
          end
        end

        S_OLD_CLEAR: begin
          sample_idx <= 3'd0;
          old_score <= 4'd0;
          state <= S_OLD_ACC;
        end

        S_OLD_ACC: begin
          if (match_old)
            old_score <= old_score + 4'd1;
          if (sample_idx == 3'd7)
            state <= S_PROPOSE;
          else
            sample_idx <= sample_idx + 3'd1;
        end

        S_PROPOSE: begin
          lfsr <= t3;
          trial_gate <= cand_fix;
          state <= S_NEW_CLEAR;
        end

        S_NEW_CLEAR: begin
          sample_idx <= 3'd0;
          new_score <= 4'd0;
          state <= S_NEW_ACC;
        end

        S_NEW_ACC: begin
          if (match_new)
            new_score <= new_score + 4'd1;
          if (sample_idx == 3'd7)
            state <= S_COMPARE;
          else
            sample_idx <= sample_idx + 3'd1;
        end

        S_COMPARE: begin
          if (PLATEAU_ESCAPE)
            lfsr <= t_cmp;
          if (accept_cmp) begin
            case (unit_sel)
              2'd0: begin
                g0 <= trial_gate;
                if (p0 != 2'd0)
                  p0 <= p0 - 2'd1;
              end
              2'd1: begin
                g1 <= trial_gate;
                if (p1 != 2'd0)
                  p1 <= p1 - 2'd1;
              end
              default: begin
                g2 <= trial_gate;
                if (p2 != 2'd0)
                  p2 <= p2 - 2'd1;
              end
            endcase
          end else begin
            case (unit_sel)
              2'd0: if (p0 != 2'd3) p0 <= p0 + 2'd1;
              2'd1: if (p1 != 2'd3) p1 <= p1 + 2'd1;
              default: if (p2 != 2'd3) p2 <= p2 + 2'd1;
            endcase
          end
          state <= S_IDLE;
        end

        default: begin
          state <= S_IDLE;
        end
      endcase
    end
  end

  assign uo_out[0]   = y;
  assign uo_out[3:1] = state[2:0];
  assign uo_out[7:4] = old_score[3:0];

  assign uio_out = 8'd0;
  assign uio_oe  = 8'd0;

  wire _unused_ok = &{ena, ui_in[7:4], uio_in};
endmodule
