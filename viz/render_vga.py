#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Rebuild a VGA frame from waves.vcd (TinyVGA uo_out packing, znah tt09-vga-ca style).

  python viz/render_vga.py

Requires: iverilog + vvp on PATH. Python: pip install -r test/requirements.txt (from repo root).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _compile_and_run(viz_dir: Path, root: Path) -> int:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        print("ERROR: iverilog and vvp must be on PATH.", file=sys.stderr)
        return 1

    vvp = viz_dir / "sim_render.vvp"
    src = [
        root / "src" / "hvsync_generator.v",
        root / "src" / "project.v",
        viz_dir / "tb_render.v",
    ]
    cmd = [
        "iverilog",
        "-g2012",
        "-DSIM",
        "-I",
        str(root / "src"),
        "-s",
        "tb_render",
        "-o",
        str(vvp),
        *[str(p) for p in src],
    ]
    subprocess.run(cmd, check=True, cwd=viz_dir)
    subprocess.run(["vvp", str(vvp.name)], check=True, cwd=viz_dir)
    return 0


def _vcd_to_png(vcd_path: Path, png_path: Path) -> None:
    import numpy as np
    from PIL import Image
    import vcdvcd

    vcd = vcdvcd.VCDVCD(str(vcd_path))
    uo = vcd["tb_render.uo_out[7:0]"]
    clk_sig = vcd["tb_render.clk"]

    pad = 48 * 2
    w, h = 640 + pad, 480 + pad
    screen = np.zeros((h, w), dtype=np.uint8)
    x = y = 0
    prev = 0

    for _t, clk_val in clk_sig.tv:
        if clk_val != "0":
            continue
        raw = uo[_t].replace("x", "0")
        val = int(raw, 2)
        if x < w and y < h:
            screen[y, x] = val
        x += 1
        if (prev & 0x80) and not (val & 0x80):
            x, y = 0, y + 1
        if (prev & 0x08) and not (val & 0x08):
            x, y = 0, 0
        prev = val

    bits = np.unpackbits(screen, bitorder="little").reshape(h, w, 8)
    rgb = bits[..., :3] * 170 + bits[..., 4:7] * 85
    Image.fromarray(rgb.astype(np.uint8)).save(png_path)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    viz_dir = root / "viz"
    vcd_path = viz_dir / "waves.vcd"
    png_path = viz_dir / "frame.png"

    try:
        if _compile_and_run(viz_dir, root):
            return 1
        if not vcd_path.is_file():
            print(f"ERROR: expected {vcd_path}", file=sys.stderr)
            return 1
        _vcd_to_png(vcd_path, png_path)
    except subprocess.CalledProcessError as e:
        print(e, file=sys.stderr)
        return e.returncode or 1
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        print("Install viz deps: pip install vcdvcd numpy pillow", file=sys.stderr)
        return 1

    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
